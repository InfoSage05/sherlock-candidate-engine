/**
 * Sherlock — Candidate Identification Engine
 * Client-side application logic
 *
 * Manages WebSocket connections, DOM rendering, and UI interactions.
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

const API_BASE = '';  // same origin
const WS_BASE  = `ws://${location.host}`;

// ============================================================================
// STATE
// ============================================================================

const appState = {
    scenarioLoaded: false,
    isPlaying:      false,
    playbackSpeed:  1.0,
    currentTab:     'monitor',
    snapshot:       null,
    scenarios:      [],
    notes:          [],
    ws:             null,
    wsReconnectTimer: null,
    liveMode:       false,
    liveRunning:    false,
    liveWs:         null,
    liveVideoUrl:   null,
    liveReconnectAttempts: 0,
    maxLiveReconnectAttempts: 10,
};

// ============================================================================
// DOM HELPERS
// ============================================================================

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);
const el = (tag, attrs = {}, children = []) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (k === 'className') node.className = v;
        else if (k === 'textContent') node.textContent = v;
        else if (k === 'innerHTML') node.innerHTML = v;
        else if (k.startsWith('on')) node.addEventListener(k.slice(2).toLowerCase(), v);
        else node.setAttribute(k, v);
    }
    for (const child of children) {
        if (typeof child === 'string') node.appendChild(document.createTextNode(child));
        else if (child) node.appendChild(child);
    }
    return node;
};

function probColor(p) {
    if (p >= 0.7) return '#66bb6a';
    if (p >= 0.4) return '#ffd54f';
    return '#ef5350';
}

function statusColor(status) {
    const map = { identified: 'green', ambiguous: 'yellow', no_participants: 'muted' };
    return map[status] || 'muted';
}

function formatSeconds(s) {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
}

// ============================================================================
// API CALLS
// ============================================================================

async function api(path, opts = {}) {
    const url = `${API_BASE}${path}`;
    const res = await fetch(url, {
        headers: { 'Content-Type': 'application/json' },
        ...opts,
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || res.statusText);
    }
    return res.json();
}

async function loadScenarios() {
    try {
        appState.scenarios = await api('/api/scenarios');
        renderScenarioSelect();
        renderScenarioGrid();
    } catch (e) {
        console.error('Failed to load scenarios:', e);
    }
}

async function loadScenario(id) {
    try {
        // Clean up any active live session first.
        if (appState.liveMode) {
            await stopLive();
        }

        await api('/api/replay/load', {
            method: 'POST',
            body: JSON.stringify({ scenario_id: id }),
        });
        appState.scenarioLoaded = true;
        appState.isPlaying = false;
        appState.liveMode = false;
        showDashboard();
        // Get initial snapshot
        const snap = await api('/api/replay/snapshot');
        updateUI(snap);
        connectReplayWS();
    } catch (e) {
        console.error('Failed to load scenario:', e);
        alert(`Error: ${e.message}`);
    }
}

// ============================================================================
// WEBSOCKET
// ============================================================================

function connectReplayWS() {
    if (appState.ws) {
        appState.ws.close();
        appState.ws = null;
    }

    const ws = new WebSocket(`${WS_BASE}/ws/replay`);
    appState.ws = ws;

    ws.onopen = () => {
        console.log('WS connected');
        if (appState.wsReconnectTimer) {
            clearTimeout(appState.wsReconnectTimer);
            appState.wsReconnectTimer = null;
        }
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            appState.snapshot = data;
            updateUI(data);

            if (data.playback_state === 'finished') {
                appState.isPlaying = false;
                updatePlayButton();
            }
        } catch (e) {
            console.error('WS message parse error:', e);
        }
    };

    ws.onclose = () => {
        console.log('WS disconnected');
        appState.ws = null;
        // Auto-reconnect if scenario is loaded and not in live mode
        if (appState.scenarioLoaded && !appState.liveMode) {
            appState.wsReconnectTimer = setTimeout(connectReplayWS, 2000);
        }
    };

    ws.onerror = (e) => console.error('WS error:', e);
}

function wsSend(cmd) {
    if (appState.ws && appState.ws.readyState === WebSocket.OPEN) {
        appState.ws.send(JSON.stringify(cmd));
    }
}

// ============================================================================
// LIVE A/V ANALYSIS
// ============================================================================

async function startLive() {
    const fileInput = $('#liveUploadInput');
    const youtubeInput = $('#liveYoutubeInput');
    const pathInput = $('#livePathInput');
    const candidateInput = $('#liveCandidateInput');

    const youtubeUrl = youtubeInput?.value?.trim() || '';
    const filePath = pathInput?.value?.trim() || '';
    const candidateName = candidateInput?.value?.trim() || 'Candidate';
    const file = fileInput?.files?.[0];

    if (!file && !youtubeUrl && !filePath) {
        showToast('Please upload a video, paste a YouTube link, or enter a file path.', true);
        return;
    }

    // Stop any previous session so the new video always takes over.
    if (appState.liveMode || appState.liveRunning) {
        await stopLive();
        // Brief pause to let the backend release the file/socket.
        await new Promise(r => setTimeout(r, 500));
    }

    let body = { candidate_name: candidateName };

    const startBtn = $('#liveStartBtn');
    setLiveLoading(true);

    try {
        if (file) {
            const form = new FormData();
            form.append('file', file);
            const uploadRes = await fetch(`${API_BASE}/api/live/upload`, {
                method: 'POST',
                body: form,
            });
            if (!uploadRes.ok) {
                const err = await uploadRes.json().catch(() => ({}));
                throw new Error(err.detail || 'Upload failed');
            }
            const uploadData = await uploadRes.json();
            body.file_path = uploadData.file_path;
        } else if (youtubeUrl) {
            body.youtube_url = youtubeUrl;
        } else {
            body.file_path = filePath;
        }

        const startRes = await api('/api/live/start', {
            method: 'POST',
            body: JSON.stringify(body),
        });

        appState.liveMode = true;
        appState.liveRunning = true;
        appState.liveVideoUrl = startRes.video_url;
        appState.liveReconnectAttempts = 0;

        showDashboard();
        setupLiveVideo(startRes.video_url, startRes.video_title);
        updateLiveStatusRow(true, startRes.video_title);
        showProcessingBanner(true);
        connectLiveWS();

        showToast('Live analysis started 🎥');
    } catch (e) {
        console.error('Failed to start live analysis:', e);
        showToast('Error: ' + e.message, true);
    } finally {
        setLiveLoading(false);
    }
}

function setLiveLoading(loading) {
    const btn = $('#liveStartBtn');
    if (!btn) return;
    btn.disabled = loading;
    btn.textContent = loading ? '⏳ Starting…' : '▶ Start Analysis';
}

async function stopLive() {
    try {
        await api('/api/live/stop', { method: 'POST' });
    } catch (e) {
        console.error('Failed to stop live analysis:', e);
    }

    const video = $('#liveVideo');
    if (video) {
        video.pause();
        video.removeAttribute('src');
        video.load();
    }

    if (appState.liveWs) {
        appState.liveWs.close();
        appState.liveWs = null;
    }

    appState.liveMode = false;
    appState.liveRunning = false;
    appState.liveVideoUrl = null;
    appState.liveReconnectAttempts = 0;

    updateLiveStatusRow(false);
    setLiveConnectionStatus(false, 'idle');
    setLiveLoading(false);
    showProcessingBanner(false);
    showLanding();
    showToast('Live analysis stopped');
}

function setupLiveVideo(url, title) {
    const panel = $('#liveVideoPanel');
    const video = $('#liveVideo');
    if (!panel || !video) return;

    panel.style.display = 'block';
    // Cache-bust so the browser fetches the newly selected video
    // instead of reusing the previous /api/live/video payload.
    const cacheBusted = `${url}${url.includes('?') ? '&' : '?'}_t=${Date.now()}`;
    video.src = cacheBusted;
    video.load();

    // Play once metadata is loaded so duration is available.
    video.onloadedmetadata = () => {
        video.play().catch(err => console.warn('Auto-play blocked:', err));
    };
}

function connectLiveWS() {
    if (appState.liveWs) {
        appState.liveWs.close();
        appState.liveWs = null;
    }

    if (appState.liveReconnectAttempts >= appState.maxLiveReconnectAttempts) {
        showToast('Lost connection to analysis backend. Please refresh.', true);
        setLiveConnectionStatus(false, 'disconnected');
        return;
    }

    const ws = new WebSocket(`${WS_BASE}/ws/live`);
    appState.liveWs = ws;
    appState.liveReconnectAttempts += 1;

    ws.onopen = () => {
        console.log('Live WS connected');
        appState.liveReconnectAttempts = 0;
        setLiveConnectionStatus(true, 'live');
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.state === 'idle' && !appState.liveRunning) return;
            appState.snapshot = data;
            updateUI(data);
            updateLiveVideoScores(data);
        } catch (e) {
            console.error('Live WS message parse error:', e);
        }
    };

    ws.onclose = () => {
        console.log('Live WS disconnected');
        appState.liveWs = null;
        setLiveConnectionStatus(false, appState.liveRunning ? 'reconnecting' : 'idle');
        if (appState.liveRunning) {
            const delay = Math.min(10000, 2000 + appState.liveReconnectAttempts * 1000);
            setTimeout(connectLiveWS, delay);
        }
    };

    ws.onerror = (e) => {
        console.error('Live WS error:', e);
        setLiveConnectionStatus(false, 'error');
    };
}

function setLiveConnectionStatus(connected, state) {
    const badge = $('#liveConnectionBadge');
    if (!badge) return;
    badge.style.display = 'inline-flex';
    if (connected) {
        badge.className = 'connection-badge connected';
        badge.textContent = '● LIVE';
    } else if (state === 'reconnecting') {
        badge.className = 'connection-badge reconnecting';
        badge.textContent = '↻ Reconnecting';
    } else if (state === 'error') {
        badge.className = 'connection-badge error';
        badge.textContent = '✕ Error';
    } else {
        badge.className = 'connection-badge idle';
        badge.textContent = '○ Idle';
    }
}

function updateLiveStatusRow(running, title = '') {
    const row = $('#liveStatusRow');
    const dot = $('#liveDot');
    const text = $('#liveStatusText');
    const titleEl = $('#liveVideoTitle');
    if (!row || !dot || !text) return;

    row.style.display = 'flex';
    if (running) {
        dot.classList.remove('idle');
        text.textContent = 'Analysing';
        titleEl.textContent = title ? `— ${title}` : '';
    } else {
        dot.classList.add('idle');
        text.textContent = 'Idle';
        titleEl.textContent = '';
    }
}

function showProcessingBanner(show) {
    const banner = $('#processingBanner');
    if (banner) banner.style.display = show ? 'flex' : 'none';
}

function updateLiveVideoScores(data) {
    if (!data || data.status === 'no_data') return;

    // Hide processing banner once real evidence starts arriving.
    if ((data.evidence_count || 0) > 1) {
        showProcessingBanner(false);
    }

    const participant = data.participants?.[0];
    const identity = participant ? participant.identity_probability : (data.top_candidate_probability || 0);
    const authenticity = participant ? participant.authenticity_probability : 0.5;

    const idEl = $('#lvIdentity');
    const idBar = $('#lvIdentityBar');
    const authEl = $('#lvAuthenticity');
    const authBar = $('#lvAuthenticityBar');
    const verdictEl = $('#lvVerdict');
    const elapsedEl = $('#lvElapsed');

    if (idEl) idEl.textContent = `${(identity * 100).toFixed(1)}%`;
    if (idBar) {
        idBar.style.width = `${identity * 100}%`;
        idBar.style.background = probColor(identity);
    }

    if (authEl) authEl.textContent = `${(authenticity * 100).toFixed(1)}%`;
    if (authBar) {
        authBar.style.width = `${authenticity * 100}%`;
        authBar.style.background = probColor(authenticity);
    }

    const verdict = computeVerdict(authenticity);
    if (verdictEl) {
        verdictEl.textContent = verdict.label;
        verdictEl.style.color = verdict.color;
    }

    renderVerdictCard(verdict, data.verdict_reasons || []);
    renderFlaggedSegments(data.flagged_segments || []);

    if (elapsedEl) elapsedEl.textContent = formatSeconds(data.elapsed_seconds || 0);
}

function computeVerdict(authenticity) {
    if (authenticity >= 0.65) {
        return { label: 'Genuine', className: 'genuine', icon: '🛡️', color: 'var(--accent-green)' };
    }
    if (authenticity >= 0.35) {
        return { label: 'Suspicious', className: 'suspicious', icon: '⚠️', color: 'var(--accent-yellow)' };
    }
    return { label: 'Likely Cheating', className: 'cheating', icon: '🚩', color: 'var(--accent-red)' };
}

function renderVerdictCard(verdict, reasons) {
    const card = $('#verdictCard');
    const icon = $('#verdictIcon');
    const title = $('#verdictTitle');
    const reasonsEl = $('#verdictReasons');
    if (!card || !icon || !title || !reasonsEl) return;

    card.style.display = 'block';
    card.className = `verdict-card ${verdict.className}`;
    icon.textContent = verdict.icon;
    title.textContent = verdict.label;
    title.style.color = verdict.color;

    if (reasons.length === 0) {
        reasonsEl.innerHTML = '<p>No authenticity concerns detected yet.</p>';
    } else {
        reasonsEl.innerHTML = '<ul>' +
            reasons.slice(-5).map(r => `<li>${escapeHtml(r)}</li>`).join('') +
            '</ul>';
    }
}

function renderFlaggedSegments(segments) {
    const panel = $('#flaggedSegmentsPanel');
    const list = $('#flaggedSegmentsList');
    if (!panel || !list) return;

    if (segments.length === 0) {
        panel.style.display = 'none';
        return;
    }

    panel.style.display = 'block';
    list.innerHTML = '';
    for (const seg of segments) {
        const cls = 'flagged-segment ' + (seg.severity === 'critical' ? 'critical' : 'warning');
        const div = el('div', { className: cls });
        div.innerHTML = `
            <div class="flagged-segment__text">"${escapeHtml(seg.text)}"</div>
            <div class="flagged-segment__meta">
                <span>${seg.time_display}</span>
                <span>${seg.source}</span>
                <span>Δ ${seg.delta_log_odds > 0 ? '+' : ''}${seg.delta_log_odds.toFixed(3)}</span>
            </div>
            <div class="flagged-segment__rationale">${escapeHtml(seg.rationale)}</div>
        `;
        list.appendChild(div);
    }
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

// ============================================================================
// UI UPDATE (called every time snapshot arrives)
// ============================================================================

function updateUI(data) {
    if (!data || data.status === 'no_data') return;

    appState.snapshot = data;

    // Status
    const statusVal = $('#statusValue');
    if (statusVal) {
        statusVal.textContent = (data.status || '—').toUpperCase();
        statusVal.className = `top-bar__status-value color-${statusColor(data.status)}`;
    }

    // Progress
    const progVal = $('#progressValue');
    if (progVal) {
        progVal.textContent = `${Math.round((data.progress || 0) * 100)}%`;
    }

    // Alert
    renderAlert(data.alert);

    // Scoreboard
    renderScoreboard(data.participants);

    // Tab-specific rendering
    switch (appState.currentTab) {
        case 'monitor':
            renderBeliefBars(data.participants);
            renderIdentityGauge(data);
            renderAuthenticityGauge(data);
            renderEventFeed(data.evidence);
            break;
        case 'flags':
            renderFlags(data.flags);
            break;
        case 'evidence':
            renderEvidenceRoom(data);
            break;
        case 'timeline':
            renderTimeline(data);
            renderTranscript(data.transcript, data.flagged_segments || []);
            break;
        case 'candidate':
            renderCandidateInfo(data);
            break;
    }

    // Update correction select & candidate select
    updateParticipantSelects(data.participants);
}

// ============================================================================
// RENDER: SCOREBOARD
// ============================================================================

function renderScoreboard(participants) {
    const container = $('#scoreboard');
    if (!container || !participants) return;

    container.innerHTML = '';
    for (const p of participants) {
        const cls = ['sb-card'];
        if (p.is_candidate) cls.push('candidate');
        if (p.is_speaker) cls.push('speaking');
        if (p.flag_count > 0) cls.push('flagged');

        const pct = (p.identity_probability * 100).toFixed(0);
        const color = probColor(p.identity_probability);

        let metaText = '';
        if (p.is_candidate) metaText = '🎯 CANDIDATE';
        else if (p.is_speaker) metaText = '🎤 speaking';
        else if (p.flag_count > 0) metaText = `⚠ ${p.flag_count} flag(s)`;
        else metaText = `A: ${(p.authenticity_probability * 100).toFixed(0)}%`;

        const card = el('div', { className: cls.join(' ') }, [
            el('div', { className: 'sb-card__name', textContent: p.name }),
            el('div', { className: 'sb-card__bar' }, [
                el('div', { className: 'sb-card__bar-fill', style: `width:${pct}%;background:${color};` }),
            ]),
            el('div', { className: 'sb-card__prob', textContent: `${(p.identity_probability * 100).toFixed(1)}%`, style: `color:${color};` }),
            el('div', { className: 'sb-card__meta', textContent: metaText }),
        ]);
        container.appendChild(card);
    }
}

// ============================================================================
// RENDER: BELIEF BARS
// ============================================================================

function renderBeliefBars(participants) {
    const container = $('#beliefBars');
    if (!container || !participants) return;

    container.innerHTML = '';
    for (const p of participants) {
        const pct = (p.identity_probability * 100).toFixed(1);
        const color = p.is_candidate ? '#66bb6a' : '#5c6bc0';

        const bar = el('div', { className: 'belief-bar' }, [
            el('div', { className: 'belief-bar__name', textContent: p.name }),
            el('div', { className: 'belief-bar__track' }, [
                el('div', { className: 'belief-bar__fill', style: `width:${pct}%;background:${color};` }, [
                    el('span', { className: 'belief-bar__pct', textContent: `${pct}%` }),
                ]),
            ]),
        ]);
        container.appendChild(bar);
    }
}

// ============================================================================
// RENDER: GAUGES (SVG arc)
// ============================================================================

function renderGauge(container, value, color, label) {
    if (!container) return;

    const pct = Math.min(100, Math.max(0, value));
    const angle = (pct / 100) * 180;
    const rad = (a) => (a * Math.PI) / 180;

    // Arc from 180° to 0° (left to right semi-circle)
    const r = 70;
    const cx = 90, cy = 90;
    const startAngle = 180;
    const endAngle = 180 - angle;

    const x1 = cx + r * Math.cos(rad(startAngle));
    const y1 = cy - r * Math.sin(rad(startAngle));
    const x2 = cx + r * Math.cos(rad(endAngle));
    const y2 = cy - r * Math.sin(rad(endAngle));

    const largeArc = angle > 180 ? 1 : 0;

    // Background arc (full semi-circle)
    const bgX1 = cx + r * Math.cos(rad(180));
    const bgY1 = cy - r * Math.sin(rad(180));
    const bgX2 = cx + r * Math.cos(rad(0));
    const bgY2 = cy - r * Math.sin(rad(0));

    container.innerHTML = `
        <svg class="gauge-svg" viewBox="0 0 180 110">
            <path d="M ${bgX1} ${bgY1} A ${r} ${r} 0 1 1 ${bgX2} ${bgY2}"
                  fill="none" stroke="#2a2f4f" stroke-width="12" stroke-linecap="round"/>
            ${pct > 0 ? `<path d="M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}"
                  fill="none" stroke="${color}" stroke-width="12" stroke-linecap="round"
                  style="transition: all 0.6s ease;"/>` : ''}
            <text x="${cx}" y="${cy - 10}" text-anchor="middle" fill="${color}"
                  font-family="JetBrains Mono, monospace" font-size="24" font-weight="700">
                ${pct.toFixed(0)}%
            </text>
        </svg>
        <div class="gauge-label">${label}</div>
    `;
}

function renderIdentityGauge(data) {
    const container = $('#identityGauge');
    if (!container) return;

    if (data.status === 'ambiguous') {
        container.innerHTML = `
            <div style="text-align:center;padding:1rem;background:rgba(255,213,79,0.08);
                        border-radius:10px;border:2px solid var(--accent-yellow);">
                <div style="font-size:1.3rem;font-weight:700;color:var(--accent-yellow);">⚠ AMBIGUOUS</div>
                <div class="text-sm color-muted mt-sm">Gap: ${(data.ambiguity_gap * 100).toFixed(1)}%<br>Cannot reliably identify candidate</div>
            </div>`;
        return;
    }

    const topP = data.participants?.find(p => p.is_candidate);
    const prob = data.top_candidate_probability || 0;
    renderGauge(container, prob * 100, '#66bb6a', topP ? `Top: ${topP.name}` : 'No candidate');
}

function renderAuthenticityGauge(data) {
    const container = $('#authenticityGauge');
    if (!container) return;

    const speaker = data.participants?.find(p => p.is_speaker);
    if (speaker) {
        const auth = speaker.authenticity_probability;
        let label = 'No concerns';
        if (auth < 0.3) label = '⚠ Low authenticity';
        else if (auth < 0.6) label = 'Moderate concerns';
        renderGauge(container, auth * 100, '#ffa726', label);
    } else {
        container.innerHTML = '<div class="empty-state"><div class="empty-state__icon">🎤</div>No active speaker</div>';
    }
}

// ============================================================================
// RENDER: EVENT FEED
// ============================================================================

function renderEventFeed(evidence) {
    const container = $('#eventFeed');
    if (!container) return;

    if (!evidence || evidence.length === 0) {
        container.innerHTML = '<div class="empty-state"><div class="empty-state__icon">📡</div>No events yet</div>';
        return;
    }

    container.innerHTML = '';
    // Show newest first
    const items = [...evidence].reverse().slice(0, 30);
    for (const ep of items) {
        const deltaCls = ep.delta_log_odds > 0 ? 'pos' : 'neg';
        const catColor = ep.axis === 'authenticity' ? 'color-orange' : 'color-blue';

        let flagIcon = '';
        if (ep.severity === 'critical') flagIcon = '🚨 ';
        else if (ep.severity === 'warning') flagIcon = '⚠️ ';
        else if (ep.severity === 'info') flagIcon = 'ℹ️ ';

        const row = el('div', { className: 'event-row' }, [
            el('span', { className: 'event-row__ts', textContent: ep.time_display }),
            el('span', { className: `event-row__cat ${catColor}`, textContent: `${flagIcon}${ep.category}` }),
            el('span', { className: 'event-row__name', textContent: ep.target_name?.slice(0, 14) || '' }),
            el('span', { className: `event-row__delta ${deltaCls}`, textContent: `${ep.delta_log_odds > 0 ? '+' : ''}${ep.delta_log_odds.toFixed(3)}` }),
            el('span', { className: 'event-row__rationale', textContent: ep.rationale?.slice(0, 80) || '' }),
        ]);
        container.appendChild(row);
    }
}

// ============================================================================
// RENDER: FLAGS
// ============================================================================

function renderFlags(flags) {
    const container = $('#flagsList');
    if (!container) return;

    if (!flags || flags.length === 0) {
        container.innerHTML = '<div class="empty-state" style="padding:1.5rem;"><div class="empty-state__icon">✅</div>No active flags. All signals nominal.</div>';
        return;
    }

    container.innerHTML = '';
    const items = [...flags].reverse().slice(0, 30);
    for (const ep of items) {
        let cls = 'evidence-item';
        let sevLabel = '';
        if (ep.severity === 'critical') { cls += ' critical'; sevLabel = '🚨 CRITICAL'; }
        else if (ep.severity === 'warning') { cls += ' warning'; sevLabel = '⚠️ WARNING'; }
        else { cls += ' info'; sevLabel = 'ℹ️ INFO'; }

        const sevColor = ep.severity === 'critical' ? 'color-red' : 'color-yellow';

        const item = el('div', { className: cls });
        item.innerHTML = `
            <div class="evidence-item__header">
                <span class="evidence-item__source ${sevColor}">${sevLabel} ${ep.source}</span>
                <span class="evidence-item__target">→ ${ep.target_name}</span>
            </div>
            <div class="evidence-item__meta">Δ log-odds: ${ep.delta_log_odds > 0 ? '+' : ''}${ep.delta_log_odds.toFixed(3)} | Confidence: ${ep.confidence.toFixed(2)}</div>
            <div class="evidence-item__rationale">${ep.rationale}</div>
            <div class="text-xs color-muted mt-sm">${ep.time_display}${ep.recommendation ? ` | Action: ${ep.recommendation}` : ''}</div>
        `;
        container.appendChild(item);
    }
}

// ============================================================================
// RENDER: EVIDENCE ROOM
// ============================================================================

function renderEvidenceRoom(data) {
    if (!data || !data.evidence) return;

    // Update participant filter options
    const partFilter = $('#evParticipantFilter');
    if (partFilter && data.participants) {
        const currentVal = partFilter.value;
        const opts = '<option value="">All Participants</option>' +
            data.participants.map(p => `<option value="${p.name}">${p.name}</option>`).join('');
        if (partFilter.innerHTML !== opts) {
            partFilter.innerHTML = opts;
            partFilter.value = currentVal;
        }
    }

    // Apply filters
    const catFilter = $('#evCatFilter')?.value || '';
    const pidFilter = partFilter?.value || '';
    const searchText = ($('#evSearchInput')?.value || '').toLowerCase();

    let filtered = data.evidence;
    if (catFilter) filtered = filtered.filter(e => e.category === catFilter);
    if (pidFilter) filtered = filtered.filter(e => e.target_name === pidFilter);
    if (searchText) filtered = filtered.filter(e => e.rationale.toLowerCase().includes(searchText));

    // Count
    const countEl = $('#evidenceCount');
    if (countEl) countEl.textContent = `Showing ${filtered.length} of ${data.evidence.length} evidence items`;

    // Render
    const container = $('#evidenceList');
    if (!container) return;

    container.innerHTML = '';
    const items = [...filtered].reverse().slice(0, 40);
    for (const ep of items) {
        let cls = 'evidence-item';
        if (ep.severity === 'critical') cls += ' critical';
        else if (ep.severity === 'warning') cls += ' warning';
        else if (ep.axis === 'identity') cls += ' identity';
        else cls += ' authenticity';

        const item = el('div', { className: cls });
        item.innerHTML = `
            <div class="evidence-item__header">
                <span class="evidence-item__source fw-700">${ep.source}</span>
                <span class="evidence-item__target">→ ${ep.target_name}</span>
            </div>
            <div class="evidence-item__meta">Δ log-odds: ${ep.delta_log_odds > 0 ? '+' : ''}${ep.delta_log_odds.toFixed(3)} | Confidence: ${ep.confidence.toFixed(2)}</div>
            <div class="evidence-item__rationale">${ep.rationale}</div>
            <div class="text-xs color-muted mt-sm">${ep.time_display} | ${ep.category}${ep.severity !== 'none' ? ` | Severity: ${ep.severity.toUpperCase()}` : ''}</div>
        `;
        container.appendChild(item);
    }
}

// ============================================================================
// RENDER: TIMELINE & TRANSCRIPT
// ============================================================================

function renderTimeline(data) {
    const scrubber = $('#timelineScrubber');
    const currentLabel = $('#timelineCurrent');
    const totalLabel = $('#timelineTotal');
    if (!scrubber || !data) return;

    const total = data.total_duration || 1;
    const current = data.elapsed_seconds || 0;

    scrubber.max = total;
    scrubber.value = current;
    if (currentLabel) currentLabel.textContent = formatSeconds(current);
    if (totalLabel) totalLabel.textContent = formatSeconds(total);
}

function renderTranscript(transcript, flaggedSegments = []) {
    const container = $('#transcriptContainer');
    if (!container) return;

    if (!transcript || transcript.length === 0) {
        container.innerHTML = '<div class="empty-state"><div class="empty-state__icon">💬</div>No transcript yet</div>';
        return;
    }

    container.innerHTML = '';
    for (const seg of transcript) {
        const isFlagged = flaggedSegments.some(fs =>
            fs.text && seg.text && (fs.text.includes(seg.text) || seg.text.includes(fs.text))
        );
        const cls = 'transcript-segment' +
            (seg.is_question ? ' question' : '') +
            (isFlagged ? ' flagged' : '');
        const flagIcon = isFlagged ? '🚩 ' : '';
        const div = el('div', { className: cls });
        div.innerHTML = `
            <span class="transcript-segment__speaker">${flagIcon}${seg.name}</span>: ${seg.text}
            <div class="transcript-segment__time">${seg.time_display}</div>
        `;
        container.appendChild(div);
    }
    // Auto-scroll to bottom
    container.scrollTop = container.scrollHeight;
}

// ============================================================================
// RENDER: CANDIDATE INTELLIGENCE
// ============================================================================

function renderCandidateInfo(data) {
    if (!data || !data.participants) return;

    const select = $('#candidateSelect');
    if (!select) return;

    // Update select options
    const currentVal = select.value;
    const opts = data.participants.map(p => `<option value="${p.id}">${p.name}</option>`).join('');
    if (select.innerHTML !== opts) {
        select.innerHTML = opts;
        select.value = currentVal || (data.participants[0]?.id || '');
    }

    const pid = select.value;
    const participant = data.participants.find(p => p.id === pid);
    if (!participant) return;

    // Metrics
    const metricsContainer = $('#candidateMetrics');
    if (metricsContainer) {
        metricsContainer.innerHTML = '';
        const metrics = [
            { label: 'Identity', value: `${(participant.identity_probability * 100).toFixed(1)}%`, color: probColor(participant.identity_probability) },
            { label: 'Authenticity', value: `${(participant.authenticity_probability * 100).toFixed(1)}%`, color: '#ffa726' },
            { label: 'Evidence', value: participant.evidence_count, color: 'var(--text-primary)' },
            { label: 'Log-Odds', value: `${participant.identity_log_odds >= 0 ? '+' : ''}${participant.identity_log_odds.toFixed(2)}`, color: 'var(--accent-blue-l)' },
        ];
        for (const m of metrics) {
            metricsContainer.appendChild(el('div', { className: 'metric-card' }, [
                el('div', { className: 'metric-card__label', textContent: m.label }),
                el('div', { className: 'metric-card__value', textContent: m.value, style: `color:${m.color};` }),
            ]));
        }
    }

    // Profile
    const profileCard = $('#candidateProfile');
    const profileFields = $('#profileFields');
    if (profileCard && profileFields) {
        profileCard.style.display = 'block';
        profileFields.innerHTML = '';
        const fields = [];
        if (participant.email) fields.push({ label: 'Email', value: participant.email });
        if (participant.join_time) fields.push({ label: 'Joined', value: participant.join_time });
        fields.push({ label: 'Webcam', value: participant.webcam_on ? 'On' : 'Off' });
        fields.push({ label: 'Screen Share', value: participant.is_screen_sharing ? 'Yes' : 'No' });
        if (participant.device_name) fields.push({ label: 'Device', value: participant.device_name });

        for (const f of fields) {
            profileFields.appendChild(el('div', { className: 'profile-field' }, [
                el('div', { className: 'profile-field__label', textContent: f.label }),
                el('div', { className: 'profile-field__value', textContent: f.value }),
            ]));
        }
    }

    // Evidence for this participant
    const evContainer = $('#candidateEvidence');
    if (evContainer && data.evidence) {
        const pEvidence = data.evidence.filter(e => e.target_id === pid);
        evContainer.innerHTML = '';
        if (pEvidence.length === 0) {
            evContainer.innerHTML = '<div class="empty-state text-sm">No evidence for this participant.</div>';
        } else {
            const items = [...pEvidence].reverse().slice(0, 15);
            for (const ep of items) {
                const cls = 'evidence-item' + (ep.axis === 'identity' ? ' identity' : ' authenticity');
                const item = el('div', { className: cls });
                item.innerHTML = `
                    <div class="evidence-item__header"><span class="evidence-item__source fw-700">${ep.source}</span></div>
                    <div class="evidence-item__meta">Δ: ${ep.delta_log_odds > 0 ? '+' : ''}${ep.delta_log_odds.toFixed(3)} | C: ${ep.confidence.toFixed(2)}</div>
                    <div class="evidence-item__rationale">${ep.rationale}</div>
                    <div class="text-xs color-muted">${ep.time_display}</div>
                `;
                evContainer.appendChild(item);
            }
        }
    }

    // Timeline chart
    const chartCard = $('#candidateChartCard');
    const canvas = $('#candidateChart');
    if (chartCard && canvas && data.timelines && data.timelines[pid]) {
        const timeline = data.timelines[pid];
        if (timeline.length > 1) {
            chartCard.style.display = 'block';
            drawLineChart(canvas, timeline);
        } else {
            chartCard.style.display = 'none';
        }
    }
}

// ============================================================================
// SIMPLE CANVAS LINE CHART
// ============================================================================

function drawLineChart(canvas, dataPoints) {
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;

    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.scale(dpr, dpr);

    const pad = { top: 20, right: 15, bottom: 25, left: 45 };
    const chartW = w - pad.left - pad.right;
    const chartH = h - pad.top - pad.bottom;

    // Clear
    ctx.clearRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = '#2a2f4f';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = pad.top + (chartH / 4) * i;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(pad.left + chartW, y);
        ctx.stroke();
    }

    // Y-axis labels
    ctx.fillStyle = '#9fa8da';
    ctx.font = '10px JetBrains Mono';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
        const y = pad.top + (chartH / 4) * i;
        const val = 100 - (i * 25);
        ctx.fillText(`${val}%`, pad.left - 6, y + 3);
    }

    // Data
    const values = dataPoints.map(d => d.value * 100);
    const n = values.length;
    if (n < 2) return;

    const xStep = chartW / (n - 1);

    // Line
    ctx.beginPath();
    ctx.strokeStyle = '#5c6bc0';
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    for (let i = 0; i < n; i++) {
        const x = pad.left + i * xStep;
        const y = pad.top + chartH - (values[i] / 100) * chartH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Points
    ctx.fillStyle = '#5c6bc0';
    for (let i = 0; i < n; i++) {
        const x = pad.left + i * xStep;
        const y = pad.top + chartH - (values[i] / 100) * chartH;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
    }
}

// ============================================================================
// RENDER: ALERT BANNER
// ============================================================================

function renderAlert(alert) {
    const banner = $('#alertBanner');
    if (!banner) return;

    if (alert) {
        banner.classList.remove('alert-banner--hidden');
        $('#alertText').textContent = `CRITICAL: ${alert.message} — ${alert.participant}`;
    } else {
        banner.classList.add('alert-banner--hidden');
    }
}

// ============================================================================
// RENDER: SCENARIO SELECT & GRID
// ============================================================================

function renderScenarioSelect() {
    const select = $('#scenarioSelect');
    if (!select) return;

    select.innerHTML = '<option value="">— Select Scenario —</option>' +
        appState.scenarios.map(s => `<option value="${s.id}">${s.name}</option>`).join('');
}

function renderScenarioGrid() {
    const grid = $('#scenarioGrid');
    if (!grid) return;

    grid.innerHTML = '';
    appState.scenarios.forEach((s, i) => {
        const card = el('div', { className: 'scenario-card' }, [
            el('span', { className: 'scenario-card__num', textContent: i + 1 }),
            el('h4', { textContent: s.name }),
            el('p', { textContent: s.description }),
            el('button', {
                className: 'btn btn-primary',
                textContent: '▶  Open',
                style: 'width:100%;margin-top:0.5rem;',
                onClick: () => loadScenario(s.id),
            }),
        ]);
        grid.appendChild(card);
    });
}

// ============================================================================
// PARTICIPANT SELECTS
// ============================================================================

function updateParticipantSelects(participants) {
    if (!participants) return;

    // Correction select
    const corrSel = $('#correctionSelect');
    if (corrSel) {
        const currentVal = corrSel.value;
        corrSel.innerHTML = participants.map(p => `<option value="${p.id}">${p.name}</option>`).join('');
        if (currentVal) corrSel.value = currentVal;
    }
}

// ============================================================================
// SHOW/HIDE PAGES
// ============================================================================

function showDashboard() {
    $('#landingPage').classList.add('hidden');
    $('#dashboardPage').classList.remove('hidden');
    $('#statusDisplay').style.display = '';
    $('#playbackControls').style.display = appState.liveMode ? 'none' : '';
    $('#progressDisplay').style.display = '';
    $('#loadBtn').style.display = 'none';
    $('#newAnalysisBtn').style.display = 'inline-flex';

    if (appState.liveMode) {
        $('#liveVideoPanel').style.display = 'block';
    } else {
        $('#liveVideoPanel').style.display = 'none';
    }
}

function showLanding() {
    $('#landingPage').classList.remove('hidden');
    $('#dashboardPage').classList.add('hidden');
    $('#statusDisplay').style.display = 'none';
    $('#playbackControls').style.display = 'none';
    $('#progressDisplay').style.display = 'none';
    $('#liveVideoPanel').style.display = 'none';
    $('#newAnalysisBtn').style.display = 'none';
}

function updatePlayButton() {
    const btn = $('#playBtn');
    if (btn) {
        btn.textContent = appState.isPlaying ? '⏸' : '▶';
        btn.title = appState.isPlaying ? 'Pause' : 'Play';
    }
}

// ============================================================================
// EVENT HANDLERS
// ============================================================================

function setupEventHandlers() {
    // Scenario select
    const scenarioSelect = $('#scenarioSelect');
    scenarioSelect?.addEventListener('change', (e) => {
        const id = e.target.value;
        if (id) {
            loadScenario(id);
        }
    });

    // Live analysis start/stop
    $('#liveStartBtn')?.addEventListener('click', startLive);
    $('#liveStopBtn')?.addEventListener('click', stopLive);
    $('#newAnalysisBtn')?.addEventListener('click', async () => {
        await stopLive();
        showLanding();
        // Scroll the upload panel into view.
        const panel = $('.live-panel');
        if (panel) panel.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });

    // Sync live video current time to elapsed display
    const liveVideo = $('#liveVideo');
    liveVideo?.addEventListener('timeupdate', () => {
        const elapsedEl = $('#lvElapsed');
        if (elapsedEl && liveVideo.duration) {
            elapsedEl.textContent = formatSeconds(liveVideo.currentTime);
        }
    });

    // Play/Pause
    $('#playBtn')?.addEventListener('click', () => {
        appState.isPlaying = !appState.isPlaying;
        updatePlayButton();
        wsSend({ action: appState.isPlaying ? 'play' : 'pause' });
    });

    // Step
    $('#stepBtn')?.addEventListener('click', () => {
        appState.isPlaying = false;
        updatePlayButton();
        wsSend({ action: 'step' });
    });

    // Reset
    $('#resetBtn')?.addEventListener('click', () => {
        appState.isPlaying = false;
        updatePlayButton();
        wsSend({ action: 'reset' });
    });

    // Speed slider
    const speedSlider = $('#speedSlider');
    speedSlider?.addEventListener('input', (e) => {
        const speed = parseFloat(e.target.value);
        appState.playbackSpeed = speed;
        $('#speedLabel').textContent = `${speed}×`;
        wsSend({ action: 'speed', value: speed });
    });

    // Tabs
    $$('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const tabName = tab.dataset.tab;
            appState.currentTab = tabName;

            // Active tab styling
            $$('.tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            // Show/hide panels
            $$('.tab-panel').forEach(p => p.classList.remove('active'));
            $(`#panel-${tabName}`)?.classList.add('active');

            // Re-render current tab with latest data
            if (appState.snapshot) updateUI(appState.snapshot);
        });
    });

    // Timeline scrubber
    const scrubber = $('#timelineScrubber');
    scrubber?.addEventListener('change', (e) => {
        const seconds = parseFloat(e.target.value);
        wsSend({ action: 'seek', seconds });
    });

    // Evidence filters
    $('#evCatFilter')?.addEventListener('change', () => { if (appState.snapshot) renderEvidenceRoom(appState.snapshot); });
    $('#evParticipantFilter')?.addEventListener('change', () => { if (appState.snapshot) renderEvidenceRoom(appState.snapshot); });
    $('#evSearchInput')?.addEventListener('input', () => { if (appState.snapshot) renderEvidenceRoom(appState.snapshot); });

    // Candidate select
    $('#candidateSelect')?.addEventListener('change', () => { if (appState.snapshot) renderCandidateInfo(appState.snapshot); });

    // Feedback: Confirm
    $('#confirmBtn')?.addEventListener('click', async () => {
        if (!appState.snapshot?.top_candidate_id) return;
        try {
            await api('/api/feedback/confirm', {
                method: 'POST',
                body: JSON.stringify({ participant_id: appState.snapshot.top_candidate_id }),
            });
            showToast('Candidate confirmed ✅');
        } catch (e) {
            showToast('Error: ' + e.message, true);
        }
    });

    // Feedback: Correct
    $('#correctBtn')?.addEventListener('click', async () => {
        const pid = $('#correctionSelect')?.value;
        if (!pid) return;
        try {
            await api('/api/feedback/correct', {
                method: 'POST',
                body: JSON.stringify({ participant_id: pid }),
            });
            showToast('Correction applied 🔧');
        } catch (e) {
            showToast('Error: ' + e.message, true);
        }
    });

    // Feedback: Save note
    $('#saveNoteBtn')?.addEventListener('click', async () => {
        const input = $('#noteInput');
        const text = input?.value?.trim();
        if (!text) return;
        try {
            const result = await api('/api/feedback/note', {
                method: 'POST',
                body: JSON.stringify({
                    text,
                    candidate_id: appState.snapshot?.top_candidate_id || null,
                }),
            });
            input.value = '';
            appState.notes.push(result.note);
            renderNotes();
            showToast('Note saved 📝');
        } catch (e) {
            showToast('Error: ' + e.message, true);
        }
    });

    // Export report
    $('#exportReportBtn')?.addEventListener('click', () => {
        window.open('/api/report', '_blank');
    });

    // Export evidence JSON
    $('#exportEvidenceBtn')?.addEventListener('click', async () => {
        try {
            const data = await api('/api/report/json');
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `evidence_${new Date().toISOString().slice(0, 16).replace(/:/g, '')}.json`;
            a.click();
            URL.revokeObjectURL(url);
        } catch (e) {
            showToast('Error exporting: ' + e.message, true);
        }
    });
}

// ============================================================================
// OPERATOR NOTES
// ============================================================================

function renderNotes() {
    const container = $('#notesContainer');
    const wrapper = $('#operatorNotes');
    if (!container || !wrapper) return;

    if (appState.notes.length === 0) {
        wrapper.style.display = 'none';
        return;
    }

    wrapper.style.display = 'block';
    container.innerHTML = '';
    for (const note of appState.notes.slice(-5)) {
        container.appendChild(el('div', { className: 'operator-note' }, [
            el('span', { className: 'operator-note__time', textContent: `[${(note.timestamp || '').slice(0, 16)}] ` }),
            document.createTextNode(note.text),
        ]));
    }
}

// ============================================================================
// TOAST NOTIFICATIONS
// ============================================================================

function showToast(message, isError = false) {
    const toast = el('div', {
        className: 'alert-banner',
        style: `position:fixed;bottom:1.5rem;right:1.5rem;z-index:9999;max-width:350px;
                animation:fadeIn 0.3s ease;${isError ? 'border-color:var(--accent-red);color:var(--accent-red);' : 'border-color:var(--accent-green);color:var(--accent-green);background:rgba(102,187,106,0.1);'}`,
        textContent: message,
    });
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 2500);
}

// ============================================================================
// INIT
// ============================================================================

async function checkExistingLiveSession() {
    try {
        const info = await api('/api/live/info');
        if (info.running) {
            appState.liveMode = true;
            appState.liveRunning = true;
            appState.liveVideoUrl = info.video_url;
            showDashboard();
            setupLiveVideo(info.video_url, info.video_title);
            updateLiveStatusRow(true, info.video_title);
            connectLiveWS();
        }
    } catch (e) {
        console.log('No existing live session');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    setupEventHandlers();
    loadScenarios();
    checkExistingLiveSession();
});
