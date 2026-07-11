"""
Sherlock Candidate Identification Engine - Streamlit Demo

Operations console with live scoreboard, flags, evidence room, and candidate intelligence.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import plotly.graph_objects as go
import streamlit as st

from sherlock.feedback import FeedbackLoop
from sherlock.models import FlagSeverity, SignalAxis, SignalSource
from sherlock.session_replay import SessionReplay, list_available_fixtures, load_fixture

# Optional live A/V analysis (experimental). Guarded so a missing dependency
# never breaks the fixture demo.
try:
    from sherlock.live import LiveSession
    _LIVE_AVAILABLE = True
except Exception:  # pragma: no cover
    _LIVE_AVAILABLE = False


# ============================================================================
# CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="Sherlock - Candidate Identification Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DARK_THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Sora:wght@600;700;800&display=swap');

    :root {
        --bg-primary: #0a0e27;
        --bg-secondary: #151932;
        --bg-tertiary: #1e2340;
        --text-primary: #e8eaf6;
        --text-secondary: #9fa8da;
        --accent-blue: #5c6bc0;
        --accent-green: #66bb6a;
        --accent-red: #ef5350;
        --accent-yellow: #ffd54f;
        --accent-orange: #ffa726;
        --border-color: #2a2f4f;
        --font-body: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        --font-head: 'Sora', 'Inter', sans-serif;
    }

    html, body, [class*="st-"], .stApp {
        font-family: var(--font-body) !important;
    }

    .stApp {
        background: linear-gradient(160deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        color: var(--text-primary);
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-head) !important;
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    .main .block-container {
        max-width: 1320px;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        padding-top: 1rem;
    }

    .stMetric {
        background: var(--bg-tertiary);
        padding: 0.85rem 1.25rem;
        border-radius: 10px;
        border: 1px solid var(--border-color);
    }

    .stMetric label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }

    .stMetric [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
    }

    .stButton > button {
        background: var(--accent-blue);
        color: white;
        border: none;
        padding: 0.45rem 1.4rem;
        border-radius: 8px;
        font-weight: 600;
        font-family: var(--font-body) !important;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: #7986cb;
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(92, 107, 192, 0.35);
    }

    .stSelectbox > div > div {
        background: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px;
    }

    .stSlider > div > div > div { background: var(--accent-blue) !important; }

    .stCheckbox > label > div { color: var(--text-primary) !important; }

    /* ---- Custom components ---- */
    .top-bar {
        background: var(--bg-secondary);
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
    }

    .bottom-bar {
        background: var(--bg-secondary);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        margin-top: 1.5rem;
    }

    .panel-card {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem 1.25rem 1.15rem;
        margin-bottom: 0.75rem;
    }

    .panel-card h3 {
        margin-top: 0 !important;
        font-size: 1rem !important;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 0.7rem !important;
    }

    .dash-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--accent-blue), transparent);
        border: none;
        margin: 1.25rem 0;
        opacity: 0.5;
    }

    /* --- Scoreboard cards --- */
    .sb-card {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
        text-align: center;
        transition: all 0.2s;
    }
    .sb-card.candidate { border-color: var(--accent-green); box-shadow: 0 0 12px rgba(102,187,106,0.15); }
    .sb-card.speaking  { animation: sb-pulse 1.5s infinite; }
    .sb-card.flagged   { border-color: var(--accent-red); }
    .sb-card .name     { font-weight: 600; font-size: 0.85rem; margin-bottom: 0.3rem; }
    .sb-card .prob-bar { height: 5px; border-radius: 3px; margin: 0.25rem 0; background: var(--border-color); overflow: hidden; }
    .sb-card .prob-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }
    .sb-card .meta     { font-size: 0.7rem; color: var(--text-secondary); }

    @keyframes sb-pulse {
        0%, 100% { box-shadow: 0 0 10px rgba(102,187,106,0.1); }
        50%      { box-shadow: 0 0 20px rgba(102,187,106,0.25); }
    }

    /* --- Event feed --- */
    .event-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0.5rem;
        border-bottom: 1px solid rgba(42,47,79,0.5);
        font-size: 0.82rem;
        line-height: 1.3;
    }
    .event-row .ts    { color: var(--text-secondary); min-width: 5rem; }
    .event-row .cat   { font-weight: 600; min-width: 6.5rem; }
    .event-row .delta { font-family: 'Inter', monospace; min-width: 4rem; text-align: right; }
    .event-row .delta.pos { color: var(--accent-green); }
    .event-row .delta.neg { color: var(--accent-red); }

    .flag-critical { border-left: 3px solid var(--accent-red); background: rgba(239,83,80,0.08); }
    .flag-warning  { border-left: 3px solid var(--accent-yellow); background: rgba(255,213,79,0.06); }
    .flag-info     { border-left: 3px solid var(--accent-blue); }

    /* --- Landing --- */
    .hero { text-align: center; padding: 2.5rem 1rem 1.5rem; border-radius: 16px; background: linear-gradient(135deg, rgba(92,107,192,0.12) 0%, rgba(30,35,64,0.3) 100%); border: 1px solid var(--border-color); margin-bottom: 2rem; }
    .hero h1 { font-size: 2.8rem; margin-bottom: 0.4rem; background: linear-gradient(90deg, #e8eaf6, #9fa8da); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero p  { font-size: 1.05rem; color: var(--text-secondary); max-width: 600px; margin: 0 auto; }
    .section-title { font-size: 1.4rem; margin: 2rem 0 0.8rem; padding-bottom: 0.4rem; border-bottom: 2px solid var(--border-color); }
    .feature-card { background: var(--bg-tertiary); border: 1px solid var(--border-color); border-radius: 12px; padding: 1.25rem; height: 100%; }
    .feature-card .icon { font-size: 1.5rem; margin-bottom: 0.3rem; }
    .feature-card h4   { font-size: 1rem; margin-bottom: 0.3rem; }
    .feature-card p    { font-size: 0.85rem; color: var(--text-secondary); margin: 0; }
    .scenario-card { background: var(--bg-tertiary); border: 1px solid var(--border-color); border-radius: 12px; padding: 1rem; height: 100%; }
    .scenario-card .num { display: inline-block; width: 1.8rem; height: 1.8rem; line-height: 1.8rem; text-align: center; border-radius: 50%; background: var(--accent-blue); color: white; font-weight: 700; font-family: var(--font-head); margin-bottom: 0.5rem; }

    /* --- Participant tile (Transcript & Candidate Info) --- */
    .participant-tile {
        background: var(--bg-tertiary);
        border: 2px solid var(--border-color);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin: 0.3rem 0;
    }
    .participant-tile.candidate { border-color: var(--accent-green); box-shadow: 0 0 15px rgba(102,187,106,0.15); }

    .status-badge { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 10px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }
    .status-badge.identified { background: rgba(102,187,106,0.2); color: var(--accent-green); }
    .status-badge.ambiguous  { background: rgba(255,213,79,0.2); color: var(--accent-yellow); }
    .status-badge.flagged    { background: rgba(239,83,80,0.2); color: var(--accent-red); }

    .transcript-segment { background: var(--bg-tertiary); padding: 0.55rem 0.75rem; margin: 0.35rem 0; border-radius: 6px; border-left: 3px solid var(--accent-blue); font-size: 0.88rem; }
    .transcript-segment.question { border-left-color: var(--accent-orange); }

    .evidence-item { background: var(--bg-tertiary); border-left: 3px solid var(--accent-blue); padding: 0.55rem 0.75rem; margin: 0.35rem 0; border-radius: 4px; font-size: 0.82rem; }
    .evidence-item.identity    { border-left-color: var(--accent-blue); }
    .evidence-item.authenticity { border-left-color: var(--accent-orange); }
    .evidence-item.critical    { border-left-color: var(--accent-red); }
    .evidence-item.warning     { border-left-color: var(--accent-yellow); }
</style>
"""

st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


# ============================================================================
# SIGNAL CATEGORIES
# ============================================================================

SIGNAL_CATEGORIES = {
    "Identity Priors": [
        SignalSource.CALENDAR_MATCH, SignalSource.INTERVIEWER_NEGATIVE,
        SignalSource.EMAIL_DOMAIN, SignalSource.JOIN_TIMING,
        SignalSource.DISPLAY_NAME_CHANGE,
    ],
    "Behavioral": [
        SignalSource.TURN_TAKING, SignalSource.SPEAKING_RATIO,
        SignalSource.SCREEN_SHARE, SignalSource.WEBCAM_STATE,
    ],
    "Semantic LLM": [SignalSource.LLM_ROLE_CLASSIFIER],
    "Authenticity": [
        SignalSource.DISFLUENCY_ANOMALY, SignalSource.PAUSE_FLUENCY_PATTERN,
        SignalSource.CODING_TELEMETRY, SignalSource.GAZE_DETECTION,
    ],
}


def get_signal_category(source: SignalSource) -> str:
    for category, sources in SIGNAL_CATEGORIES.items():
        if source in sources:
            return category
    return "Other"


# ============================================================================
# SESSION STATE
# ============================================================================

def init_session_state():
    if "replay" not in st.session_state:
        st.session_state.replay = None
    if "current_snapshot" not in st.session_state:
        st.session_state.current_snapshot = None
    if "is_playing" not in st.session_state:
        st.session_state.is_playing = False
    if "playback_speed" not in st.session_state:
        st.session_state.playback_speed = 1.0
    if "active_categories" not in st.session_state:
        st.session_state.active_categories = list(SIGNAL_CATEGORIES.keys())
    if "feedback_loop" not in st.session_state:
        st.session_state.feedback_loop = None
    if "feedback_applied" not in st.session_state:
        st.session_state.feedback_applied = False
    if "operator_notes" not in st.session_state:
        st.session_state.operator_notes = []
    if "selected_candidate" not in st.session_state:
        st.session_state.selected_candidate = None
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = 0
    if "live_session" not in st.session_state:
        st.session_state.live_session = None


init_session_state()


def load_scenario(fixture_path: Path) -> None:
    fixture = load_fixture(fixture_path)
    st.session_state.replay = SessionReplay(fixture)
    st.session_state.feedback_loop = FeedbackLoop(st.session_state.replay.engine)
    st.session_state.current_snapshot = None
    st.session_state.is_playing = False
    st.session_state.feedback_applied = False
    st.session_state.selected_candidate = None
    st.rerun()


# ============================================================================
# TOP BAR
# ============================================================================

def render_top_bar():
    """Render the top command bar with global status."""
    st.markdown('<div class="top-bar">', unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6 = st.columns([2.5, 1.2, 1, 1, 1, 1.5])

    with col1:
        fixtures_dir = Path(__file__).parent / "sherlock" / "fixtures"
        available_fixtures = list_available_fixtures(fixtures_dir)
        fixture_options = {f["name"]: f for f in available_fixtures}
        selected_name = st.selectbox("Scenario", options=list(fixture_options.keys()), index=0)
        if selected_name:
            if st.session_state.replay is None or st.button("Load Scenario", use_container_width=True):
                load_scenario(fixture_options[selected_name]["path"])

    with col2:
        if st.session_state.replay:
            snap = st.session_state.current_snapshot
            if snap:
                status = snap.status
                color = {"identified": "green", "ambiguous": "yellow", "no_participants": "gray"}.get(status, "gray")
                st.markdown(f"""
                    <div style="text-align:center;padding:0.3rem 0;">
                        <div style="font-size:0.7rem;color:var(--text-secondary);text-transform:uppercase;">Status</div>
                        <div style="font-weight:700;color:var(--accent-{color});">{status.upper()}</div>
                    </div>
                """, unsafe_allow_html=True)

    with col3:
        if st.session_state.replay:
            if st.button("▶ Run" if not st.session_state.is_playing else "⏸ Pause", use_container_width=True):
                st.session_state.is_playing = not st.session_state.is_playing

    with col4:
        if st.session_state.replay:
            if st.button("↺ Reset", use_container_width=True):
                st.session_state.replay.reset()
                st.session_state.current_snapshot = None
                st.session_state.is_playing = False

    with col5:
        speed = st.slider("Speed", 0.5, 4.0, st.session_state.playback_speed, 0.5, label_visibility="collapsed")
        if speed != st.session_state.playback_speed:
            st.session_state.playback_speed = speed
            if st.session_state.replay:
                st.session_state.replay.set_playback_speed(speed)

    with col6:
        if st.session_state.replay and st.session_state.current_snapshot:
            progress = st.session_state.replay.get_progress()
            st.metric("Progress", f"{progress:.0%}")

    # Alert banner for critical flags
    if st.session_state.replay:
        flags = st.session_state.replay.get_active_flags()
        critical = [f for f in flags if f.severity == FlagSeverity.CRITICAL]
        if critical:
            f = critical[0]
            pname = st.session_state.replay.fixture.meeting_context.participants.get(f.target_participant_id)
            name = pname.display_name if pname else f.target_participant_id
            st.markdown(f"""
                <div style="padding:0.4rem 0.8rem;margin-top:0.5rem;border-radius:8px;
                            background:rgba(239,83,80,0.12);border:1px solid var(--accent-red);
                            font-size:0.85rem;color:var(--accent-red);">
                    🚨 CRITICAL: {f.rationale[:120]} — <strong>{name}</strong>
                </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# SCOREBOARD
# ============================================================================

def prob_color(p: float) -> str:
    if p >= 0.7:
        return "#66bb6a"
    if p >= 0.4:
        return "#ffd54f"
    return "#ef5350"


def render_scoreboard():
    """Render a horizontal scoreboard of all participants."""
    snap = st.session_state.current_snapshot
    if not snap or not st.session_state.replay:
        return

    st.markdown('<div style="margin-bottom:0.5rem;">', unsafe_allow_html=True)
    st.markdown("### 📊 Live Scoreboard")

    participants = []
    for pid in snap.beliefs.keys():
        belief = snap.beliefs[pid]
        participant = st.session_state.replay.fixture.meeting_context.participants.get(pid)
        name = participant.display_name if participant else pid
        is_candidate = (pid == snap.top_candidate_id and snap.status == "identified")
        is_speaker = (pid == snap.current_speaker_id)

        flags = [f for f in snap.evidence_ledger if f.target_participant_id == pid
                 and f.severity not in (FlagSeverity.NONE,)]
        is_flagged = len(flags) > 0

        prob = belief.identity_probability
        participants.append((pid, name, prob, belief.authenticity_probability, is_candidate, is_speaker, is_flagged))

    cols = st.columns(len(participants))
    for col, (pid, name, prob, auth_prob, is_candidate, is_speaker, is_flagged) in zip(cols, participants):
        cls = "sb-card"
        if is_candidate: cls += " candidate"
        if is_speaker:   cls += " speaking"
        if is_flagged:   cls += " flagged"
        pct = prob * 100
        color = prob_color(prob)

        with col:
            st.markdown(f"""
                <div class="{cls}">
                    <div class="name">{name}</div>
                    <div class="prob-bar"><div class="prob-fill" style="width:{pct:.0f}%;background:{color};"></div></div>
                    <div style="font-size:0.95rem;font-weight:700;">{prob:.1%}</div>
                    <div class="meta">
                        {is_candidate and "🎯 CANDIDATE" or ""}
                        {is_speaker and "🎤 speaking" or ""}
                        {is_flagged and f"⚠ {len(flags)} flag(s)" or ""}
                        {not is_candidate and not is_speaker and not is_flagged and f"A:{auth_prob:.0%}" or ""}
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# EVENT FEED
# ============================================================================

def render_event_feed(max_items: int = 20):
    """Render a console-style chronological event feed."""
    snap = st.session_state.current_snapshot
    if not snap:
        return

    st.markdown('<div style="max-height:320px;overflow-y:auto;">', unsafe_allow_html=True)
    for ep in reversed(snap.evidence_ledger[-max_items:]):
        delta_cls = "pos" if ep.delta_log_odds > 0 else "neg"
        cat = get_signal_category(ep.source)
        participant = st.session_state.replay.fixture.meeting_context.participants.get(ep.target_participant_id)
        name = participant.display_name[:14] if participant else ep.target_participant_id[:14]
        flag_icon = ""
        if ep.severity == FlagSeverity.CRITICAL:
            flag_icon = "🚨"
        elif ep.severity == FlagSeverity.WARNING:
            flag_icon = "⚠️"
        elif ep.severity == FlagSeverity.INFO:
            flag_icon = "ℹ️"

        st.markdown(f"""
            <div class="event-row">
                <span class="ts">{ep.timestamp.strftime('%H:%M:%S')}</span>
                <span class="cat" style="color:var(--accent-{'orange' if ep.axis == SignalAxis.AUTHENTICITY else 'blue'});">
                    {flag_icon} {cat}
                </span>
                <span style="min-width:7rem;">{name}</span>
                <span class="delta {delta_cls}">{ep.delta_log_odds:+.3f}</span>
                <span style="color:var(--text-secondary);font-size:0.78rem;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
                    {ep.rationale[:80]}
                </span>
            </div>
        """, unsafe_allow_html=True)

    if not snap.evidence_ledger:
        st.caption("No events yet.")
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# BELIEF VISUALIZATION (existing)
# ============================================================================

def render_belief_bars():
    """Render animated belief distribution bars."""
    snap = st.session_state.current_snapshot
    if not snap:
        return

    participants, probabilities, colors = [], [], []
    for pid, belief in snap.beliefs.items():
        participant = st.session_state.replay.fixture.meeting_context.participants.get(pid)
        name = participant.display_name if participant else pid
        participants.append(name)
        probabilities.append(belief.identity_probability)
        colors.append("#66bb6a" if (pid == snap.top_candidate_id and snap.status == "identified") else "#5c6bc0")

    fig = go.Figure(data=[go.Bar(x=probabilities, y=participants, orientation='h',
                                  marker_color=colors, text=[f"{p:.1%}" for p in probabilities],
                                  textposition='auto')])
    fig.update_layout(xaxis_title="Probability", xaxis_range=[0, 1], height=280,
                      margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaf6"),
                      xaxis=dict(gridcolor="#2a2f4f"), yaxis=dict(gridcolor="#2a2f4f"))
    st.plotly_chart(fig, use_container_width=True)


def render_confidence_gauge():
    """Render confidence gauge or ambiguous state."""
    snap = st.session_state.current_snapshot
    if not snap:
        return

    if snap.status == "ambiguous":
        st.markdown(f"""
            <div style="text-align:center;padding:1.5rem;background:rgba(255,213,79,0.08);
                        border-radius:12px;border:2px solid var(--accent-yellow);margin-bottom:0.5rem;">
                <h3 style="color:var(--accent-yellow);margin:0;">⚠ AMBIGUOUS</h3>
                <p style="color:var(--text-secondary);margin-top:0.5rem;font-size:0.9rem;">
                    Gap: {snap.ambiguity_gap:.1%}<br>Cannot reliably identify candidate</p>
            </div>
        """, unsafe_allow_html=True)
        return

    fig = go.Figure(go.Indicator(mode="gauge+number", value=snap.top_candidate_probability * 100,
                                  number={'suffix': "%", 'font': {'size': 36, 'color': "#66bb6a"}},
                                  gauge={'axis': {'range': [0, 100], 'tickcolor': "#e8eaf6"},
                                         'bar': {'color': "#66bb6a"}, 'bgcolor': "#1e2340",
                                         'bordercolor': "#2a2f4f"}))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    if snap.top_candidate_id:
        p = st.session_state.replay.fixture.meeting_context.participants.get(snap.top_candidate_id)
        name = p.display_name if p else snap.top_candidate_id
        st.success(f"Top candidate: **{name}**")


def render_authenticity_gauge():
    """Render authenticity gauge for current speaker."""
    snap = st.session_state.current_snapshot
    if not snap:
        return

    if snap.current_speaker_id and snap.current_speaker_id in snap.beliefs:
        auth_prob = snap.beliefs[snap.current_speaker_id].authenticity_probability
        fig = go.Figure(go.Indicator(mode="gauge+number", value=auth_prob * 100,
                                      number={'suffix': "%", 'font': {'size': 28, 'color': "#ffa726"}},
                                      gauge={'axis': {'range': [0, 100], 'tickcolor': "#e8eaf6"},
                                             'bar': {'color': "#ffa726"}, 'bgcolor': "#1e2340",
                                             'bordercolor': "#2a2f4f"}))
        fig.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        auth_pct = auth_prob * 100
        if auth_pct < 30:
            st.error("⚠ Low authenticity — candidate may be assisted")
        elif auth_pct < 60:
            st.warning("Moderate authenticity concerns")
        else:
            st.info("No authenticity concerns")
    else:
        st.info("No active speaker")


# ============================================================================
# FLAGS & ALERTS
# ============================================================================

def render_flags():
    """Render active flags panel with severity."""
    snap = st.session_state.current_snapshot
    if not snap:
        st.info("No scenario loaded.")
        return

    flags = [ep for ep in snap.evidence_ledger if ep.severity not in (FlagSeverity.NONE,)]
    if not flags:
        st.success("No active flags. All signals nominal.")
        return

    for ep in reversed(flags[-30:]):
        cls = "evidence-item"
        sev = ""
        if ep.severity == FlagSeverity.CRITICAL:
            cls += " critical"
            sev = "🚨 CRITICAL"
        elif ep.severity == FlagSeverity.WARNING:
            cls += " warning"
            sev = "⚠️ WARNING"
        else:
            cls += " info"
            sev = "ℹ️ INFO"

        p = st.session_state.replay.fixture.meeting_context.participants.get(ep.target_participant_id)
        name = p.display_name if p else ep.target_participant_id

        # correlation note
        note = ""
        if ep.axis == SignalAxis.AUTHENTICITY and snap.top_candidate_id != ep.target_participant_id:
            top_p = st.session_state.replay.fixture.meeting_context.participants.get(snap.top_candidate_id)
            top_name = top_p.display_name if top_p else snap.top_candidate_id or "unknown"
            note = f"<br><em style='color:var(--accent-yellow);font-size:0.8rem;'>ⓘ Flag raised on <strong>{name}</strong> while top candidate is <strong>{top_name}</strong> — verify identity before acting on the flag.</em>"

        st.markdown(f"""
            <div class="{cls}">
                <strong style="color:var(--accent-{'red' if ep.severity == FlagSeverity.CRITICAL else 'yellow'});">
                    {sev} {ep.source.value}</strong> → {name}<br>
                Δ log-odds: {ep.delta_log_odds:+.3f} | Confidence: {ep.confidence:.2f}<br>
                <em>{ep.rationale}</em>
                {note}
                <div style="font-size:0.75rem;color:var(--text-secondary);margin-top:0.15rem;">
                    {ep.timestamp.strftime('%H:%M:%S')}{ep.recommendation and f' | Action: {ep.recommendation}' or ''}
                </div>
            </div>
        """, unsafe_allow_html=True)


# ============================================================================
# EVIDENCE ROOM
# ============================================================================

def render_evidence_room():
    """Render filterable evidence ledger with search and export."""
    snap = st.session_state.current_snapshot
    if not snap:
        return

    st.subheader("📜 Evidence Room")

    # Filters: category + participant
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        cat_filter = st.multiselect("Category", list(SIGNAL_CATEGORIES.keys()),
                                    default=list(SIGNAL_CATEGORIES.keys()), key="ev_cat_filter")
    with filter_col2:
        all_pids = list(snap.beliefs.keys())
        pid_labels = []
        for pid in all_pids:
            p = st.session_state.replay.fixture.meeting_context.participants.get(pid)
            pid_labels.append(p.display_name if p else pid)
        pid_filter = st.multiselect("Participant", pid_labels, key="ev_pid_filter")

    with filter_col3:
        search_text = st.text_input("🔍 Search rationale", key="ev_search")

    # Apply filters
    filtered = []
    for ep in snap.evidence_ledger:
        cat = get_signal_category(ep.source)
        if cat not in cat_filter:
            continue
        p = st.session_state.replay.fixture.meeting_context.participants.get(ep.target_participant_id)
        pname = p.display_name if p else ep.target_participant_id
        if pid_filter and pname not in pid_filter:
            continue
        if search_text and search_text.lower() not in ep.rationale.lower():
            continue
        filtered.append(ep)

    if not filtered:
        st.caption("No evidence matches filters.")
        return

    st.caption(f"Showing {len(filtered)} of {len(snap.evidence_ledger)} evidence items")
    container = st.container(height=400)
    with container:
        for ep in reversed(filtered[-40:]):
            cls = "evidence-item"
            if ep.severity == FlagSeverity.CRITICAL: cls += " critical"
            elif ep.severity == FlagSeverity.WARNING: cls += " warning"
            elif ep.axis == SignalAxis.IDENTITY: cls += " identity"
            else: cls += " authenticity"

            p = st.session_state.replay.fixture.meeting_context.participants.get(ep.target_participant_id)
            name = p.display_name if p else ep.target_participant_id
            delta_sign = "+" if ep.delta_log_odds > 0 else ""

            st.markdown(f"""
                <div class="{cls}">
                    <strong>{ep.source.value}</strong> → {name}<br>
                    Δ log-odds: {delta_sign}{ep.delta_log_odds:.3f} | Confidence: {ep.confidence:.2f}<br>
                    <em>{ep.rationale}</em>
                    <div style="font-size:0.75rem;color:var(--text-secondary);margin-top:0.15rem;">
                        {ep.timestamp.strftime('%H:%M:%S')} | {get_signal_category(ep.source)}
                        {ep.severity != FlagSeverity.NONE and f' | Severity: {ep.severity.value.upper()}' or ''}
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("📥 Export Evidence as JSON", use_container_width=True):
        ledger_data = [{
            "source": ep.source.value, "axis": ep.axis.value,
            "target_participant_id": ep.target_participant_id,
            "delta_log_odds": ep.delta_log_odds, "confidence": ep.confidence,
            "rationale": ep.rationale, "timestamp": ep.timestamp.isoformat(),
            "severity": ep.severity.value, "flag_type": ep.flag_type,
        } for ep in snap.evidence_ledger]
        st.download_button("💾 Download JSON", data=json.dumps(ledger_data, indent=2),
                           file_name=f"evidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                           mime="application/json")


# ============================================================================
# TIMELINE & TRANSCRIPT
# ============================================================================

def render_timeline():
    """Render scrubber with synchronized transcript."""
    snap = st.session_state.current_snapshot
    if not st.session_state.replay:
        return

    total_duration = st.session_state.replay.get_total_duration()
    current_time = st.session_state.replay.current_time
    start_time = st.session_state.replay.start_time

    if current_time and start_time:
        current_seconds = (current_time - start_time).total_seconds()
        scrub = st.slider("⏱️ Timeline", 0.0, total_duration, current_seconds, 1.0, format="%f s", key="timeline_scrub")
        if abs(scrub - current_seconds) > 1.0:
            target_time = start_time + timedelta(seconds=scrub)
            st.session_state.current_snapshot = st.session_state.replay.advance_to_time(target_time)
            st.rerun()

    if not snap:
        return

    st.markdown("---")
    st.subheader("💬 Transcript")

    transcript_container = st.container(height=350)
    with transcript_container:
        for segment in snap.transcript_segments[-15:]:
            p = st.session_state.replay.fixture.meeting_context.participants.get(segment.participant_id)
            name = p.display_name if p else segment.participant_id
            cls = "transcript-segment"
            if segment.is_question:
                cls += " question"
            st.markdown(f"""
                <div class="{cls}">
                    <strong>{name}</strong>: {segment.text}
                    <div style="font-size:0.75rem;color:var(--text-secondary);">{segment.start_time.strftime('%H:%M:%S')}</div>
                </div>
            """, unsafe_allow_html=True)


# ============================================================================
# CANDIDATE INTELLIGENCE
# ============================================================================

def render_candidate_intelligence():
    """Render per-participant deep-dive panel."""
    snap = st.session_state.current_snapshot
    if not snap:
        return

    st.subheader("🎯 Candidate Intelligence")

    pids = list(snap.beliefs.keys())
    pid_labels = []
    for pid in pids:
        p = st.session_state.replay.fixture.meeting_context.participants.get(pid)
        pid_labels.append(p.display_name if p else pid)

    selected = st.selectbox("Select participant", pid_labels, key="ci_select")
    if not selected:
        return

    pid = pids[pid_labels.index(selected)]
    participant = st.session_state.replay.fixture.meeting_context.participants.get(pid)
    belief = snap.beliefs[pid]

    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    with stats_col1:
        st.metric("Identity", f"{belief.identity_probability:.1%}")
    with stats_col2:
        st.metric("Authenticity", f"{belief.authenticity_probability:.1%}")
    with stats_col3:
        st.metric("Evidence", f"{len(belief.identity_evidence) + len(belief.authenticity_evidence)}")
    with stats_col4:
        log_odds = belief.identity_log_odds
        st.metric("Log-Odds", f"{log_odds:+.2f}")

    # Profile card
    if participant:
        st.markdown("""
            <div class="panel-card" style="margin-top:0.5rem;">
                <h3>Profile</h3>
        """, unsafe_allow_html=True)
        info = []
        if participant.email:
            info.append(f"**Email:** `{participant.email}`")
        if participant.join_time:
            info.append(f"**Joined:** {participant.join_time.strftime('%H:%M:%S')}")
        info.append(f"**Webcam:** {'On' if participant.webcam_on else 'Off'}")
        info.append(f"**Screen share:** {'Yes' if participant.is_screen_sharing else 'No'}")
        if participant.device_name:
            info.append(f"**Device:** {participant.device_name}")
        if participant.display_name_history:
            last_change = participant.display_name_history[-1]
            info.append(f"**Name history:** {last_change.get('old_name', '?')} → {last_change.get('new_name', '?')}")
        for line in info:
            st.markdown(f"<div style='font-size:0.88rem;margin:0.15rem 0;'>{line}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Evidence for this participant
    st.markdown('<div class="panel-card"><h3>All Evidence</h3>', unsafe_allow_html=True)
    all_ev = belief.identity_evidence + belief.authenticity_evidence
    all_ev.sort(key=lambda e: e.timestamp, reverse=True)
    for ep in all_ev[:15]:
        cls = "evidence-item"
        if ep.axis == SignalAxis.IDENTITY:
            cls += " identity"
        else:
            cls += " authenticity"
        delta_sign = "+" if ep.delta_log_odds > 0 else ""
        st.markdown(f"""
            <div class="{cls}">
                <strong>{ep.source.value}</strong><br>
                Δ: {delta_sign}{ep.delta_log_odds:.3f} | C: {ep.confidence:.2f}<br>
                <em>{ep.rationale}</em>
                <div style="font-size:0.75rem;color:var(--text-secondary);">{ep.timestamp.strftime('%H:%M:%S')}</div>
            </div>
        """, unsafe_allow_html=True)
    if not all_ev:
        st.caption("No evidence for this participant.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Confidence timeline sparkline
    hist = st.session_state.replay.get_participant_timeline(pid, "identity_probability")
    if len(hist) > 1:
        st.markdown('<div class="panel-card"><h3>Confidence Over Time</h3>', unsafe_allow_html=True)
        times = [t for t, _ in hist]
        vals = [v * 100 for _, v in hist]
        fig = go.Figure(data=go.Scatter(x=times, y=vals, mode="lines+markers",
                                         line=dict(color="#5c6bc0", width=2),
                                         marker=dict(size=5, color="#5c6bc0")))
        fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#e8eaf6"), xaxis=dict(gridcolor="#2a2f4f", showticklabels=False),
                          yaxis=dict(gridcolor="#2a2f4f", title="Probability %"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# LANDING PAGE
# ============================================================================

def render_landing_page():
    """Render structured landing page when no scenario loaded."""
    st.markdown("""
        <div class="hero">
            <h1>🔍 Sherlock</h1>
            <p>Real-time Bayesian candidate identification &amp; authenticity tracking for live
            interviews. Fuses many weak signals and explains every decision.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<h2 class="section-title">What Sherlock answers</h2>', unsafe_allow_html=True)
    cols = st.columns(2)
    with cols[0]:
        st.markdown('<div class="feature-card"><div class="icon">🎯</div><h4>WHO is the candidate?</h4><p>Live belief distribution over every participant — survives nicknames, device names, and mid-call name changes.</p></div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown('<div class="feature-card"><div class="icon">🛡️</div><h4>HOW authentic?</h4><p>Separate authenticity stream flags reading, paste bursts, and assisted-work signatures without contaminating identity.</p></div>', unsafe_allow_html=True)

    st.markdown('<h2 class="section-title">How it works</h2>', unsafe_allow_html=True)
    steps = [
        ("📡", "Ingestion", "Per-participant audio, calendar events, and transcripts."),
        ("🧩", "Signal Extractors", "14 independent weak signals emit evidence packets."),
        ("📊", "Fusion Engine", "Bayesian log-odds tracker updates beliefs per participant."),
        ("🧾", "Explanation", "Every number traces back to an ordered evidence ledger."),
        ("👤", "Feedback", "Interviewer confirm/correct recalibrates signal weights."),
    ]
    sc = st.columns(len(steps))
    for col, (icon, title, desc) in zip(sc, steps):
        with col:
            st.markdown(f'<div class="feature-card"><div class="icon">{icon}</div><h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

    st.markdown('<h2 class="section-title">Try a scenario</h2>', unsafe_allow_html=True)
    st.caption("Pick an edge-case scenario to watch Sherlock identify the candidate in real time.")
    fixtures_dir = Path(__file__).parent / "sherlock" / "fixtures"
    available_fixtures = list_available_fixtures(fixtures_dir)

    for i in range(0, len(available_fixtures), 3):
        row = available_fixtures[i:i + 3]
        cols = st.columns(3)
        for j, fx in enumerate(row):
            with cols[j]:
                global_idx = i + j + 1
                st.markdown(f'<div class="scenario-card"><span class="num">{global_idx}</span><h4>{fx["name"]}</h4><p style="color:var(--text-secondary);font-size:0.82rem;">{fx["description"]}</p></div>', unsafe_allow_html=True)
                if st.button(f"▶  Open", key=f"landing_{fx['id']}", use_container_width=True):
                    load_scenario(fx["path"])


# ============================================================================
# BOTTOM BAR
# ============================================================================

def generate_full_report():
    """Generate an HTML report string with full analysis for export."""
    snap = st.session_state.current_snapshot
    replay = st.session_state.replay
    if not snap or not replay:
        return "<p>No data to report.</p>"

    ctx = replay.fixture.meeting_context

    lines = [f"""<html><head><meta charset="utf-8"><title>Sherlock Report</title>
<style>body{{font-family:Inter,sans-serif;background:#0a0e27;color:#e8eaf6;padding:2rem;max-width:900px;margin:0 auto;}}
h1{{font-family:Sora,sans-serif;font-size:2rem;border-bottom:2px solid #2a2f4f;padding-bottom:0.5rem;}}
h2{{font-size:1.3rem;margin-top:2rem;color:#9fa8da;}}
table{{width:100%;border-collapse:collapse;margin:1rem 0;}}
th,td{{text-align:left;padding:0.5rem;border-bottom:1px solid #2a2f4f;}}
th{{color:#9fa8da;font-weight:600;}}
.flag-c{{color:#ef5350;font-weight:700;}}
.flag-w{{color:#ffd54f;font-weight:700;}}
.sig-pos{{color:#66bb6a;}}
.sig-neg{{color:#ef5350;}}
</style></head><body>"""]

    lines.append(f"<h1>🔍 Sherlock — Candidate Identification Report</h1>")
    lines.append(f"<p>Meeting: <strong>{ctx.meeting_id}</strong> | "
                 f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>")

    # Status
    status_color = {"identified": "green", "ambiguous": "yellow", "no_participants": "gray"}.get(snap.status, "gray")
    lines.append(f"<h2 style='color:{status_color};'>{snap.status.upper()}</h2>")
    if snap.top_candidate_id:
        p = ctx.participants.get(snap.top_candidate_id)
        tn = p.display_name if p else snap.top_candidate_id
        lines.append(f"<p>Top candidate: <strong>{tn}</strong> — {snap.top_candidate_probability:.1%} confidence "
                     f"(gap: {snap.ambiguity_gap:.1%})</p>")

    # Participants table
    lines.append("<h2>Participants</h2>")
    lines.append("<table><tr><th>Name</th><th>Identity</th><th>Authenticity</th><th>Evidence</th><th>Status</th></tr>")
    for pid, belief in sorted(snap.beliefs.items(), key=lambda x: x[1].identity_probability, reverse=True):
        p = ctx.participants.get(pid)
        name = p.display_name if p else pid
        is_top = pid == snap.top_candidate_id and snap.status == "identified"
        status = "🎯 CANDIDATE" if is_top else ""
        lines.append(f"<tr><td>{name}</td><td>{belief.identity_probability:.1%}</td>"
                     f"<td>{belief.authenticity_probability:.1%}</td>"
                     f"<td>{len(belief.identity_evidence) + len(belief.authenticity_evidence)}</td>"
                     f"<td>{status}</td></tr>")
    lines.append("</table>")

    # Flags
    flags = [ep for ep in snap.evidence_ledger if ep.severity not in (FlagSeverity.NONE,)]
    if flags:
        lines.append("<h2>🚨 Flags & Alerts</h2>")
        lines.append("<table><tr><th>Severity</th><th>Source</th><th>Participant</th><th>Delta</th><th>Rationale</th></tr>")
        for ep in reversed(flags[-30:]):
            sev_cls = "flag-c" if ep.severity == FlagSeverity.CRITICAL else "flag-w"
            p = ctx.participants.get(ep.target_participant_id)
            name = p.display_name if p else ep.target_participant_id
            delta_cls = "sig-pos" if ep.delta_log_odds > 0 else "sig-neg"
            lines.append(f"<tr class='{sev_cls}'><td>{ep.severity.value.upper()}</td><td>{ep.source.value}</td>"
                         f"<td>{name}</td><td class='{delta_cls}'>{ep.delta_log_odds:+.3f}</td>"
                         f"<td>{ep.rationale[:100]}</td></tr>")
        lines.append("</table>")

    # Evidence summary
    lines.append("<h2>Evidence Summary</h2>")
    lines.append(f"<p>Total evidence packets: {len(snap.evidence_ledger)}</p>")
    by_source = {}
    for ep in snap.evidence_ledger:
        key = f"{ep.source.value} ({ep.axis.value})"
        by_source.setdefault(key, 0)
        by_source[key] += 1
    lines.append("<table><tr><th>Signal Source</th><th>Count</th></tr>")
    for src, count in sorted(by_source.items(), key=lambda x: -x[1]):
        lines.append(f"<tr><td>{src}</td><td>{count}</td></tr>")
    lines.append("</table>")

    lines.append("</body></html>")
    return "\n".join(lines)


def render_bottom_bar():
    """Render the bottom feedback bar with operator actions."""
    snap = st.session_state.current_snapshot
    if not snap:
        return

    st.markdown('<div class="bottom-bar">', unsafe_allow_html=True)
    st.subheader("👤 Interviewer Feedback & Actions")
    st.markdown('<hr class="dash-divider">', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([1.5, 1.5, 1.5, 1])

    with col1:
        if st.button("✅ Confirm Candidate", use_container_width=True):
            if snap.top_candidate_id and st.session_state.feedback_loop:
                st.session_state.feedback_loop.record_confirmation(
                    snap.top_candidate_id, notes="Interviewer confirmed via UI")
                st.session_state.feedback_applied = True

    with col2:
        participant_options = list(snap.beliefs.keys())
        pnames = []
        for pid in participant_options:
            p = st.session_state.replay.fixture.meeting_context.participants.get(pid)
            pnames.append(p.display_name if p else pid)
        selected_pid = st.selectbox("Correct to", options=participant_options, format_func=lambda x: pnames[participant_options.index(x)], key="correction_select")
        if st.button("🔧 Apply Correction", use_container_width=True):
            if selected_pid and st.session_state.feedback_loop:
                st.session_state.feedback_loop.record_correction(selected_pid, notes="Interviewer corrected via UI")
                st.session_state.feedback_applied = True

    with col3:
        note_text = st.text_input("✏ Operator note", key="op_note_input", placeholder="Why this correction?")
        if st.button("📝 Save Note", use_container_width=True):
            if note_text:
                st.session_state.operator_notes.append({
                    "text": note_text, "timestamp": datetime.utcnow().isoformat(),
                    "candidate_id": snap.top_candidate_id,
                })
                st.success("Note saved")

    with col4:
        if st.session_state.feedback_applied:
            summary = st.session_state.feedback_loop.get_feedback_summary()
            with st.popover("⚙️ Weights"):
                st.text(summary)

    # Export report row
    st.markdown("---")
    exp_col1, exp_col2 = st.columns([1, 4])
    with exp_col1:
        if st.button("📄 Export Full Report", use_container_width=True):
            report_html = generate_full_report()
            st.download_button("💾 Download HTML Report", data=report_html,
                               file_name=f"sherlock_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                               mime="text/html")

    # Show operator notes
    if st.session_state.operator_notes:
        st.markdown("---")
        st.caption("Operator Notes")
        for note in st.session_state.operator_notes[-5:]:
            st.markdown(f"""
                <div style="font-size:0.82rem;padding:0.3rem 0.5rem;border-left:2px solid var(--accent-blue);
                            margin:0.2rem 0;color:var(--text-secondary);">
                    <strong>[{note.get('timestamp','?')[:16]}]</strong> {note['text']}
                </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================

def render_live_panel():
    """Experimental live A/V analysis (Prompt 10.3.7).

    Runs independently of the fixture replay: it launches the real-time
    orchestrator on a media file (or a synthetic frame stream) and displays the
    live candidate identification plus any authenticity flags. Only the
    identified candidate's frames are analyzed.
    """
    if not _LIVE_AVAILABLE:
        return
    with st.expander("🟢 Live A/V Analysis (experimental)", expanded=False):
        st.caption(
            "Runs the real-time orchestrator on a media file (or synthetic frames). "
            "Only the identified candidate's frames are sent to fraud-detection pipelines."
        )
        col1, col2 = st.columns([3, 1])
        with col1:
            file_path = st.text_input("Media file path (blank = synthetic)", key="live_path")
        with col2:
            cand_name = st.text_input("Candidate name", value="Candidate", key="live_cand")
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.button("▶ Start live", key="live_start"):
                sess = LiveSession()
                sess.start(file_path if file_path else None, candidate_name=cand_name)
                st.session_state.live_session = sess
                st.rerun()
        with bcol2:
            if st.button("■ Stop live", key="live_stop"):
                if st.session_state.live_session:
                    st.session_state.live_session.stop()
                st.session_state.live_session = None
                st.rerun()

        sess = st.session_state.live_session
        if sess:
            status = sess.refresh_status()
            st.write(f"**State:** {status['state']}")
            st.write(
                f"**Top candidate:** {status['top_candidate_id']} "
                f"({status['confidence']:.1%} confidence)"
            )
            st.write(
                f"**p95 latency:** {status['p95_latency_ms']:.1f} ms | "
                f"**non-candidate frames dropped:** {status['dropped_non_candidate']}"
            )
            if status["flags"]:
                for f in status["flags"]:
                    st.warning(f"[{f['severity']}] {f['source']}: {f['rationale']}")
            else:
                st.info("No active flags.")


def main():
    """Main application entry point."""
    st.title("🔍 Sherlock — Candidate Identification Engine")
    st.caption("Real-time Bayesian fusion for interview candidate identification | Multi-signal evidence tracking")

    # Top bar (always visible)
    render_top_bar()

    # Experimental live A/V analysis (independent of the fixture replay).
    render_live_panel()

    if st.session_state.replay is None:
        render_landing_page()
        return

    # Auto-advance if playing
    if st.session_state.is_playing:
        snapshot = st.session_state.replay.step_forward()
        if snapshot:
            st.session_state.current_snapshot = snapshot
        else:
            st.session_state.is_playing = False

    # Get current snapshot
    if st.session_state.current_snapshot is None:
        st.session_state.current_snapshot = st.session_state.replay.get_current_snapshot()

    # --- SCOREBOARD ---
    render_scoreboard()

    st.markdown('<hr class="dash-divider">', unsafe_allow_html=True)

    # --- TABS for main content ---
    tab_labels = ["📊 Monitor", "🚨 Flags & Alerts", "📜 Evidence Room", "⏱ Timeline", "🎯 Candidate Info"]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        col_left, col_right = st.columns([1.2, 1])
        with col_left:
            render_belief_bars()
        with col_right:
            st.markdown('<div class="panel-card"><h3>🔐 Identity Confidence</h3>', unsafe_allow_html=True)
            render_confidence_gauge()
            st.markdown('</div>')
            st.markdown('<div class="panel-card"><h3>🔐 Authenticity (Current Speaker)</h3>', unsafe_allow_html=True)
            render_authenticity_gauge()
            st.markdown('</div>')

        st.markdown('<hr class="dash-divider">', unsafe_allow_html=True)
        st.markdown('<div class="dash-divider" style="margin:0.25rem 0;"></div>')

        # Event feed in a panel
        st.markdown('<div class="panel-card"><h3>📡 Live Event Feed</h3>', unsafe_allow_html=True)
        render_event_feed(30)
        st.markdown('</div>')

    with tabs[1]:
        st.markdown('<div class="panel-card"><h3>🚨 Active Flags & Alerts</h3>', unsafe_allow_html=True)
        render_flags()
        st.markdown('</div>')

    with tabs[2]:
        render_evidence_room()

    with tabs[3]:
        render_timeline()

    with tabs[4]:
        render_candidate_intelligence()

    # --- BOTTOM BAR ---
    render_bottom_bar()


if __name__ == "__main__":
    main()
