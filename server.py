"""
Sherlock Candidate Identification Engine — FastAPI Server

Replaces the Streamlit app with a proper REST + WebSocket backend.
Serves the frontend static files and provides all pipeline endpoints.
"""

import asyncio
import json
import logging
import mimetypes
import shutil
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sherlock.feedback import FeedbackLoop
from sherlock.models import FlagSeverity, SignalAxis, SignalSource
from sherlock.session_replay import SessionReplay, list_available_fixtures, load_fixture

# Optional live A/V analysis
try:
    from sherlock.live import LiveSession
    _LIVE_AVAILABLE = True
except Exception:
    _LIVE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# APP SETUP
# ============================================================================

app = FastAPI(title="Sherlock — Candidate Identification Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
FRONTEND_DIR = Path(__file__).parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# ============================================================================
# SERVER STATE (single-user demo)
# ============================================================================

class ServerState:
    def __init__(self):
        self.replay: Optional[SessionReplay] = None
        self.feedback_loop: Optional[FeedbackLoop] = None
        self.operator_notes: List[Dict] = []
        self.live_session: Optional[Any] = None
        self.is_playing: bool = False
        self.playback_speed: float = 1.0
        self.live_video_path: Optional[Path] = None
        self.live_video_title: str = ""
        self.uploaded_files: Dict[str, Path] = {}

state = ServerState()

UPLOAD_DIR = Path(tempfile.gettempdir()) / "sherlock_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

FIXTURES_DIR = Path(__file__).parent / "sherlock" / "fixtures"

# ============================================================================
# SIGNAL CATEGORIES (same as original)
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
        SignalSource.AI_GENERATED_TEXT, SignalSource.AI_GENERATED_SPEECH,
        SignalSource.READING_PATTERN, SignalSource.UNNATURAL_PAUSE,
    ],
}


def get_signal_category(source: SignalSource) -> str:
    for category, sources in SIGNAL_CATEGORIES.items():
        if source in sources:
            return category
    return "Other"


def _build_explainability(engine, top_id: str, participant: Dict, sess=None) -> Dict:
    """Build a rich explainability payload for the current live snapshot."""
    from collections import defaultdict

    all_eps = engine.evidence_ledger

    # ── Evidence breakdown by source ──────────────────────────────────
    by_source = defaultdict(lambda: {"count": 0, "total_delta": 0.0, "confidence_sum": 0.0,
                                      "identity_delta": 0.0, "authenticity_delta": 0.0})
    for ep in all_eps:
        s = ep.source.value
        by_source[s]["count"] += 1
        by_source[s]["total_delta"] += abs(ep.delta_log_odds)
        by_source[s]["confidence_sum"] += ep.confidence
        if ep.axis == SignalAxis.IDENTITY:
            by_source[s]["identity_delta"] += ep.delta_log_odds
        else:
            by_source[s]["authenticity_delta"] += ep.delta_log_odds

    evidence_by_source = {}
    for src, d in sorted(by_source.items(), key=lambda x: x[1]["total_delta"], reverse=True):
        d["avg_confidence"] = round(d["confidence_sum"] / d["count"], 3) if d["count"] else 0
        d["identity_delta"] = round(d["identity_delta"], 3)
        d["authenticity_delta"] = round(d["authenticity_delta"], 3)
        d["total_delta"] = round(d["total_delta"], 3)
        d["category"] = get_signal_category(SignalSource(src))
        evidence_by_source[src] = d

    # ── Evidence breakdown by category ────────────────────────────────
    by_cat = defaultdict(lambda: {"count": 0, "total_identity_delta": 0.0, "total_authenticity_delta": 0.0})
    for ep in all_eps:
        cat = get_signal_category(ep.source)
        by_cat[cat]["count"] += 1
        if ep.axis == SignalAxis.IDENTITY:
            by_cat[cat]["total_identity_delta"] += ep.delta_log_odds
        else:
            by_cat[cat]["total_authenticity_delta"] += ep.delta_log_odds

    evidence_by_category = {}
    for cat, d in by_cat.items():
        evidence_by_category[cat] = {
            "count": d["count"],
            "identity_delta": round(d["total_identity_delta"], 3),
            "authenticity_delta": round(d["total_authenticity_delta"], 3),
        }

    # ── Top contributing evidence ─────────────────────────────────────
    top_contributors = []
    for ep in sorted(all_eps[-30:], key=lambda e: abs(e.delta_log_odds), reverse=True)[:8]:
        top_contributors.append({
            "source": ep.source.value,
            "axis": ep.axis.value,
            "delta_log_odds": round(ep.delta_log_odds, 3),
            "confidence": round(ep.confidence, 3),
            "rationale": ep.rationale[:120],
            "category": get_signal_category(ep.source),
            "flag_type": ep.flag_type,
        })

    # ── Pipeline status ───────────────────────────────────────────────
    orc = sess.orchestrator if sess else None
    if orc:
        pipeline_status = {
            "deepfake_video": {"active": True, "evidence_count": sum(
                1 for ep in engine.evidence_ledger if ep.source == SignalSource.DEEPFAKE_VIDEO)},
            "voice_liveness": {"active": True, "evidence_count": sum(
                1 for ep in engine.evidence_ledger if ep.source == SignalSource.VOICE_LIVENESS)},
            "gaze_detection": {"active": True, "evidence_count": sum(
                1 for ep in engine.evidence_ledger if ep.source == SignalSource.GAZE_DETECTION)},
            "audio_authenticity": {"active": True, "evidence_count": sum(
                1 for ep in engine.evidence_ledger if ep.source == SignalSource.AI_GENERATED_SPEECH)},
            "text_authenticity": {"active": True, "evidence_count": sum(
                1 for ep in engine.evidence_ledger
                if ep.source in (SignalSource.AI_GENERATED_TEXT, SignalSource.HUMAN_SPONTANEOUS_TEXT,
                                 SignalSource.READING_PATTERN, SignalSource.UNNATURAL_PAUSE))},
            "transcription_live": {"active": bool(orc.transcript_segments),
                                   "evidence_count": len(orc.transcript_segments)},
            "behavioral_signals": {"active": True, "evidence_count": sum(
                1 for ep in engine.evidence_ledger
                if ep.source in (SignalSource.TURN_TAKING, SignalSource.SPEAKING_RATIO))},
        }
    else:
        pipeline_status = {}

    # ── Verdict summary ───────────────────────────────────────────────
    belief = engine.beliefs.get(top_id)
    identity_log_odds = belief.identity_log_odds if belief else 0.0
    auth_log_odds = belief.authenticity_log_odds if belief else 0.0
    id_prob = participant.get("identity_probability", 0)
    auth_prob = participant.get("authenticity_probability", 0)

    verdict_summary = {
        "identity_probability": id_prob,
        "authenticity_probability": auth_prob,
        "identity_confidence_level": _confidence_level(id_prob),
        "authenticity_confidence_level": _confidence_level(auth_prob),
        "identity_log_odds": round(identity_log_odds, 3),
        "authenticity_log_odds": round(auth_log_odds, 3),
        "total_evidence_count": len(all_eps),
        "identity_evidence_count": len(belief.identity_evidence) if belief else 0,
        "authenticity_evidence_count": len(belief.authenticity_evidence) if belief else 0,
        "flags_active": participant.get("flag_count", 0),
    }

    return {
        "evidence_by_source": evidence_by_source,
        "evidence_by_category": evidence_by_category,
        "top_contributors": top_contributors,
        "pipeline_status": pipeline_status,
        "verdict_summary": verdict_summary,
    }


def _confidence_level(prob: float) -> str:
    if prob >= 0.9:
        return "high"
    if prob >= 0.7:
        return "moderate"
    if prob >= 0.4:
        return "low"
    return "very_low"


def serialize_live_snapshot(sess) -> Dict:
    """Convert a LiveSession into the same snapshot shape as replay."""
    if not sess or not sess.engine:
        return {"status": "no_data", "participants": [], "state": "idle"}

    engine = sess.engine
    result = engine.get_result()
    status = sess.status

    elapsed = status.get("elapsed_seconds", 0.0)
    total = status.get("total_duration", 0.0)
    progress = min(1.0, elapsed / total) if total > 0 else 0.0

    top_id = result.top_candidate_id or "candidate"
    top_prob = result.top_candidate_probability
    status_str = result.status

    # Single candidate participant.
    belief = engine.beliefs.get(top_id)
    identity_prob = belief.identity_probability if belief else top_prob
    auth_prob = belief.authenticity_probability if belief else 0.5

    # LiveSession stores context on the engine, not directly on itself.
    ctx = getattr(sess.engine, "context", None)
    candidate_name = (
        status.get("video_title")
        or (ctx.candidate_name if ctx else None)
        or "Candidate"
    )

    participant = {
        "id": top_id,
        "name": candidate_name,
        "identity_probability": round(identity_prob, 4),
        "authenticity_probability": round(auth_prob, 4),
        "identity_log_odds": round(belief.identity_log_odds, 3) if belief else 0.0,
        "evidence_count": len(engine.evidence_ledger),
        "is_candidate": True,
        "is_speaker": True,
        "flag_count": len([ep for ep in engine.evidence_ledger
                           if ep.target_participant_id == top_id
                           and ep.severity != FlagSeverity.NONE]),
        "email": None,
        "join_time": None,
        "webcam_on": True,
        "is_screen_sharing": False,
        "device_name": "Live Video",
        "display_name_history": [],
    }

    evidence = []
    for ep in engine.evidence_ledger[-50:]:
        evidence.append({
            "source": ep.source.value,
            "axis": ep.axis.value,
            "target_id": ep.target_participant_id,
            "target_name": participant["name"],
            "delta_log_odds": round(ep.delta_log_odds, 4),
            "confidence": round(ep.confidence, 3),
            "rationale": ep.rationale,
            "timestamp": ep.timestamp.isoformat(),
            "time_display": ep.timestamp.strftime('%H:%M:%S'),
            "severity": ep.severity.value,
            "flag_type": ep.flag_type,
            "recommendation": ep.recommendation,
            "category": get_signal_category(ep.source),
        })

    flags = [e for e in evidence if e["severity"] != "none"]

    # Transcript from orchestrator if available.
    transcript = []
    orchestrator = getattr(sess, "orchestrator", None)
    if orchestrator:
        for seg in getattr(orchestrator, "transcript_segments", [])[-30:]:
            transcript.append({
                "participant_id": seg.participant_id,
                "name": participant["name"],
                "text": seg.text,
                "start_time": seg.start_time.isoformat(),
                "time_display": seg.start_time.strftime('%H:%M:%S'),
                "is_question": seg.is_question,
            })

    # Flagged transcript segments (text/audio authenticity signals).
    flagged_segments = []
    text_flags = [ep for ep in engine.evidence_ledger
                  if ep.source in (
                      SignalSource.AI_GENERATED_TEXT,
                      SignalSource.READING_PATTERN,
                      SignalSource.UNNATURAL_PAUSE,
                      SignalSource.AI_GENERATED_SPEECH,
                  ) and ep.severity != FlagSeverity.NONE]
    for ep in text_flags[-10:]:
        seg_text = ep.metadata.get("segment_text", "") if ep.metadata else ""
        flagged_segments.append({
            "text": seg_text or ep.rationale,
            "time_display": ep.timestamp.strftime('%H:%M:%S'),
            "source": ep.source.value,
            "severity": ep.severity.value,
            "rationale": ep.rationale,
            "delta_log_odds": round(ep.delta_log_odds, 3),
        })

    # Verdict reasons from recent authenticity flags.
    verdict_reasons = [
        ep.rationale for ep in engine.evidence_ledger[-20:]
        if ep.axis == SignalAxis.AUTHENTICITY and ep.severity != FlagSeverity.NONE
    ][-5:]

    # Timeline for charts.
    timelines = {
        top_id: [
            {"time": datetime.utcnow().isoformat(), "value": round(top_prob, 4)}
        ] + [{"time": datetime.utcnow().isoformat(), "value": round(t.get("confidence", 0), 4)}
             for t in status.get("timeline", [])[-30:]]
    }

    alert = None
    critical = [ep for ep in engine.evidence_ledger if ep.severity == FlagSeverity.CRITICAL]
    if critical:
        f = critical[-1]
        alert = {
            "message": f.rationale[:150],
            "participant": participant["name"],
        }

    return {
        "status": status_str,
        "top_candidate_id": top_id,
        "top_candidate_probability": round(top_prob, 4),
        "ambiguity_gap": round(result.ambiguity_gap, 4),
        "current_speaker_id": top_id,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.utcnow().isoformat(),
        "progress": round(progress, 4),
        "total_duration": round(total, 1),
        "participants": [participant],
        "evidence": evidence,
        "flags": flags,
        "transcript": transcript,
        "flagged_segments": flagged_segments,
        "verdict_reasons": verdict_reasons,
        "timelines": timelines,
        "alert": alert,
        "evidence_count": len(engine.evidence_ledger),
        "packet_index": len(engine.evidence_ledger),
        "total_packets": 0,
        "live_state": status.get("state", "idle"),
        "video_title": status.get("video_title", ""),
        "p95_latency_ms": status.get("p95_latency_ms", 0.0),
        "dropped_non_candidate": status.get("dropped_non_candidate", 0),
        "explainability": _build_explainability(engine, top_id, participant, sess),
    }


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class LoadScenarioRequest(BaseModel):
    scenario_id: str

class SeekRequest(BaseModel):
    seconds: float

class LiveStartRequest(BaseModel):
    file_path: Optional[str] = None
    youtube_url: Optional[str] = None
    candidate_name: str = "Candidate"

class FeedbackConfirmRequest(BaseModel):
    participant_id: str
    notes: str = ""

class FeedbackCorrectRequest(BaseModel):
    participant_id: str
    notes: str = ""

class OperatorNoteRequest(BaseModel):
    text: str
    candidate_id: Optional[str] = None

class SpeedRequest(BaseModel):
    speed: float


# ============================================================================
# SERIALIZATION HELPERS
# ============================================================================

def serialize_snapshot(replay: SessionReplay) -> Dict:
    """Serialize the current snapshot + replay state to JSON-safe dict."""
    try:
        snap = replay.get_current_snapshot()
    except ValueError:
        return {"status": "no_data", "participants": []}

    participants = []
    for pid, belief in snap.beliefs.items():
        participant = replay.fixture.meeting_context.participants.get(pid)
        name = participant.display_name if participant else pid
        is_candidate = (pid == snap.top_candidate_id and snap.status == "identified")
        is_speaker = (pid == snap.current_speaker_id)

        flags = [ep for ep in snap.evidence_ledger
                 if ep.target_participant_id == pid
                 and ep.severity not in (FlagSeverity.NONE,)]

        participants.append({
            "id": pid,
            "name": name,
            "identity_probability": round(belief.identity_probability, 4),
            "authenticity_probability": round(belief.authenticity_probability, 4),
            "identity_log_odds": round(belief.identity_log_odds, 3),
            "evidence_count": len(belief.identity_evidence) + len(belief.authenticity_evidence),
            "is_candidate": is_candidate,
            "is_speaker": is_speaker,
            "flag_count": len(flags),
            "email": participant.email if participant else None,
            "join_time": participant.join_time.strftime('%H:%M:%S') if participant and participant.join_time else None,
            "webcam_on": participant.webcam_on if participant else False,
            "is_screen_sharing": participant.is_screen_sharing if participant else False,
            "device_name": participant.device_name if participant else None,
            "display_name_history": participant.display_name_history if participant else [],
        })

    # Evidence ledger
    evidence = []
    for ep in snap.evidence_ledger[-50:]:
        p = replay.fixture.meeting_context.participants.get(ep.target_participant_id)
        evidence.append({
            "source": ep.source.value,
            "axis": ep.axis.value,
            "target_id": ep.target_participant_id,
            "target_name": p.display_name if p else ep.target_participant_id,
            "delta_log_odds": round(ep.delta_log_odds, 4),
            "confidence": round(ep.confidence, 3),
            "rationale": ep.rationale,
            "timestamp": ep.timestamp.isoformat(),
            "time_display": ep.timestamp.strftime('%H:%M:%S'),
            "severity": ep.severity.value,
            "flag_type": ep.flag_type,
            "recommendation": ep.recommendation,
            "category": get_signal_category(ep.source),
        })

    # Flags only
    flags = [e for e in evidence if e["severity"] != "none"]

    # Transcript
    transcript = []
    for seg in snap.transcript_segments[-20:]:
        p = replay.fixture.meeting_context.participants.get(seg.participant_id)
        transcript.append({
            "participant_id": seg.participant_id,
            "name": p.display_name if p else seg.participant_id,
            "text": seg.text,
            "start_time": seg.start_time.isoformat(),
            "time_display": seg.start_time.strftime('%H:%M:%S'),
            "is_question": seg.is_question,
        })

    # Participant timeline data (for charts)
    timelines = {}
    for pid in snap.beliefs.keys():
        hist = replay.get_participant_timeline(pid, "identity_probability")
        if hist:
            timelines[pid] = [
                {"time": t.isoformat(), "value": round(v, 4)}
                for t, v in hist
            ]

    # Critical alert
    critical_flags = [ep for ep in snap.evidence_ledger
                      if ep.severity == FlagSeverity.CRITICAL]
    alert = None
    if critical_flags:
        f = critical_flags[-1]
        p = replay.fixture.meeting_context.participants.get(f.target_participant_id)
        alert = {
            "message": f.rationale[:150],
            "participant": p.display_name if p else f.target_participant_id,
        }

    return {
        "status": snap.status,
        "top_candidate_id": snap.top_candidate_id,
        "top_candidate_probability": round(snap.top_candidate_probability, 4),
        "ambiguity_gap": round(snap.ambiguity_gap, 4),
        "current_speaker_id": snap.current_speaker_id,
        "elapsed_seconds": round(snap.elapsed_seconds, 1),
        "timestamp": snap.timestamp.isoformat(),
        "progress": round(replay.get_progress(), 4),
        "total_duration": round(replay.get_total_duration(), 1),
        "participants": participants,
        "evidence": evidence,
        "flags": flags,
        "transcript": transcript,
        "timelines": timelines,
        "alert": alert,
        "evidence_count": len(snap.evidence_ledger),
        "packet_index": replay.current_index,
        "total_packets": len(replay.fixture.evidence_packets) if replay.fixture else 0,
    }


def generate_full_report(replay: SessionReplay) -> str:
    """Generate an HTML report string (ported from Streamlit app)."""
    try:
        snap = replay.get_current_snapshot()
    except ValueError:
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

    lines.append("<h1>🔍 Sherlock — Candidate Identification Report</h1>")
    lines.append(f"<p>Meeting: <strong>{ctx.meeting_id}</strong> | "
                 f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>")

    status_color = {"identified": "#66bb6a", "ambiguous": "#ffd54f", "no_participants": "gray"}.get(snap.status, "gray")
    lines.append(f"<h2 style='color:{status_color};'>{snap.status.upper()}</h2>")
    if snap.top_candidate_id:
        p = ctx.participants.get(snap.top_candidate_id)
        tn = p.display_name if p else snap.top_candidate_id
        lines.append(f"<p>Top candidate: <strong>{tn}</strong> — {snap.top_candidate_probability:.1%} confidence "
                     f"(gap: {snap.ambiguity_gap:.1%})</p>")

    lines.append("<h2>Participants</h2>")
    lines.append("<table><tr><th>Name</th><th>Identity</th><th>Authenticity</th><th>Evidence</th><th>Status</th></tr>")
    for pid, belief in sorted(snap.beliefs.items(), key=lambda x: x[1].identity_probability, reverse=True):
        p = ctx.participants.get(pid)
        name = p.display_name if p else pid
        is_top = pid == snap.top_candidate_id and snap.status == "identified"
        status_str = "🎯 CANDIDATE" if is_top else ""
        lines.append(f"<tr><td>{name}</td><td>{belief.identity_probability:.1%}</td>"
                     f"<td>{belief.authenticity_probability:.1%}</td>"
                     f"<td>{len(belief.identity_evidence) + len(belief.authenticity_evidence)}</td>"
                     f"<td>{status_str}</td></tr>")
    lines.append("</table>")

    flag_entries = [ep for ep in snap.evidence_ledger if ep.severity not in (FlagSeverity.NONE,)]
    if flag_entries:
        lines.append("<h2>🚨 Flags & Alerts</h2>")
        lines.append("<table><tr><th>Severity</th><th>Source</th><th>Participant</th><th>Delta</th><th>Rationale</th></tr>")
        for ep in reversed(flag_entries[-30:]):
            sev_cls = "flag-c" if ep.severity == FlagSeverity.CRITICAL else "flag-w"
            p = ctx.participants.get(ep.target_participant_id)
            name = p.display_name if p else ep.target_participant_id
            delta_cls = "sig-pos" if ep.delta_log_odds > 0 else "sig-neg"
            lines.append(f"<tr class='{sev_cls}'><td>{ep.severity.value.upper()}</td><td>{ep.source.value}</td>"
                         f"<td>{name}</td><td class='{delta_cls}'>{ep.delta_log_odds:+.3f}</td>"
                         f"<td>{ep.rationale[:100]}</td></tr>")
        lines.append("</table>")

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
    lines.append("</table></body></html>")
    return "\n".join(lines)


# ============================================================================
# ROOT / FRONTEND ROUTES
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("<h1>Frontend not found</h1><p>Create frontend/index.html</p>")


# ============================================================================
# REST ENDPOINTS: SCENARIOS
# ============================================================================

@app.get("/api/scenarios")
async def list_scenarios():
    """List all available fixture scenarios."""
    fixtures = list_available_fixtures(FIXTURES_DIR)
    return [{
        "id": f["id"],
        "name": f["name"],
        "description": f["description"],
    } for f in fixtures]


# ============================================================================
# REST ENDPOINTS: REPLAY
# ============================================================================

@app.post("/api/replay/load")
async def load_scenario(req: LoadScenarioRequest):
    """Load a scenario by ID."""
    fixture_path = FIXTURES_DIR / f"{req.scenario_id}.json"
    if not fixture_path.exists():
        raise HTTPException(status_code=404, detail=f"Scenario '{req.scenario_id}' not found")

    fixture = load_fixture(fixture_path)
    state.replay = SessionReplay(fixture)
    state.feedback_loop = FeedbackLoop(state.replay.engine)
    state.is_playing = False
    state.operator_notes = []

    return {
        "success": True,
        "meeting_id": fixture.meeting_context.meeting_id,
        "participant_count": len(fixture.meeting_context.participants),
        "evidence_count": len(fixture.evidence_packets),
        "transcript_count": len(fixture.transcript_segments),
        "total_duration": round(state.replay.get_total_duration(), 1),
    }


@app.post("/api/replay/step")
async def step_forward():
    """Advance one evidence packet."""
    if not state.replay:
        raise HTTPException(status_code=400, detail="No scenario loaded")

    snapshot = state.replay.step_forward()
    if snapshot is None:
        return {"done": True, "snapshot": serialize_snapshot(state.replay)}

    return {"done": False, "snapshot": serialize_snapshot(state.replay)}


@app.post("/api/replay/reset")
async def reset_replay():
    """Reset replay to beginning."""
    if not state.replay:
        raise HTTPException(status_code=400, detail="No scenario loaded")

    state.replay.reset()
    state.is_playing = False
    return {"success": True}


@app.post("/api/replay/seek")
async def seek_replay(req: SeekRequest):
    """Seek to specific time offset in seconds."""
    if not state.replay:
        raise HTTPException(status_code=400, detail="No scenario loaded")

    if not state.replay.start_time:
        raise HTTPException(status_code=400, detail="Replay not initialized")

    target_time = state.replay.start_time + timedelta(seconds=req.seconds)
    state.replay.advance_to_time(target_time)
    return {"snapshot": serialize_snapshot(state.replay)}


@app.get("/api/replay/snapshot")
async def get_snapshot():
    """Get current replay snapshot."""
    if not state.replay:
        raise HTTPException(status_code=400, detail="No scenario loaded")

    return serialize_snapshot(state.replay)


@app.post("/api/replay/speed")
async def set_speed(req: SpeedRequest):
    """Set playback speed."""
    state.playback_speed = max(0.5, min(4.0, req.speed))
    if state.replay:
        state.replay.set_playback_speed(state.playback_speed)
    return {"speed": state.playback_speed}


# ============================================================================
# REST ENDPOINTS: LIVE A/V ANALYSIS
# ============================================================================

@app.get("/api/live/available")
async def live_available():
    """Check if live analysis is available."""
    return {"available": _LIVE_AVAILABLE}


@app.post("/api/live/start")
async def start_live(req: LiveStartRequest):
    """Start live A/V analysis."""
    if not _LIVE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Live analysis dependencies not available")

    # If a session is already running, stop it cleanly so the new video takes over.
    if state.live_session and state.live_session.running:
        state.live_session.stop()
        state.live_session = None

    # Resolve uploaded file by token if provided.
    file_path = req.file_path or None
    if file_path and file_path.startswith("upload://"):
        token = file_path.replace("upload://", "")
        uploaded = state.uploaded_files.get(token)
        if uploaded and uploaded.exists():
            file_path = str(uploaded)
        else:
            raise HTTPException(status_code=404, detail="Uploaded file not found")

    # Validate local file path early so the thread doesn't crash silently.
    if file_path and not Path(file_path).exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {file_path}")

    sess = LiveSession()
    sess.start(
        file_path=file_path,
        youtube_url=req.youtube_url or None,
        candidate_name=req.candidate_name,
    )
    state.live_session = sess
    state.feedback_loop = FeedbackLoop(sess.engine)

    # Determine playable video path for the frontend.
    video_path = None
    video_title = ""
    if sess.youtube_info:
        video_path = Path(sess.youtube_info.file_path)
        video_title = sess.youtube_info.title
    elif file_path:
        video_path = Path(file_path)
        video_title = video_path.name

    state.live_video_path = video_path
    state.live_video_title = video_title

    return {
        "success": True,
        "state": "running",
        "video_url": "/api/live/video",
        "video_title": video_title,
    }


@app.post("/api/live/upload")
async def upload_live_video(file: UploadFile = File(...)):
    """Upload a video file for live analysis and return a token."""
    filename = file.filename or "video.mp4"
    stem = Path(filename).stem
    ext = Path(filename).suffix or ".mp4"
    token = f"{int(time.time())}_{stem}"
    dest = UPLOAD_DIR / f"{token}{ext}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    state.uploaded_files[token] = dest
    return {
        "success": True,
        "token": token,
        "filename": file.filename,
        "file_path": f"upload://{token}",
        "size_bytes": dest.stat().st_size,
    }


@app.get("/api/live/video")
async def serve_live_video():
    """Serve the current live analysis video file."""
    if not state.live_video_path or not state.live_video_path.exists():
        raise HTTPException(status_code=404, detail="No live video available")

    path = state.live_video_path
    media_type, _ = mimetypes.guess_type(str(path))
    media_type = media_type or "video/mp4"
    return FileResponse(str(path), media_type=media_type, filename=path.name)


@app.get("/api/live/info")
async def live_info():
    """Get information about the current live video."""
    return {
        "running": bool(state.live_session and state.live_session.running),
        "video_url": "/api/live/video" if state.live_video_path else None,
        "video_title": state.live_video_title,
        "video_path": str(state.live_video_path) if state.live_video_path else None,
    }


@app.post("/api/live/stop")
async def stop_live():
    """Stop live analysis."""
    if state.live_session:
        state.live_session.stop()
        state.live_session = None
    state.live_video_path = None
    state.live_video_title = ""
    return {"success": True}


@app.get("/api/live/status")
async def live_status():
    """Get live analysis status as a dashboard-compatible snapshot."""
    if not state.live_session:
        return {"state": "idle", "confidence": 0, "flags": [], "status": "no_data"}

    sess = state.live_session
    sess.refresh_status()
    snapshot = serialize_live_snapshot(sess)
    snapshot["playback_state"] = "live"
    return snapshot


# ============================================================================
# REST ENDPOINTS: FEEDBACK
# ============================================================================

@app.post("/api/feedback/confirm")
async def confirm_candidate(req: FeedbackConfirmRequest):
    """Confirm the top candidate."""
    if not state.feedback_loop:
        raise HTTPException(status_code=400, detail="No scenario loaded")

    state.feedback_loop.record_confirmation(req.participant_id, notes=req.notes)
    return {"success": True}


@app.post("/api/feedback/correct")
async def correct_candidate(req: FeedbackCorrectRequest):
    """Correct to a different participant."""
    if not state.feedback_loop:
        raise HTTPException(status_code=400, detail="No scenario loaded")

    state.feedback_loop.record_correction(req.participant_id, notes=req.notes)
    return {"success": True}


@app.post("/api/feedback/note")
async def save_note(req: OperatorNoteRequest):
    """Save an operator note."""
    note = {
        "text": req.text,
        "timestamp": datetime.utcnow().isoformat(),
        "candidate_id": req.candidate_id,
    }
    state.operator_notes.append(note)
    return {"success": True, "note": note}


@app.get("/api/feedback/notes")
async def get_notes():
    """Get all operator notes."""
    return state.operator_notes


@app.get("/api/feedback/summary")
async def feedback_summary():
    """Get feedback/calibration summary."""
    if not state.feedback_loop:
        return {"summary": "No feedback recorded yet."}

    return {"summary": state.feedback_loop.get_feedback_summary()}


# ============================================================================
# REST ENDPOINTS: REPORT
# ============================================================================

@app.get("/api/report")
async def export_report():
    """Export full HTML report."""
    if not state.replay:
        raise HTTPException(status_code=400, detail="No scenario loaded")

    html = generate_full_report(state.replay)
    return HTMLResponse(content=html)


@app.get("/api/report/json")
async def export_evidence_json():
    """Export evidence ledger as JSON."""
    if not state.replay:
        raise HTTPException(status_code=400, detail="No scenario loaded")

    try:
        snap = state.replay.get_current_snapshot()
    except ValueError:
        return []

    return [{
        "source": ep.source.value,
        "axis": ep.axis.value,
        "target_participant_id": ep.target_participant_id,
        "delta_log_odds": ep.delta_log_odds,
        "confidence": ep.confidence,
        "rationale": ep.rationale,
        "timestamp": ep.timestamp.isoformat(),
        "severity": ep.severity.value,
        "flag_type": ep.flag_type,
    } for ep in snap.evidence_ledger]


# ============================================================================
# WEBSOCKET: REPLAY (real-time score streaming)
# ============================================================================

@app.websocket("/ws/replay")
async def ws_replay(websocket: WebSocket):
    """
    WebSocket for real-time replay streaming.

    The server advances through evidence packets at time intervals proportional
    to the actual timestamp gaps between packets, divided by playback speed.
    This ensures scores evolve in sync with the video duration.

    Client can send JSON commands:
      {"action": "play"}
      {"action": "pause"}
      {"action": "speed", "value": 2.0}
      {"action": "step"}
      {"action": "reset"}
      {"action": "seek", "seconds": 30.0}
    """
    await websocket.accept()
    logger.info("WebSocket /ws/replay connected")

    is_playing = False

    async def listen_commands():
        nonlocal is_playing
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    cmd = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                action = cmd.get("action", "")

                if action == "play":
                    is_playing = True
                elif action == "pause":
                    is_playing = False
                elif action == "speed":
                    state.playback_speed = max(0.5, min(4.0, cmd.get("value", 1.0)))
                    if state.replay:
                        state.replay.set_playback_speed(state.playback_speed)
                elif action == "step":
                    if state.replay:
                        state.replay.step_forward()
                        snapshot = serialize_snapshot(state.replay)
                        snapshot["playback_state"] = "paused"
                        await websocket.send_json(snapshot)
                elif action == "reset":
                    if state.replay:
                        state.replay.reset()
                        is_playing = False
                        await websocket.send_json({
                            "status": "no_data",
                            "participants": [],
                            "playback_state": "stopped",
                        })
                elif action == "seek":
                    if state.replay and state.replay.start_time:
                        secs = cmd.get("seconds", 0)
                        target = state.replay.start_time + timedelta(seconds=secs)
                        state.replay.advance_to_time(target)
                        snapshot = serialize_snapshot(state.replay)
                        snapshot["playback_state"] = "paused" if not is_playing else "playing"
                        await websocket.send_json(snapshot)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning("WS command listener error: %s", e)

    # Start command listener in background
    cmd_task = asyncio.create_task(listen_commands())

    try:
        while True:
            if is_playing and state.replay:
                # Get next packet's timestamp to calculate delay
                replay = state.replay
                if replay.current_index < len(replay.fixture.evidence_packets):
                    current_packet = replay.fixture.evidence_packets[replay.current_index]

                    # Calculate time gap to next packet
                    if replay.current_index > 0:
                        prev_packet = replay.fixture.evidence_packets[replay.current_index - 1]
                        gap_seconds = (current_packet.timestamp - prev_packet.timestamp).total_seconds()
                    else:
                        gap_seconds = 0.5  # Initial delay

                    # Scale by playback speed, clamp to reasonable range
                    delay = max(0.1, min(3.0, gap_seconds / state.playback_speed))
                    await asyncio.sleep(delay)

                    # Advance
                    snapshot_obj = replay.step_forward()
                    if snapshot_obj is None:
                        is_playing = False
                        snapshot = serialize_snapshot(replay)
                        snapshot["playback_state"] = "finished"
                        await websocket.send_json(snapshot)
                    else:
                        snapshot = serialize_snapshot(replay)
                        snapshot["playback_state"] = "playing"
                        snapshot["playback_speed"] = state.playback_speed
                        await websocket.send_json(snapshot)
                else:
                    is_playing = False
                    snapshot = serialize_snapshot(replay)
                    snapshot["playback_state"] = "finished"
                    await websocket.send_json(snapshot)
            else:
                await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        logger.info("WebSocket /ws/replay disconnected")
    except Exception as e:
        logger.error("WS replay error: %s", e)
    finally:
        cmd_task.cancel()


# ============================================================================
# WEBSOCKET: LIVE A/V ANALYSIS
# ============================================================================

@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    """WebSocket for live A/V analysis updates (streams dashboard snapshots ~3×/s)."""
    await websocket.accept()
    logger.info("WebSocket /ws/live connected")

    try:
        while True:
            if state.live_session and state.live_session.running:
                sess = state.live_session
                sess.refresh_status()
                snapshot = serialize_live_snapshot(sess)
                snapshot["playback_state"] = "live"
                await websocket.send_json(snapshot)
            else:
                await websocket.send_json({"state": "idle", "status": "no_data"})
            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        logger.info("WebSocket /ws/live disconnected")
    except Exception as e:
        logger.error("WS live error: %s", e)


# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
