"""Live A/V analysis session (experimental, optional).

Wraps the ``RealtimeInferenceOrchestrator`` so it can be launched from a
background thread (e.g. by the Streamlit demo) or from the CLI. The orchestrator
runs its own asyncio event loop; this class exposes a simple start/stop/status
API and keeps the latest identification + authenticity flags readable from the
main thread.

Supports:
- Local video files (MP4/AVI/etc.)
- YouTube URLs (auto-downloaded via yt-dlp)
- Synthetic streams (no file)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import List, Optional

from .fusion import FusionEngine
from .ingestion import FileSource
from .ingestion.youtube import YouTubeVideo, download_youtube_video, is_youtube_url
from .models import (
    EvidencePacket,
    MeetingContext,
    Participant,
    SignalAxis,
    SignalSource,
    SnapshotEntry,
)
from .orchestrator import RealtimeInferenceOrchestrator

logger = logging.getLogger(__name__)


class LiveSession:
    def __init__(self) -> None:
        self.engine: Optional[FusionEngine] = None
        self.orchestrator: Optional[RealtimeInferenceOrchestrator] = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.running = False
        self.youtube_info: Optional[YouTubeVideo] = None
        self._start_wall: Optional[float] = None
        self.status = {
            "state": "idle",
            "top_candidate_id": None,
            "confidence": 0.0,
            "flags": [],
            "p95_latency_ms": 0.0,
            "dropped_non_candidate": 0,
            "elapsed_seconds": 0.0,
            "total_duration": 0.0,
            "video_title": "",
            "timeline": [],   # list of {timestamp, confidence, status, flags_count}
            "evidence_log": [],  # recent evidence packets
        }

    # ------------------------------------------------------------------ #
    def build(self, file_path: Optional[str] = None,
              youtube_url: Optional[str] = None,
              candidate_name: str = "Candidate",
              participant_id: str = "candidate") -> FusionEngine:

        # Resolve YouTube URL -> local file.
        if youtube_url and is_youtube_url(youtube_url):
            logger.info("Downloading YouTube video: %s", youtube_url)
            self.youtube_info = download_youtube_video(youtube_url)
            file_path = self.youtube_info.file_path
            logger.info("Downloaded: %s -> %s", self.youtube_info.title, file_path)

        context = MeetingContext(
            meeting_id="live",
            candidate_name=candidate_name,
            participants={
                participant_id: Participant(id=participant_id, display_name=candidate_name)
            },
        )
        engine = FusionEngine(context)
        # Seed identity so the gate will pass candidate frames immediately.
        engine.ingest(EvidencePacket(
            source=SignalSource.CALENDAR_MATCH, axis=SignalAxis.IDENTITY,
            target_participant_id=participant_id, delta_log_odds=3.0, confidence=0.9,
            rationale="Live join as candidate", timestamp=datetime.utcnow(),
        ))
        source = FileSource(
            path=file_path, participant_id=participant_id,
            generate_synthetic=(file_path is None),
        )
        self.engine = engine
        self.orchestrator = RealtimeInferenceOrchestrator(
            engine=engine, source=source, context=context,
        )
        return engine

    # ------------------------------------------------------------------ #
    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self.orchestrator.start())
            self._loop.run_forever()
        finally:
            self._loop.close()
            self._loop = None

    def start(self, file_path: Optional[str] = None,
              youtube_url: Optional[str] = None,
              candidate_name: str = "Candidate",
              participant_id: str = "candidate") -> None:
        if self.running:
            return
        self.build(file_path, youtube_url, candidate_name, participant_id)
        self.running = True
        self._start_wall = time.time()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self.orchestrator and self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.orchestrator.stop(), self._loop)
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ------------------------------------------------------------------ #
    def refresh_status(self) -> dict:
        if not self.engine or not self.orchestrator:
            return self.status

        result = self.engine.get_result()
        elapsed = time.time() - self._start_wall if self._start_wall else 0.0

        # Total duration (from YouTube metadata or file source).
        total = 0.0
        if self.youtube_info:
            total = self.youtube_info.duration_seconds

        # Build timeline entry.
        timeline_entry = {
            "elapsed": round(elapsed, 1),
            "confidence": round(result.top_candidate_probability, 4),
            "status": result.status,
            "gap": round(result.ambiguity_gap, 4),
            "flags_count": len(self.engine._collect_active_flags()),
        }

        # Recent evidence log (last 10).
        recent_evidence = []
        for ep in self.engine.evidence_ledger[-10:]:
            recent_evidence.append({
                "source": ep.source.value,
                "axis": ep.axis.value,
                "delta_log_odds": round(ep.delta_log_odds, 3),
                "confidence": round(ep.confidence, 3),
                "rationale": ep.rationale[:120],
                "severity": ep.severity.value,
            })

        self.status.update(
            state="running" if self.running else "idle",
            top_candidate_id=result.top_candidate_id,
            confidence=result.top_candidate_probability,
            flags=[
                {
                    "source": ep.source.value,
                    "severity": ep.severity.value,
                    "rationale": ep.rationale,
                }
                for ep in self.engine._collect_active_flags()
            ],
            p95_latency_ms=self.orchestrator.p95_latency_ms(),
            dropped_non_candidate=self.orchestrator.gate.dropped_frames_non_candidate,
            elapsed_seconds=elapsed,
            total_duration=total,
            video_title=self.youtube_info.title if self.youtube_info else "",
            timeline=self.status.get("timeline", []) + [timeline_entry],
            evidence_log=recent_evidence,
        )
        return self.status
