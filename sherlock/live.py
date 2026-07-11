"""Live A/V analysis session (experimental, optional).

Wraps the ``RealtimeInferenceOrchestrator`` so it can be launched from a
background thread (e.g. by the Streamlit demo) or from the CLI. The orchestrator
runs its own asyncio event loop; this class exposes a simple start/stop/status
API and keeps the latest identification + authenticity flags readable from the
main thread.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime
from typing import Optional

from .fusion import FusionEngine
from .ingestion import FileSource
from .models import (
    EvidencePacket,
    MeetingContext,
    Participant,
    SignalAxis,
    SignalSource,
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
        self.status = {
            "state": "idle",
            "top_candidate_id": None,
            "confidence": 0.0,
            "flags": [],
            "p95_latency_ms": 0.0,
            "dropped_non_candidate": 0,
        }

    # ------------------------------------------------------------------ #
    def build(self, file_path: Optional[str] = None,
              candidate_name: str = "Candidate",
              participant_id: str = "candidate") -> FusionEngine:
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
            engine=engine, source=source, context=context
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
              candidate_name: str = "Candidate",
              participant_id: str = "candidate") -> None:
        if self.running:
            return
        self.build(file_path, candidate_name, participant_id)
        self.running = True
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
        )
        return self.status
