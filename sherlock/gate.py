"""Candidate stream gate.

After the identity engine has locked onto a candidate, this gate ensures that
ONLY the candidate's audio/video frames are forwarded to the fraud-detection
pipelines. Interviewer and observer frames are dropped. The gate also emits
operator-facing flags when identity is ambiguous or when the candidate changes
mid-call.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

from ..models import (
    CandidateMediaFrame,
    EvidencePacket,
    FlagSeverity,
    RawMediaFrame,
    SignalAxis,
    SignalSource,
)
from ..fusion import FusionEngine

logger = logging.getLogger(__name__)


class CandidateStreamGate:
    def __init__(
        self,
        engine: FusionEngine,
        ambiguity_threshold: Optional[float] = None,
    ) -> None:
        self.engine = engine
        self.ambiguity_threshold = ambiguity_threshold
        self.candidate_id: Optional[str] = None
        self._last_status: Optional[str] = None
        self._uncertainty_flagged = False
        self._pending_flags: List[EvidencePacket] = []
        self.dropped_frames_non_candidate: int = 0

    # ----- flag helpers ------------------------------------------------- #
    def _emit_flag(
        self,
        source: SignalSource,
        severity: FlagSeverity,
        rationale: str,
        recommendation: str = "",
    ) -> None:
        self._pending_flags.append(
            EvidencePacket(
                source=source,
                axis=SignalAxis.AUTHENTICITY,
                target_participant_id=self.candidate_id or "unknown",
                delta_log_odds=0.0,
                confidence=1.0,
                rationale=rationale,
                timestamp=__import__("datetime").datetime.utcnow(),
                severity=severity,
                flag_type=source.value,
                recommendation=recommendation,
            )
        )

    def consume_flags(self) -> List[EvidencePacket]:
        """Drain and return any pending operator flags."""
        flags, self._pending_flags = self._pending_flags, []
        return flags

    def _refresh_candidate(self) -> None:
        result = self.engine.get_result()
        status = result.status
        top_id = result.top_candidate_id

        if status != "identified" and not self._uncertainty_flagged:
            self._emit_flag(
                SignalSource.IDENTITY_UNCERTAIN,
                FlagSeverity.WARNING,
                f"Identity ambiguous (status={status}); not analyzing any "
                f"participant's media until the candidate is confirmed.",
                recommendation="Confirm the candidate before trusting A/V flags.",
            )
            self._uncertainty_flagged = True
            self.candidate_id = None
            return

        if status == "identified":
            self._uncertainty_flagged = False

        if top_id != self.candidate_id and top_id is not None:
            if self.candidate_id is not None:
                logger.info("Candidate switched: %s -> %s", self.candidate_id, top_id)
                self._emit_flag(
                    SignalSource.JOIN_TIMING,
                    FlagSeverity.INFO,
                    f"Candidate changed mid-call from {self.candidate_id} to {top_id}.",
                    recommendation="Verify the new candidate before acting on flags.",
                )
            self.candidate_id = top_id

    # ----- main entrypoint --------------------------------------------- #
    def process(self, frame: RawMediaFrame) -> Optional[CandidateMediaFrame]:
        """Return a ``CandidateMediaFrame`` if this frame belongs to the locked
        candidate, otherwise ``None`` (and count it as dropped)."""
        self._refresh_candidate()

        if self.candidate_id is None:
            self.dropped_frames_non_candidate += 1
            return None

        if frame.participant_id != self.candidate_id:
            self.dropped_frames_non_candidate += 1
            return None

        return CandidateMediaFrame(
            participant_id=frame.participant_id,
            audio_chunk=frame.audio_chunk,
            video_frame=frame.video_frame,
            timestamp_ms=frame.timestamp_ms,
            candidate_id=self.candidate_id,
        )
