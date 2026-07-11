"""Voice-liveness / anti-cloning pipeline.

Analyzes candidate **audio** for replayed, synthesized, or cloned speech. A
real backend should compute speaker embeddings (e.g. ``resemblyzer`` /
``speechbrain``), detect replay artifacts, and run an anti-spoofing classifier
(e.g. RawNet2 / AASIST). This module ships with a pure-numpy heuristic backend
that flags obviously low-energy / clipped audio as weak liveness cues. Inject a
real ``VoiceLivenessDetector`` for production use.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from ..models import CandidateMediaFrame, SignalSource
from .base import BaseAuthenticityPipeline
from .speaker_store import SpeakerStore

logger = logging.getLogger(__name__)


class VoiceLivenessDetector:
    """Protocol: ``detect(pcm: np.ndarray) -> (fake_score, metadata)``. """

    def detect(self, pcm: np.ndarray) -> Tuple[float, dict]:
        raise NotImplementedError


class HeuristicVoiceLivenessDetector(VoiceLivenessDetector):
    def detect(self, pcm: np.ndarray) -> Tuple[float, dict]:
        if pcm.size == 0:
            return 0.0, {"energy": 0.0}
        # Clipped / saturated audio is a weak replay/synthesis cue.
        peak = float(np.abs(pcm).max()) / 32768.0
        clip_ratio = float(np.mean(np.abs(pcm) > 30000)) if pcm.size else 0.0
        score = float(np.clip((peak > 0.98) * 0.5 + clip_ratio * 2.0, 0.0, 1.0))
        return score, {"peak": peak, "clip_ratio": clip_ratio}


class VoiceLivenessPipeline(BaseAuthenticityPipeline):
    source = SignalSource.VOICE_LIVENESS

    def __init__(self, context=None, detector: Optional[VoiceLivenessDetector] = None,
                 speaker_store: Optional[SpeakerStore] = None,
                 confidence_threshold: float = 0.65):
        super().__init__(
            context=context,
            detector=detector or HeuristicVoiceLivenessDetector(),
            confidence_threshold=confidence_threshold,
        )
        self.speaker_store = speaker_store or SpeakerStore()

    def process(self, frame: CandidateMediaFrame):
        if frame.audio_chunk is None:
            return None
        pcm = np.frombuffer(frame.audio_chunk, dtype=np.int16)
        if pcm.size == 0:
            return None
        score, meta = self.detector.detect(pcm)
        # Cross-check against the candidate's own earlier-call baseline.
        emb = self._embedding(pcm)
        if emb is not None:
            if not self.speaker_store.has_baseline(frame.candidate_id):
                self.speaker_store.update(frame.candidate_id, emb)
            else:
                drift = self.speaker_store.drift(frame.candidate_id, emb)
                meta["speaker_drift"] = float(drift)
                # Large drift on the same session boosts the fake score.
                score = float(np.clip(score + drift * 0.3, 0.0, 1.0))
        if score < self.confidence_threshold:
            return None
        return self._build_packet(
            target_participant_id=frame.candidate_id,
            confidence=score,
            rationale=(
                f"Audio liveness/clone suspicion {score:.2f}"
                + (f" (speaker_drift={meta['speaker_drift']:.2f})"
                   if "speaker_drift" in meta else "")
            ),
            recommendation="Verify the candidate's voice is live, not replayed/cloned.",
            metadata={"fake_score": score, **meta},
        )

    @staticmethod
    def _embedding(pcm: np.ndarray) -> Optional[np.ndarray]:
        # Placeholder embedding: MFCC-free energy contour summary. Replace with
        # a real speaker-embedding model (resemblyzer/speechbrain) in prod.
        if pcm.size < 1600:
            return None
        win = pcm.reshape(-1, 1600)
        return win.std(axis=1).astype(np.float32)
