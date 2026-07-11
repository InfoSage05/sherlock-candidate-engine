"""Deepfake / face-manipulation detection pipeline.

Analyzes candidate **video** frames for synthetic or replayed faces. A real
backend should implement face-anti-spoofing, lip-sync consistency, and 3D
face-mesh stability (e.g. MediaPipe Face Mesh + a liveness classifier). This
module ships with a pure-numpy heuristic backend so it runs and is testable
without OpenCV/MediaPipe installed. Inject a real ``DeepfakeDetector`` to use
production models.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..models import CandidateMediaFrame, SignalSource
from .base import BaseAuthenticityPipeline


class DeepfakeDetector:
    """Protocol: ``detect(frame: np.ndarray) -> (fake_score, metadata)``.

    ``fake_score`` is P(frame is manipulated) in [0, 1].
    """

    def detect(self, frame: np.ndarray) -> Tuple[float, dict]:
        raise NotImplementedError


class HeuristicDeepfakeDetector(DeepfakeDetector):
    """Lightweight, dependency-free stand-in.

    Flags *extreme* frame artifacts (near-uniform flat color, or very low
    high-frequency texture) as weak manipulation cues. This is intentionally
    conservative and is NOT a real deepfake classifier.
    """

    def detect(self, frame: np.ndarray) -> Tuple[float, dict]:
        f = frame.astype(np.float32)
        gray = f.mean(axis=2)
        # High-frequency texture via Laplacian variance.
        lap = np.abs(gray[1:, :] - gray[:-1, :]) + np.abs(gray[:, 1:] - gray[:, :-1])
        texture = float(lap.var())
        # Flat (poster/photo) frames have very low texture variance.
        score = float(np.clip(0.7 - texture / 200.0, 0.0, 1.0))
        return score, {"texture_variance": texture}


def _default_detector() -> DeepfakeDetector:
    try:
        from .real_detectors import RealDeepfakeDetector

        return RealDeepfakeDetector()
    except Exception:
        return HeuristicDeepfakeDetector()


class DeepfakeVideoPipeline(BaseAuthenticityPipeline):
    source = SignalSource.DEEPFAKE_VIDEO

    def __init__(self, context=None, detector: Optional[DeepfakeDetector] = None,
                 confidence_threshold: float = 0.65):
        super().__init__(
            context=context,
            detector=detector or _default_detector(),
            confidence_threshold=confidence_threshold,
        )

    def process(self, frame: CandidateMediaFrame):
        if frame.video_frame is None:
            return None
        img = np.asarray(frame.video_frame)
        if img.ndim != 3:
            return None
        score, meta = self.detector.detect(img)
        if score < self.confidence_threshold:
            return None
        return self._build_packet(
            target_participant_id=frame.candidate_id,
            confidence=score,
            rationale=(
                f"Video manipulation likelihood {score:.2f} "
                f"(texture_variance={meta.get('texture_variance', 0):.1f})"
            ),
            recommendation="Verify the candidate's video is live and unaltered.",
            metadata={"fake_score": score, **meta},
        )
