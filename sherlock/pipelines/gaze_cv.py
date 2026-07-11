"""Behavioral computer-vision pipeline (gaze / off-screen reading).

Produces ``GazeEvent`` samples from candidate **video** frames and reuses the
existing ``AuthenticitySignalExtractor.extract_gaze_detection`` rules to emit
authenticity evidence. A real backend should use MediaPipe Face Mesh + L2CS-Net
to estimate the gaze vector. This module ships with a pure-numpy heuristic
backend (bright-pixel centroid) so it runs without those libraries. Inject a
real ``GazeDetector`` for production use.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from ..models import CandidateMediaFrame, GazeEvent, MeetingContext
from ..signals.authenticity import AuthenticitySignalExtractor
from .base import BaseAuthenticityPipeline

logger = logging.getLogger(__name__)


class GazeDetector:
    """Protocol: ``detect(frame) -> (gaze_vector, is_off_screen, saccade_periodicity)``.

    - ``gaze_vector`` : (x, y) normalized gaze direction.
    - ``is_off_screen``: True when looking away from the camera.
    - ``saccade_periodicity``: seconds between rhythmic saccades, or None.
    """

    def detect(self, frame: np.ndarray):
        raise NotImplementedError


class HeuristicGazeDetector(GazeDetector):
    def detect(self, frame: np.ndarray):
        f = np.asarray(frame, dtype=np.float32)
        gray = f.mean(axis=2)
        total = gray.sum()
        if total == 0:
            return (0.0, 0.0), False, None
        ys, xs = np.indices(gray.shape)
        cx = float((xs * gray).sum() / total) / gray.shape[1]
        cy = float((ys * gray).sum() / total) / gray.shape[0]
        # Centroid near the middle => looking at camera.
        off = (cx < 0.3 or cx > 0.7 or cy < 0.3 or cy > 0.7)
        return (cx, cy), off, None


class GazeBehavioralPipeline(BaseAuthenticityPipeline):
    source = None  # evidence is produced via AuthenticitySignalExtractor

    def __init__(self, context: MeetingContext, detector: Optional[GazeDetector] = None,
                 window: int = 30):
        self.context = context
        self.detector = detector or HeuristicGazeDetector()
        self._window: Deque[GazeEvent] = deque(maxlen=window)

    def process(self, frame: CandidateMediaFrame):
        if frame.video_frame is None:
            return None
        img = np.asarray(frame.video_frame)
        gaze_vec, off, periodicity = self.detector.detect(img)
        self._window.append(GazeEvent(
            participant_id=frame.candidate_id,
            timestamp=__import__("datetime").datetime.utcnow(),
            gaze_vector=tuple(gaze_vec),
            is_off_screen=off,
            saccade_periodicity=periodicity,
        ))
        if len(self._window) < 3:
            return None
        # Reuse the existing, tested gaze rule over the recent window.
        extractor = AuthenticitySignalExtractor(self.context)
        for ev in self._window:
            extractor.add_gaze_event(ev)
        packets = extractor.extract_gaze_detection()
        return packets if packets else None
