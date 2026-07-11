"""Shared helpers for real-time authenticity pipelines.

Each pipeline turns a media frame into an ``EvidencePacket`` on the
``AUTHENTICITY`` axis. Because the heavy CV/ML models (OpenCV, MediaPipe,
InsightFace, torch, etc.) are optional, every pipeline accepts an
**injectable detector backend** and ships with a pure-Python heuristic backend
so it runs and is fully testable without those libraries. A real backend can
be supplied at construction time (e.g. lazy-loaded from ``model_cache``).
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

from ..models import (
    EvidencePacket,
    FlagSeverity,
    SignalAxis,
    SignalSource,
)

logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.5
MAX_CONFIDENCE = 0.98


def confidence_to_log_odds(confidence: float, is_fraud: bool) -> float:
    """Map a P(fraud) confidence to a (signed) delta log-odds.

    A high fraud confidence yields a large *negative* delta (authenticity
    drops); a confident "clean" reading yields a small positive delta.
    """
    confidence = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, confidence))
    log_odds = math.log(confidence / (1 - confidence))
    return -log_odds if is_fraud else log_odds


def severity_from_confidence(confidence: float) -> FlagSeverity:
    if confidence >= 0.85:
        return FlagSeverity.CRITICAL
    if confidence >= 0.65:
        return FlagSeverity.WARNING
    return FlagSeverity.INFO


class BaseAuthenticityPipeline:
    """Common boilerplate for pipelines that emit authenticity evidence."""

    source: SignalSource
    is_fraud_signal: bool = True

    def __init__(self, context=None, detector=None, confidence_threshold: float = 0.6):
        self.context = context
        self.detector = detector
        self.confidence_threshold = confidence_threshold

    def _build_packet(
        self,
        target_participant_id: str,
        confidence: float,
        rationale: str,
        recommendation: str = "",
        metadata: Optional[dict] = None,
    ) -> EvidencePacket:
        is_fraud = confidence >= self.confidence_threshold
        delta = confidence_to_log_odds(confidence, is_fraud)
        severity = (
            severity_from_confidence(confidence) if is_fraud else FlagSeverity.NONE
        )
        return EvidencePacket(
            source=self.source,
            axis=SignalAxis.AUTHENTICITY,
            target_participant_id=target_participant_id,
            delta_log_odds=delta,
            confidence=confidence,
            rationale=rationale,
            timestamp=__import__("datetime").datetime.utcnow(),
            severity=severity,
            flag_type=self.source.value,
            recommendation=recommendation,
            metadata=metadata or {},
        )
