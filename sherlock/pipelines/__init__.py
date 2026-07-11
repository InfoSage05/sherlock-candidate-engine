"""Real-time fraud-detection pipelines for Sherlock.

Each pipeline turns candidate audio/video (or, for transcription, all
participants' audio) into ``EvidencePacket`` / ``TranscriptSegment`` objects
that feed the existing ``FusionEngine``.
"""

from __future__ import annotations

from .base import BaseAuthenticityPipeline, confidence_to_log_odds, severity_from_confidence
from .model_cache import ModelCache, default_cache
from .deepfake import (
    DeepfakeVideoPipeline,
    DeepfakeDetector,
    HeuristicDeepfakeDetector,
)
from .real_detectors import (
    RealDeepfakeDetector,
    RealVoiceLivenessDetector,
    RealGazeDetector,
    resemblyzer_embedding,
)
from .voice_liveness import (
    VoiceLivenessPipeline,
    VoiceLivenessDetector,
    HeuristicVoiceLivenessDetector,
)
from .speaker_store import SpeakerStore
from .gaze_cv import GazeBehavioralPipeline, GazeDetector, HeuristicGazeDetector
from .transcription_live import (
    LiveTranscriptionPipeline,
    Transcriber,
    WhisperTranscriber,
)

__all__ = [
    "BaseAuthenticityPipeline",
    "confidence_to_log_odds",
    "severity_from_confidence",
    "ModelCache",
    "default_cache",
    "DeepfakeVideoPipeline",
    "DeepfakeDetector",
    "HeuristicDeepfakeDetector",
    "RealDeepfakeDetector",
    "RealVoiceLivenessDetector",
    "RealGazeDetector",
    "resemblyzer_embedding",
    "VoiceLivenessPipeline",
    "VoiceLivenessDetector",
    "HeuristicVoiceLivenessDetector",
    "SpeakerStore",
    "GazeBehavioralPipeline",
    "GazeDetector",
    "HeuristicGazeDetector",
    "LiveTranscriptionPipeline",
    "Transcriber",
    "WhisperTranscriber",
]
