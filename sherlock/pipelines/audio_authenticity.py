"""Audio-level AI-generated speech detection.

Detects whether a candidate's voice is live human speech or synthetically
generated / voice-cloned / TTS output by analysing raw audio.  The pipeline
first attempts to load an open-source anti-spoofing model (SpeechBrain AASIST
or RawNet2).  If the model is unavailable it falls back to lightweight
heuristic spectral/prosodic features so the pipeline always runs.

All outputs are emitted as AUTHENTICITY evidence packets with negative
log-odds when manipulation is suspected — i.e. a weak signal, never a verdict.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from ..models import CandidateMediaFrame, EvidencePacket, FlagSeverity, SignalAxis, SignalSource

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class AudioAuthenticityDetector:
    """Protocol: ``detect(pcm: np.ndarray, sr: int) -> (score, metadata)``."""

    def detect(self, pcm: np.ndarray, sr: int) -> Tuple[float, dict]:
        raise NotImplementedError


class SpeechBrainAntiSpoofingDetector(AudioAuthenticityDetector):
    """Open-source AASIST anti-spoofing via SpeechBrain.

    Source model:
      speechbrain/asvspoof-aasist  (pre-trained on ASVspoof 2021 LA)

    Returns a spoof score in [0, 1]; higher means more likely synthetic.
    """

    def __init__(self, model_source: str = "speechbrain/asvspoof-aasist") -> None:
        from speechbrain.pretrained import EncoderClassifier  # type: ignore

        self.classifier = EncoderClassifier.from_hparams(source=model_source)
        self._model_source = model_source

    def detect(self, pcm: np.ndarray, sr: int) -> Tuple[float, dict]:
        # speechbrain expects a file path; write a temporary WAV and clean up.
        import os
        import tempfile

        tmp_path = tempfile.mktemp(suffix=".wav")
        try:
            import soundfile as sf

            sf.write(tmp_path, pcm, sr)

            # Try the high-level classify_file API first.
            if hasattr(self.classifier, "classify_file"):
                try:
                    out, _, _, _ = self.classifier.classify_file(tmp_path)
                    # out is typically a logit tensor of shape (1, 1) or (1, 2).
                    import torch

                    if out.dim() > 1:
                        out = out.squeeze()
                    if out.numel() == 1:
                        bona_fide_score = float(torch.sigmoid(out))
                    else:
                        # Softmax over classes: assume class 0 = bona-fide.
                        probs = torch.softmax(out, dim=0)
                        bona_fide_score = float(probs[0])
                    spoof_score = 1.0 - bona_fide_score
                    return spoof_score, {"model": self._model_source, "bona_fide_score": bona_fide_score}
                except Exception:
                    pass

            # Fallback: encode_batch and interpret embedding norm.
            try:
                import torch

                waveform = self.classifier.load_audio(tmp_path)
                waveform = waveform.unsqueeze(0)
                with torch.no_grad():
                    emb = self.classifier.encode_batch(waveform)
                    score = float(torch.sigmoid(emb).mean())
                spoof_score = 1.0 - score
                return spoof_score, {"model": self._model_source, "bona_fide_score": score}
            except Exception as exc:
                raise RuntimeError(f"SpeechBrain detector inference failed: {exc}") from exc
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


class HeuristicAudioAuthenticityDetector(AudioAuthenticityDetector):
    """Dependency-free detector based on spectral/prosodic cues.

    Synthetic/TTS speech often shows:
      - unnaturally stable pitch / low jitter
      - flat spectral envelope
      - periodic excitation artifacts
      - absence of breath / mouth noise
      - clipped or over-normalised amplitude distribution
    """

    def detect(self, pcm: np.ndarray, sr: int) -> Tuple[float, dict]:
        if pcm.size < sr * 0.5:  # need at least 0.5s
            return 0.0, {"reason": "too_short"}

        # Normalise.
        float_pcm = pcm.astype(np.float32)
        peak = np.abs(float_pcm).max()
        if peak == 0:
            return 0.0, {"reason": "silent"}
        float_pcm /= peak

        # Frame-level features (25 ms frames, 10 ms hop).
        frame_len = int(0.025 * sr)
        hop_len = int(0.010 * sr)
        # Use librosa if available; otherwise manual RMS.
        frames = None
        try:
            import librosa

            rms = librosa.feature.rms(y=float_pcm, frame_length=frame_len, hop_length=hop_len)[0]
            f0, _, _ = librosa.pyin(
                float_pcm,
                fmin=65,
                fmax=300,
                sr=sr,
                frame_length=frame_len,
                hop_length=hop_len,
            )
            f0_valid = f0[~np.isnan(f0)]
        except Exception:
            # Manual framing for RMS fallback.
            num_frames = max(1, (float_pcm.size - frame_len) // hop_len + 1)
            rms = np.array([
                np.sqrt(np.mean(float_pcm[i * hop_len:i * hop_len + frame_len] ** 2))
                for i in range(int(num_frames))
            ])
            f0_valid = np.array([])

        # Feature 1: amplitude clipping / over-normalisation.
        clip_ratio = float(np.mean(np.abs(float_pcm) > 0.98))

        # Feature 2: RMS flatness (synthetic speech is often unnaturally steady).
        if rms.size > 1 and rms.mean() > 0:
            rms_cv = float(rms.std() / rms.mean())
        else:
            rms_cv = 1.0

        # Feature 3: pitch jitter (live human speech has higher jitter).
        if f0_valid.size > 10:
            # Relative perturbation of consecutive pitch periods.
            diffs = np.diff(f0_valid)
            jitter = float(np.mean(np.abs(diffs)) / (np.mean(f0_valid) + 1e-6))
        else:
            jitter = 0.05  # default moderate value

        # Feature 4: spectral centroid movement.
        try:
            import librosa

            spec = np.abs(librosa.stft(float_pcm, n_fft=512, hop_length=hop_len))
            centroid = librosa.feature.spectral_centroid(S=spec, sr=sr)[0]
            centroid_cv = float(centroid.std() / (centroid.mean() + 1e-6))
        except Exception:
            centroid_cv = 0.5

        # Combine into a spoof score in [0, 1].
        # Higher score = more synthetic.
        score = min(1.0, max(0.0,
            clip_ratio * 1.5
            + (1.0 - min(1.0, rms_cv / 0.4)) * 0.25
            + (1.0 - min(1.0, jitter / 0.08)) * 0.30
            + (1.0 - min(1.0, centroid_cv / 0.5)) * 0.20
        ))

        return score, {
            "clip_ratio": round(clip_ratio, 4),
            "rms_cv": round(rms_cv, 4),
            "jitter": round(jitter, 4),
            "centroid_cv": round(centroid_cv, 4),
            "detector": "heuristic",
        }


def _default_detector() -> AudioAuthenticityDetector:
    """Return the best available detector.

    By default we try the open-source SpeechBrain AASIST anti-spoofing model.
    If it fails to load, we fall back to the fast heuristic detector.
    Set ``SHERLOCK_USE_HEURISTIC_AUDIO=1`` to force the heuristic detector.
    """
    import os

    if os.environ.get("SHERLOCK_USE_HEURISTIC_AUDIO") != "1":
        try:
            return SpeechBrainAntiSpoofingDetector()
        except Exception as exc:
            logger.info(
                "Open-source SpeechBrain anti-spoofing model not available (%s); "
                "falling back to heuristic audio authenticity detector.",
                exc,
            )
    return HeuristicAudioAuthenticityDetector()


class AudioAuthenticityPipeline:
    """Processes candidate audio frames and emits AI-generated-speech signals."""

    source = SignalSource.AI_GENERATED_SPEECH

    def __init__(
        self,
        context=None,
        detector: Optional[AudioAuthenticityDetector] = None,
        window_duration_seconds: float = 3.0,
        confidence_threshold: float = 0.55,
    ) -> None:
        self.context = context
        self._detector: Optional[AudioAuthenticityDetector] = detector
        self.window_duration_seconds = window_duration_seconds
        self.confidence_threshold = confidence_threshold
        self._buffers: Dict[str, List[np.ndarray]] = {}

    @property
    def detector(self) -> AudioAuthenticityDetector:
        if self._detector is None:
            self._detector = _default_detector()
        return self._detector

    def process(self, frame: CandidateMediaFrame) -> Optional[List[EvidencePacket]]:
        if frame.audio_chunk is None:
            return None

        pcm = np.frombuffer(frame.audio_chunk, dtype=np.int16)
        if pcm.size == 0:
            return None

        pid = frame.candidate_id
        buf = self._buffers.setdefault(pid, [])
        buf.append(pcm)

        total_samples = sum(b.size for b in buf)
        if total_samples < self.window_duration_seconds * SAMPLE_RATE:
            return None

        combined = np.concatenate(buf)
        self._buffers[pid] = []

        try:
            score, meta = self.detector.detect(combined, SAMPLE_RATE)
        except Exception as exc:
            logger.warning("Audio authenticity detector failed: %s", exc)
            return None

        if score < self.confidence_threshold:
            return None

        severity = FlagSeverity.CRITICAL if score > 0.8 else FlagSeverity.WARNING
        delta = -(score - 0.4) * 2.5

        model_name = meta.get("model") or meta.get("detector", "unknown")
        rationale = (
            f"Audio suggests synthetic/AI-generated speech "
            f"(score={score:.2f}, detector={model_name})"
        )

        return [EvidencePacket(
            source=SignalSource.AI_GENERATED_SPEECH,
            axis=SignalAxis.AUTHENTICITY,
            target_participant_id=pid,
            delta_log_odds=delta,
            confidence=min(0.95, score),
            severity=severity,
            flag_type="ai_generated_speech",
            recommendation="Verify the candidate is speaking live and not playing pre-recorded or AI-generated audio.",
            rationale=rationale,
            timestamp=datetime.utcnow(),
            metadata={
                "spoof_score": round(score, 4),
                **meta,
            },
        )]
