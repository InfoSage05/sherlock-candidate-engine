"""Live transcription + diarization pipeline.

Replaces fixture transcripts with real, speaker-attributed transcript segments.
In the live path each ``RawMediaFrame`` is already tagged with a
``participant_id`` (the platform/ingestion layer knows who is speaking), so we
transcribe per-participant audio directly. The optional ``pyannote`` diarizer
is only needed when frames arrive without speaker tags.

Real transcription uses ``faster-whisper`` (already a dependency). For tests a
``Transcriber`` backend can be injected. A heuristic question detector tags
segments with ``is_question`` for the identity/behavioral signals.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import List, Optional

import numpy as np

from ..models import RawMediaFrame, TranscriptSegment


class Transcriber:
    """Protocol: ``transcribe(pcm: np.ndarray) -> Optional[str]``."""

    def transcribe(self, pcm: np.ndarray) -> Optional[str]:
        raise NotImplementedError


class WhisperTranscriber(Transcriber):
    """Lazy-loaded faster-whisper backend."""

    def __init__(self, model_size: str = "base"):
        self._model = None
        self._model_size = model_size

    def _ensure(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self._model_size, device="cpu")
        return self._model

    def transcribe(self, pcm: np.ndarray) -> Optional[str]:
        model = self._ensure()
        # faster-whisper expects float32 audio at 16 kHz.
        audio = pcm.astype(np.float32) / 32768.0
        segments, _ = model.transcribe(audio, language="en", beam_size=1)
        text = " ".join(s.text for s in segments).strip()
        return text or None


class LiveTranscriptionPipeline:
    def __init__(self, context=None, transcriber: Optional[Transcriber] = None,
                 buffer_seconds: float = 2.0, sample_rate: int = 16000):
        self.context = context
        self.transcriber = transcriber or WhisperTranscriber()
        self.buffer_seconds = buffer_seconds
        self.sample_rate = sample_rate
        self._buffers: dict = {}

    def _is_question(self, text: str) -> bool:
        t = text.strip()
        if t.endswith("?"):
            return True
        return bool(re.search(r"\b(what|why|how|who|when|where|can you|do you|are you)\b.*\??$",
                              t, re.IGNORECASE))

    def process(self, frame: RawMediaFrame) -> Optional[List[TranscriptSegment]]:
        if frame.audio_chunk is None:
            return None
        pcm = np.frombuffer(frame.audio_chunk, dtype=np.int16)
        if pcm.size == 0:
            return None
        buf = self._buffers.setdefault(frame.participant_id, [])
        buf.append(pcm)
        total = sum(b.size for b in buf)
        if total < self.buffer_seconds * self.sample_rate:
            return None
        combined = np.concatenate(buf)
        self._buffers[frame.participant_id] = []
        text = self.transcriber.transcribe(combined)
        if not text:
            return None
        now = datetime.utcnow()
        seg = TranscriptSegment(
            participant_id=frame.participant_id,
            text=text,
            start_time=now,
            end_time=now,
            is_question=self._is_question(text),
        )
        return [seg]
