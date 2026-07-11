from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Union

from .models import TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionConfig:
    model_size: str = "base"
    device: str = "auto"
    compute_type: str = "default"
    language: Optional[str] = None
    beam_size: int = 5
    vad_filter: bool = True
    vad_parameters: Optional[dict] = None
    preserve_disfluencies: bool = True
    condition_on_previous_text: bool = False
    initial_prompt: Optional[str] = None


@dataclass
class RawTranscriptionResult:
    text: str
    start_seconds: float
    end_seconds: float
    language: Optional[str] = None
    language_probability: Optional[float] = None


class TranscriptionPipeline:
    def __init__(self, config: Optional[TranscriptionConfig] = None):
        self.config = config or TranscriptionConfig()
        self._model = None
        self._meeting_start_time: Optional[datetime] = None

    def set_meeting_start_time(self, start_time: datetime) -> None:
        self._meeting_start_time = start_time

    def _load_model(self):
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            logger.warning("faster-whisper not available, falling back to openai-whisper")
            try:
                import whisper
                self._model = whisper.load_model(self.config.model_size)
                self._use_faster_whisper = False
                return
            except ImportError:
                raise ImportError(
                    "Neither faster-whisper nor openai-whisper is installed. "
                    "Install with: pip install faster-whisper"
                )

        self._model = WhisperModel(
            self.config.model_size,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )
        self._use_faster_whisper = True

    def transcribe_audio(
        self,
        audio_path: Union[str, Path],
        participant_id: str,
        meeting_start_offset: float = 0.0,
    ) -> List[TranscriptSegment]:
        self._load_model()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return []

        segments = []

        try:
            if self._use_faster_whisper:
                segments = self._transcribe_faster_whisper(
                    audio_path, participant_id, meeting_start_offset
                )
            else:
                segments = self._transcribe_openai_whisper(
                    audio_path, participant_id, meeting_start_offset
                )
        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            return []

        return segments

    def _transcribe_faster_whisper(
        self,
        audio_path: Path,
        participant_id: str,
        meeting_start_offset: float,
    ) -> List[TranscriptSegment]:
        segments = []

        transcribe_kwargs = {
            "beam_size": self.config.beam_size,
            "vad_filter": self.config.vad_filter,
            "condition_on_previous_text": self.config.condition_on_previous_text,
        }

        if self.config.language:
            transcribe_kwargs["language"] = self.config.language

        if self.config.vad_parameters:
            transcribe_kwargs["vad_parameters"] = self.config.vad_parameters

        if self.config.initial_prompt:
            transcribe_kwargs["initial_prompt"] = self.config.initial_prompt

        if self.config.preserve_disfluencies:
            transcribe_kwargs["word_timestamps"] = True

        segs, info = self._model.transcribe(str(audio_path), **transcribe_kwargs)

        for seg in segs:
            text = seg.text.strip()

            if not text:
                continue

            start_seconds = meeting_start_offset + seg.start
            end_seconds = meeting_start_offset + seg.end

            start_time = self._seconds_to_datetime(start_seconds)
            end_time = self._seconds_to_datetime(end_seconds)

            is_question = text.endswith("?") or self._looks_like_question(text)

            segments.append(TranscriptSegment(
                participant_id=participant_id,
                text=text,
                start_time=start_time,
                end_time=end_time,
                is_question=is_question,
            ))

        return segments

    def _transcribe_openai_whisper(
        self,
        audio_path: Path,
        participant_id: str,
        meeting_start_offset: float,
    ) -> List[TranscriptSegment]:
        segments = []

        transcribe_kwargs = {
            "beam_size": self.config.beam_size,
            "condition_on_previous_text": self.config.condition_on_previous_text,
        }

        if self.config.language:
            transcribe_kwargs["language"] = self.config.language

        if self.config.initial_prompt:
            transcribe_kwargs["initial_prompt"] = self.config.initial_prompt

        if not self.config.preserve_disfluencies:
            transcribe_kwargs["no_speech_threshold"] = 0.6

        result = self._model.transcribe(str(audio_path), **transcribe_kwargs)

        for seg in result.get("segments", []):
            text = seg.get("text", "").strip()

            if not text:
                continue

            start_seconds = meeting_start_offset + seg.get("start", 0.0)
            end_seconds = meeting_start_offset + seg.get("end", 0.0)

            start_time = self._seconds_to_datetime(start_seconds)
            end_time = self._seconds_to_datetime(end_seconds)

            is_question = text.endswith("?") or self._looks_like_question(text)

            segments.append(TranscriptSegment(
                participant_id=participant_id,
                text=text,
                start_time=start_time,
                end_time=end_time,
                is_question=is_question,
            ))

        return segments

    def _seconds_to_datetime(self, seconds: float) -> datetime:
        if self._meeting_start_time is None:
            return datetime.utcnow()

        return self._meeting_start_time + timedelta(seconds=seconds)

    def _looks_like_question(self, text: str) -> bool:
        question_indicators = [
            "can you", "could you", "tell me", "what", "how", "why", "when", "where",
            "who", "which", "do you", "did you", "have you", "are you", "would you",
            "please explain", "describe", "walk me through",
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in question_indicators)

    def transcribe_audio_buffer(
        self,
        audio_buffer: bytes,
        participant_id: str,
        meeting_start_offset: float = 0.0,
        sample_rate: int = 16000,
    ) -> List[TranscriptSegment]:
        import tempfile
        import soundfile as sf
        import numpy as np

        audio_array = np.frombuffer(audio_buffer, dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            sf.write(tmp_path, audio_array, sample_rate)

        try:
            segments = self.transcribe_audio(tmp_path, participant_id, meeting_start_offset)
        finally:
            tmp_path.unlink(missing_ok=True)

        return segments
