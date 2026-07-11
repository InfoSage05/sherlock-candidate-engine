"""File-based ingestion source.

Replays a recorded ``.mp4`` / ``.wav`` file through PyAV, demuxing per-frame
video and per-packet audio into ``RawMediaFrame`` objects. When no file is
available (or ``generate_synthetic=True``) it synthesizes a moving object on a
black background with silent audio, which keeps the rest of the pipeline fully
testable in CI without any recording.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np

from ..models import RawMediaFrame
from .base import MediaSource

logger = logging.getLogger(__name__)

_DEFAULT_FPS = 25


class FileSource(MediaSource):
    def __init__(
        self,
        path: Optional[str] = None,
        participant_id: str = "candidate",
        generate_synthetic: bool = False,
        fps: int = _DEFAULT_FPS,
        max_frames: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.path = Path(path) if path else None
        self.participant_id = participant_id
        self.generate_synthetic = generate_synthetic or self.path is None
        self.fps = fps
        self.max_frames = max_frames
        self._container = None
        self._resampler = None
        self._frame_count = 0

    async def start(self) -> None:
        if self.generate_synthetic:
            logger.info("FileSource starting in synthetic mode (no real file).")
            self._running = True
            return

        if self.path is None or not self.path.exists():
            raise FileNotFoundError(f"Media file not found: {self.path}")

        import av  # lazy import: heavy dependency only needed for real files

        self._container = av.open(str(self.path))
        # Prepare an audio resampler so all audio is emitted as 16 kHz mono s16.
        self._resampler = av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=16000,
        )
        self._running = True
        logger.info("FileSource opened %s", self.path)

    async def stop(self) -> None:
        self._running = False
        if self._container is not None:
            self._container.close()
            self._container = None
        self._resampler = None
        logger.info("FileSource stopped.")

    # ------------------------------------------------------------------ #
    # Synthetic frame generation (used for tests / demos without a file).
    # ------------------------------------------------------------------ #
    def _synthetic_frame(self, t_ms: int) -> np.ndarray:
        h = w = 240
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # A white square that moves left->right repeatedly.
        period = 4000  # ms
        phase = (t_ms % period) / period
        x = int((w - 60) * phase)
        frame[90:150, x:x + 60] = (255, 255, 255)
        return frame

    async def _synthetic_stream(self) -> AsyncIterator[RawMediaFrame]:
        frame_idx = 0
        t_ms = 0
        frame_interval = 1000 // self.fps
        while self._running:
            if self.max_frames is not None and frame_idx >= self.max_frames:
                break
            frame = self._synthetic_frame(t_ms)
            # 1 second of silent 16-bit PCM @ 16 kHz.
            audio = np.zeros(16000, dtype=np.int16).tobytes()
            yield RawMediaFrame(
                participant_id=self.participant_id,
                audio_chunk=audio,
                video_frame=frame,
                timestamp_ms=t_ms,
            )
            frame_idx += 1
            t_ms += frame_interval
            # Yield control so the event loop stays responsive.
            await asyncio.sleep(0)

    async def _file_stream(self) -> AsyncIterator[RawMediaFrame]:
        import av  # lazy import

        for packet in self._container.demux():
            if not self._running:
                break
            try:
                for frame in packet.decode():
                    if isinstance(frame, av.VideoFrame):
                        img = frame.to_ndarray(format="rgb24")
                        yield RawMediaFrame(
                            participant_id=self.participant_id,
                            video_frame=img,
                            timestamp_ms=int(frame.time * 1000),
                        )
                    elif isinstance(frame, av.AudioFrame):
                        # Resample to 16 kHz mono s16 for downstream pipelines.
                        resampled_frames = self._resampler.resample(frame)
                        for rf in resampled_frames:
                            pcm = rf.to_ndarray().astype(np.int16).tobytes()
                            yield RawMediaFrame(
                                participant_id=self.participant_id,
                                audio_chunk=pcm,
                                timestamp_ms=int(frame.time * 1000),
                            )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Decode error: %s", exc)
                continue

    def frames(self) -> AsyncIterator[RawMediaFrame]:
        if self.generate_synthetic:
            return self._synthetic_stream()
        return self._file_stream()
