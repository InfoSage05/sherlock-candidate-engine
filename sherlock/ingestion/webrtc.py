"""WebRTC ingestion source.

Connects to a live meeting peer using ``aiortc`` and demuxes each remote
participant's audio/video tracks into ``RawMediaFrame`` objects. ``aiortc`` is
an optional dependency; the class is importable without it, but ``start()``
will raise a clear error if the library is missing.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Dict, Optional

from ..models import RawMediaFrame
from .base import MediaSource

logger = logging.getLogger(__name__)


class _TrackReceiver:
    """Receives frames from a single aiortc MediaStreamTrack."""

    def __init__(self, participant_id: str, kind: str) -> None:
        self.participant_id = participant_id
        self.kind = kind  # "audio" | "video"
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=30)
        self._closed = False

    async def on_frame(self, frame) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            # Drop oldest to make room - late frames are worthless.
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait(frame)

    async def get(self):
        return await self._queue.get()


class WebRTCSource(MediaSource):
    def __init__(self, signaling_url: str, room: str, display_name: str = "sherlock") -> None:
        super().__init__()
        self.signaling_url = signaling_url
        self.room = room
        self.display_name = display_name
        self._pc = None
        self._receivers: Dict[str, _TrackReceiver] = {}

    async def start(self) -> None:
        try:
            import aiortc  # lazy import: optional dependency
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "aiortc is required for WebRTCSource. Install with `pip install aiortc`."
            ) from exc

        from aiortc import RTCPeerConnection

        self._pc = RTCPeerConnection()
        self._pc.on("track", self._on_track)
        # A real implementation would perform signaling handshake here
        # (offer/answer exchange via self.signaling_url). Left as the
        # integration point for the chosen meeting platform.
        self._running = True
        logger.info("WebRTCSource connected to room %s", self.room)

    def _on_track(self, track):
        # Map the track to a participant id (would come from signaling metadata).
        participant_id = getattr(track, "member_id", track.id)
        receiver = _TrackReceiver(participant_id, track.kind)
        self._receivers[participant_id] = receiver
        track.on("frame", receiver.on_frame)

    async def stop(self) -> None:
        self._running = False
        if self._pc is not None:
            await self._pc.close()
            self._pc = None
        logger.info("WebRTCSource stopped.")

    async def _stream(self) -> AsyncIterator[RawMediaFrame]:
        while self._running:
            for receiver in list(self._receivers.values()):
                try:
                    frame = await asyncio.wait_for(receiver.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                if receiver.kind == "video":
                    img = frame.to_ndarray(format="rgb24")
                    yield RawMediaFrame(
                        participant_id=receiver.participant_id,
                        video_frame=img,
                        timestamp_ms=int(frame.time * 1000),
                    )
                else:
                    pcm = frame.to_ndarray().astype("int16").tobytes()
                    yield RawMediaFrame(
                        participant_id=receiver.participant_id,
                        audio_chunk=pcm,
                        timestamp_ms=int(frame.time * 1000),
                    )

    def frames(self) -> AsyncIterator[RawMediaFrame]:
        return self._stream()
