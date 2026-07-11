"""Ingestion layer for Sherlock.

Provides a platform-agnostic interface for receiving live, per-participant
audio and video streams. Concrete sources:

- ``FileSource``   : replays a recorded ``.mp4``/``.wav`` file (or generates
                     synthetic frames) for CI / local development.
- ``WebRTCSource`` : connects to a live meeting peer via ``aiortc``.

Every source exposes the same async contract so the rest of the pipeline
never cares where frames come from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from ..models import RawMediaFrame


class MediaSource(ABC):
    """Abstract base class for all meeting-platform ingestion adapters."""

    def __init__(self) -> None:
        self._running = False
        self.participant_id: Optional[str] = None

    @abstractmethod
    async def start(self) -> None:
        """Open the connection / file and begin producing frames."""
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        """Close the connection / file and stop producing frames."""
        raise NotImplementedError

    @abstractmethod
    def frames(self) -> AsyncIterator[RawMediaFrame]:
        """Yield ``RawMediaFrame`` objects for every participant."""
        raise NotImplementedError

    @property
    def running(self) -> bool:
        return self._running
