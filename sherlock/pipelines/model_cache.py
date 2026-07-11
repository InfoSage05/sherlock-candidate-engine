"""Simple in-process model cache.

Loads and memoizes heavy model objects so they are instantiated once. Real
backends (OpenCV/MediaPipe/InsightFace/torch models) are loaded lazily by the
caller-supplied ``loader`` function, so importing this module never pulls in
optional dependencies.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class ModelCache:
    def __init__(self) -> None:
        self._cache: Dict[str, object] = {}

    def load(self, name: str, loader: Callable[[], object]) -> object:
        if name not in self._cache:
            logger.info("Loading model '%s'...", name)
            self._cache[name] = loader()
        return self._cache[name]

    def get(self, name: str) -> Optional[object]:
        return self._cache.get(name)

    def clear(self) -> None:
        self._cache.clear()


# Shared singleton used across pipelines.
default_cache = ModelCache()
