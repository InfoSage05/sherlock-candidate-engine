"""Per-candidate speaker-embedding store.

Holds a rolling baseline of a candidate's own voice embeddings so the
voice-liveness pipeline can detect drift *within* a single session (a key cue
for swapped/synthesized speakers). Pure-numpy, no external dependencies.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SpeakerStore:
    def __init__(self, max_samples: int = 20) -> None:
        self._baselines: Dict[str, List[np.ndarray]] = {}
        self._max_samples = max_samples

    def update(self, participant_id: str, embedding: np.ndarray) -> None:
        emb = np.asarray(embedding, dtype=np.float32)
        bucket = self._baselines.setdefault(participant_id, [])
        bucket.append(emb)
        if len(bucket) > self._max_samples:
            bucket.pop(0)

    def has_baseline(self, participant_id: str) -> bool:
        return participant_id in self._baselines and bool(self._baselines[participant_id])

    def baseline_vector(self, participant_id: str) -> Optional[np.ndarray]:
        bucket = self._baselines.get(participant_id)
        if not bucket:
            return None
        return np.mean(np.stack(bucket), axis=0)

    def drift(self, participant_id: str, embedding: np.ndarray) -> float:
        """Cosine distance between embedding and the stored baseline."""
        base = self.baseline_vector(participant_id)
        if base is None:
            return 0.0
        emb = np.asarray(embedding, dtype=np.float32)
        if emb.shape != base.shape:
            # Different dimensionality -> treat as maximal drift.
            return 1.0
        denom = (np.linalg.norm(emb) * np.linalg.norm(base))
        if denom == 0:
            return 0.0
        cos = float(np.dot(emb, base) / denom)
        return float(1.0 - max(-1.0, min(1.0, cos)))
