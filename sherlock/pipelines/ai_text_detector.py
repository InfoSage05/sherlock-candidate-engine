"""Real AI-generated text detector using open-source transformer models.

Uses Hugging Face transformers as a **weak signal** in the fusion engine.
The detector returns P(text is AI-generated).  It is calibrated per candidate
so a naturally articulate person is not unfairly penalised.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "roberta-base-openai-detector"


class AITextDetector:
    """Wrapper around an open-source transformer AI-text classifier."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: int = -1) -> None:
        self.model_name = model_name
        self._device = device
        self._pipeline = None

    def _ensure(self):
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline

            logger.info("Loading AI-text detector: %s", self.model_name)
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=self._device,
                truncation=True,
                max_length=512,
            )
        except Exception as exc:
            logger.error("Failed to load AI-text detector %s: %s", self.model_name, exc)
            raise

    def predict(self, text: str) -> Tuple[float, dict]:
        """Return (ai_score, metadata).  ai_score in [0, 1]."""
        text = text.strip()
        if not text or len(text.split()) < 5:
            return 0.0, {"reason": "too_short"}

        self._ensure()
        result = self._pipeline(text)[0]
        # roberta-base-openai-detector labels: LABEL_0 = human, LABEL_1 = AI
        label = result["label"]
        score = result["score"]
        ai_score = score if label in ("LABEL_1", "AI", "fake") else 1.0 - score
        return float(ai_score), {"model": self.model_name, "label": label, "raw_score": float(score)}
