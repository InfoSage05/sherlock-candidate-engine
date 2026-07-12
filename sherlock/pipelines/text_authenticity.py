"""Text-level authenticity pipeline (advanced version).

Analyses candidate answers to detect:
1. AI-generated / LLM-drafted text (open-source transformer detector).
2. Generic / scripted / non-responsive answers.
3. Unnatural pause + polished-answer pattern.

Heavy transformer inference is run in a background thread pool so it does not
block the real-time audio/video ingestion loop.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, List, Optional

from ..models import EvidencePacket, FlagSeverity, SignalAxis, SignalSource, TranscriptSegment
from .ai_text_detector import AITextDetector

logger = logging.getLogger(__name__)

# Phrases that are normal in interviews and should NOT trigger AI flags.
INTERVIEW_COMMON_PHRASES = {
    "thank you", "thanks", "nice to meet you", "good luck",
    "do you have any questions", "do you have any other questions",
    "i don't think so", "no, i don't think so", "not that i can think of",
    "that's a good question", "that's an interesting question",
    "i'm not sure", "i don't know", "could you repeat",
    "yes, exactly", "absolutely", "definitely", "of course",
    "i would say", "i think", "i believe", "in my opinion",
}

# Generic filler phrases that indicate a memorised / non-specific answer.
GENERIC_ANSWER_MARKERS = {
    "hardworking", "team player", "passionate", "detail-oriented",
    "fast learner", "good communicator", "problem solver", "self-starter",
    "go above and beyond", "think outside the box", "synergy", "leverage",
    "proven track record", "results-driven", "dynamic environment",
}


@dataclass
class SegmentAnalysis:
    segment: TranscriptSegment
    ai_score: float = 0.0
    generic_score: float = 0.0
    pause_score: float = 0.0
    relevance_score: float = 1.0
    is_answer: bool = False
    rationale: str = ""


class TextAuthenticityPipeline:
    """Advanced text authenticity pipeline."""

    def __init__(
        self,
        context: Optional[object] = None,
        ai_detector: Optional[AITextDetector] = None,
        semantic_model=None,
        window_size: int = 30,
        max_workers: int = 1,
    ) -> None:
        self.context = context
        self.ai_detector = ai_detector or AITextDetector()
        self.semantic_model = semantic_model
        self._segments: Deque[TranscriptSegment] = deque(maxlen=window_size)
        self._candidate_ai_scores: List[float] = []
        self._last_segment_end: Dict[str, datetime] = {}
        self._last_question: Optional[TranscriptSegment] = None
        self._model_load_error: bool = False
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="txt_auth")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def aprocess(self, segment: TranscriptSegment) -> List[EvidencePacket]:
        """Async entry point.  Heavy inference runs in a thread pool."""
        self._segments.append(segment)

        if segment.is_question:
            self._last_question = segment
            return []

        analysis = await asyncio.get_event_loop().run_in_executor(
            self._executor, self._analyse_answer, segment
        )
        if not analysis.is_answer:
            return []

        packets: List[EvidencePacket] = []
        ai_packet = self._build_ai_packet(analysis)
        if ai_packet:
            packets.append(ai_packet)

        generic_packet = self._build_generic_packet(analysis)
        if generic_packet:
            packets.append(generic_packet)

        pause_packet = self._build_pause_packet(analysis, segment)
        if pause_packet:
            packets.append(pause_packet)

        self._last_segment_end[segment.participant_id] = segment.end_time
        return packets

    def process(self, segment: TranscriptSegment) -> List[EvidencePacket]:
        """Synchronous fallback (runs inference on calling thread)."""
        self._segments.append(segment)
        if segment.is_question:
            self._last_question = segment
            return []

        analysis = self._analyse_answer(segment)
        if not analysis.is_answer:
            return []

        packets: List[EvidencePacket] = []
        ai_packet = self._build_ai_packet(analysis)
        if ai_packet:
            packets.append(ai_packet)
        generic_packet = self._build_generic_packet(analysis)
        if generic_packet:
            packets.append(generic_packet)
        pause_packet = self._build_pause_packet(analysis, segment)
        if pause_packet:
            packets.append(pause_packet)

        self._last_segment_end[segment.participant_id] = segment.end_time
        return packets

    # ------------------------------------------------------------------ #
    # Analysis helpers
    # ------------------------------------------------------------------ #
    def _analyse_answer(self, segment: TranscriptSegment) -> SegmentAnalysis:
        text = segment.text.strip()
        lower = text.lower()
        words = re.sub(r"[^\w\s']", " ", lower).split()

        if len(words) < 6:
            return SegmentAnalysis(segment=segment, is_answer=False)

        analysis = SegmentAnalysis(segment=segment, is_answer=True)

        try:
            if not self._model_load_error:
                ai_score, _ = self.ai_detector.predict(text)
                analysis.ai_score = ai_score
        except Exception as exc:
            logger.warning("AI-text detector failed: %s", exc)
            self._model_load_error = True

        generic_hits = sum(1 for marker in GENERIC_ANSWER_MARKERS if marker in lower)
        analysis.generic_score = min(1.0, generic_hits / 3.0)

        if self._last_question and self.semantic_model is not None:
            try:
                analysis.relevance_score = self._semantic_similarity(
                    self._last_question.text, text
                )
            except Exception as exc:
                logger.debug("Semantic similarity failed: %s", exc)

        return analysis

    # ------------------------------------------------------------------ #
    # Signal builders
    # ------------------------------------------------------------------ #
    def _build_ai_packet(self, analysis: SegmentAnalysis) -> Optional[EvidencePacket]:
        text = analysis.segment.text.strip()
        lower = text.lower()

        if any(phrase in lower for phrase in INTERVIEW_COMMON_PHRASES):
            return None

        ai_score = analysis.ai_score
        if ai_score < 0.75:
            return None

        baseline = self._candidate_baseline_ai()
        if baseline > 0.5 and ai_score < 0.85:
            return None

        combined = ai_score
        if analysis.generic_score > 0.3:
            combined = min(1.0, combined + 0.1)
        if analysis.relevance_score < 0.25:
            combined = min(1.0, combined + 0.15)

        self._candidate_ai_scores.append(ai_score)

        severity = FlagSeverity.CRITICAL if combined > 0.9 else FlagSeverity.WARNING
        delta = -(combined - 0.6) * 2.5

        return EvidencePacket(
            source=SignalSource.AI_GENERATED_TEXT,
            axis=SignalAxis.AUTHENTICITY,
            target_participant_id=analysis.segment.participant_id,
            delta_log_odds=delta,
            confidence=min(0.95, combined),
            severity=severity,
            flag_type="ai_generated_text",
            recommendation="This answer has strong AI-generated text signatures. Verify spontaneity.",
            rationale=(
                f"Transformer detector score {ai_score:.2f} "
                f"(baseline {baseline:.2f}); answer shows AI-generated signatures."
            ),
            timestamp=analysis.segment.start_time,
            metadata={
                "ai_score": round(ai_score, 4),
                "baseline_ai": round(baseline, 4),
                "generic_score": round(analysis.generic_score, 4),
                "relevance_score": round(analysis.relevance_score, 4),
                "segment_text": text[:200],
                "model": getattr(self.ai_detector, "model_name", "unknown"),
            },
        )

    def _build_generic_packet(self, analysis: SegmentAnalysis) -> Optional[EvidencePacket]:
        if analysis.generic_score < 0.4:
            return None

        text = analysis.segment.text
        has_specifics = bool(re.search(r"\b\d+\b", text)) or "for example" in text.lower() or "instance" in text.lower()
        if has_specifics:
            return None

        return EvidencePacket(
            source=SignalSource.READING_PATTERN,
            axis=SignalAxis.AUTHENTICITY,
            target_participant_id=analysis.segment.participant_id,
            delta_log_odds=-0.8,
            confidence=min(0.75, analysis.generic_score),
            severity=FlagSeverity.INFO,
            flag_type="generic_scripted_answer",
            recommendation="Answer is generic and lacks concrete examples; may be rehearsed.",
            rationale="Answer relies heavily on generic interview buzzwords without specifics.",
            timestamp=analysis.segment.start_time,
            metadata={
                "generic_score": round(analysis.generic_score, 4),
                "segment_text": text[:200],
            },
        )

    def _build_pause_packet(self, analysis: SegmentAnalysis, segment: TranscriptSegment) -> Optional[EvidencePacket]:
        last_end = self._last_segment_end.get(segment.participant_id)
        if last_end is None:
            return None

        pause_seconds = (segment.start_time - last_end).total_seconds()
        if pause_seconds < 4.0:
            return None

        fluency = self._structural_fluency(segment.text)
        if fluency < 0.7:
            return None

        if analysis.ai_score < 0.6 and pause_seconds < 10.0:
            return None

        score = min(1.0, (pause_seconds / 15.0) * fluency)
        return EvidencePacket(
            source=SignalSource.UNNATURAL_PAUSE,
            axis=SignalAxis.AUTHENTICITY,
            target_participant_id=segment.participant_id,
            delta_log_odds=-(score - 0.3) * 2.0,
            confidence=min(0.85, score),
            severity=FlagSeverity.WARNING,
            flag_type="unnatural_pause",
            recommendation="Long pause before a polished answer may indicate looking up a response.",
            rationale=(
                f"Pause of {pause_seconds:.1f}s before a structurally fluent answer "
                f"(fluency={fluency:.2f})."
            ),
            timestamp=segment.start_time,
            metadata={
                "pause_seconds": round(pause_seconds, 2),
                "answer_fluency": round(fluency, 4),
                "ai_score": round(analysis.ai_score, 4),
                "segment_text": segment.text[:200],
            },
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _candidate_baseline_ai(self) -> float:
        if not self._candidate_ai_scores:
            return 0.0
        sorted_scores = sorted(self._candidate_ai_scores[-10:])
        return sorted_scores[len(sorted_scores) // 2]

    @staticmethod
    def _structural_fluency(text: str) -> float:
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) < 2:
            return 0.5
        lengths = [len(s.split()) for s in sentences]
        mean = sum(lengths) / len(lengths)
        if mean == 0:
            return 0.0
        variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
        std = math.sqrt(variance)
        cv = std / mean
        return 1.0 / (1.0 + cv)

    def _semantic_similarity(self, question: str, answer: str) -> float:
        if self.semantic_model is None:
            return 1.0
        embeddings = self.semantic_model.encode([question, answer])
        a, b = embeddings[0], embeddings[1]
        norm = (sum(a * a) ** 0.5) * (sum(b * b) ** 0.5)
        if norm == 0:
            return 0.0
        return float(sum(a * b) / norm)
