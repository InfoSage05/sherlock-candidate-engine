"""Text-level authenticity pipeline.

Analyses transcript segments to detect signatures of:
1. AI-generated / LLM-drafted text being read aloud.
2. Reading from a hidden screen / phone (reading pattern).
3. Unnatural pause placement combined with polished output.

The pipeline emits EvidencePacket objects on the AUTHENTICITY axis.  It is
intentionally conservative: no single signal is treated as a verdict; all
signals are fused as weak log-odds updates in the FusionEngine.
"""

from __future__ import annotations

import logging
import math
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

from ..models import EvidencePacket, FlagSeverity, MeetingContext, SignalAxis, SignalSource, TranscriptSegment

logger = logging.getLogger(__name__)

# Common AI/LLM filler phrases and overly formal transitions.
AI_FORMAL_TRIGGERS = {
    "it is important to note", "it is worth noting", "it should be noted",
    "in conclusion", "furthermore", "moreover", "consequently", "therefore",
    "thus", "hence", "nonetheless", "nevertheless", "in summary", "to summarize",
    "overall", "as a result", "for instance", "for example", "additionally",
    "on the other hand", "in order to", "due to the fact that", "in the event that",
    "with respect to", "with regard to", "it is evident that", "it is clear that",
    "one could argue that", "it can be argued that", "the aforementioned",
}

# Transitions that often appear when a speaker is reading bullets.
READING_TRANSITIONS = {
    "first", "second", "third", "fourth", "fifth", "firstly", "secondly",
    "thirdly", "next", "then", "finally", "lastly", "moving on", "going back",
    "as mentioned", "as previously stated", "referring back",
}

FILLER_WORDS = {
    "um", "uh", "er", "ah", "like", "you know", "i mean", "sort of", "kind of",
}


@dataclass
class SegmentAuthenticity:
    """Result of analysing a single transcript segment."""
    segment: TranscriptSegment
    ai_score: float = 0.0
    reading_score: float = 0.0
    pause_score: float = 0.0
    rationale: str = ""


class TextAuthenticityPipeline:
    """Detect AI-generated or read-aloud transcript segments.

    Parameters
    ----------
    context : MeetingContext
    llm_client : optional LLM client for an additional weak signal.
    window_size : int
        Number of recent segments to keep for baseline computation.
    """

    def __init__(
        self,
        context: Optional[MeetingContext] = None,
        llm_client=None,
        window_size: int = 20,
    ) -> None:
        self.context = context
        self.llm_client = llm_client
        self._segments: Deque[TranscriptSegment] = deque(maxlen=window_size)
        self._participant_segments: Dict[str, Deque[TranscriptSegment]] = {}
        self._last_segment_end: Dict[str, datetime] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def process(self, segment: TranscriptSegment) -> List[EvidencePacket]:
        """Analyse a new transcript segment and return evidence packets."""
        self._segments.append(segment)
        pid = segment.participant_id
        if pid not in self._participant_segments:
            self._participant_segments[pid] = deque(maxlen=self._segments.maxlen)
        self._participant_segments[pid].append(segment)

        packets: List[EvidencePacket] = []
        packets.extend(self._detect_ai_generated_text(segment, pid))
        packets.extend(self._detect_reading_pattern(segment, pid))
        packets.extend(self._detect_unnatural_pause(segment, pid))

        # Optional LLM-based weak signal (not a verdict).
        llm_packet = self._llm_ai_signal(segment, pid)
        if llm_packet:
            packets.append(llm_packet)

        self._last_segment_end[pid] = segment.end_time
        return packets

    # ------------------------------------------------------------------ #
    # Heuristic: AI-generated text
    # ------------------------------------------------------------------ #
    def _detect_ai_generated_text(
        self, segment: TranscriptSegment, participant_id: str
    ) -> List[EvidencePacket]:
        text = segment.text.strip()
        if not text:
            return []

        lower = text.lower()
        words = re.sub(r"[^\w\s']", " ", lower).split()
        if not words:
            return []

        # 1. Overly formal / AI transition density.
        formal_hits = sum(1 for phrase in AI_FORMAL_TRIGGERS if phrase in lower)
        formal_density = formal_hits / max(len(words) / 30.0, 1.0)

        # 2. Uniform sentence length (low coefficient of variation).
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        sent_cv = self._coefficient_of_variation([len(s.split()) for s in sentences])

        # 3. Low first-person / personal anecdote density.
        first_person = sum(1 for w in words if w in {"i", "my", "me", "myself"})
        personal_density = first_person / len(words)

        # 4. Absence of disfluencies (reading polished text).
        disfluency_rate = self._disfluency_rate(text)

        # Composite heuristic score in [0, 1].
        score = min(1.0, max(0.0,
            formal_density * 0.35
            + (1.0 - min(1.0, sent_cv / 0.5)) * 0.20
            + (1.0 - min(1.0, personal_density / 0.08)) * 0.20
            + (1.0 - min(1.0, disfluency_rate / 0.03)) * 0.25
        ))

        if score < 0.45:
            return []

        confidence = min(0.95, score)
        delta = -(score - 0.4) * 2.0
        severity = FlagSeverity.CRITICAL if score > 0.8 else FlagSeverity.WARNING

        reasons = []
        if formal_density > 0.3:
            reasons.append("overly formal / AI-like transitions")
        if sent_cv < 0.35 and len(sentences) > 2:
            reasons.append("uniform sentence length")
        if personal_density < 0.03 and len(words) > 20:
            reasons.append("low personal language")
        if disfluency_rate < 0.005 and len(words) > 20:
            reasons.append("unnaturally polished delivery")

        rationale = (
            f"Segment shows AI-generated / read-aloud signatures "
            f"({', '.join(reasons) or 'weak combined cues'}). "
            f"Score={score:.2f}"
        )

        return [EvidencePacket(
            source=SignalSource.AI_GENERATED_TEXT,
            axis=SignalAxis.AUTHENTICITY,
            target_participant_id=participant_id,
            delta_log_odds=delta,
            confidence=confidence,
            severity=severity,
            flag_type="ai_generated_text",
            recommendation="Verify the candidate is answering spontaneously, not reading generated text.",
            rationale=rationale,
            timestamp=segment.start_time,
            metadata={
                "ai_score": round(score, 4),
                "formal_density": round(formal_density, 4),
                "sentence_cv": round(sent_cv, 4),
                "personal_density": round(personal_density, 4),
                "disfluency_rate": round(disfluency_rate, 4),
                "segment_text": text[:200],
            },
        )]

    # ------------------------------------------------------------------ #
    # Heuristic: reading pattern (read from screen / phone)
    # ------------------------------------------------------------------ #
    def _detect_reading_pattern(
        self, segment: TranscriptSegment, participant_id: str
    ) -> List[EvidencePacket]:
        text = segment.text.strip()
        if not text:
            return []

        lower = text.lower()
        words = re.sub(r"[^\w\s']", " ", lower).split()
        if len(words) < 5:
            return []

        # Reading cues: enumerated transitions, repeated exact phrases, low filler rate.
        reading_transitions = sum(1 for w in words if w in READING_TRANSITIONS)
        transition_density = reading_transitions / len(words)

        # Repeated phrase within segment (copy-pasted bullets read aloud).
        repeated_score = self._repeated_phrase_score(text)

        # Polished + enumerations strongly suggests reading.
        disfluency_rate = self._disfluency_rate(text)

        score = min(1.0, max(0.0,
            transition_density * 2.5 * 0.35
            + repeated_score * 0.35
            + (1.0 - min(1.0, disfluency_rate / 0.03)) * 0.30
        ))

        if score < 0.45:
            return []

        confidence = min(0.9, score)
        delta = -(score - 0.4) * 1.8
        severity = FlagSeverity.WARNING

        reasons = []
        if transition_density > 0.08:
            reasons.append("enumerated transitions")
        if repeated_score > 0.3:
            reasons.append("repeated phrasing")
        if disfluency_rate < 0.005 and len(words) > 15:
            reasons.append("low disfluency while reading-like")

        rationale = (
            f"Delivery pattern suggests reading from another screen "
            f"({', '.join(reasons)}). Score={score:.2f}"
        )

        return [EvidencePacket(
            source=SignalSource.READING_PATTERN,
            axis=SignalAxis.AUTHENTICITY,
            target_participant_id=participant_id,
            delta_log_odds=delta,
            confidence=confidence,
            severity=severity,
            flag_type="reading_pattern",
            recommendation="Check if the candidate is looking at a phone/second screen while speaking.",
            rationale=rationale,
            timestamp=segment.start_time,
            metadata={
                "reading_score": round(score, 4),
                "transition_density": round(transition_density, 4),
                "repeated_score": round(repeated_score, 4),
                "disfluency_rate": round(disfluency_rate, 4),
                "segment_text": text[:200],
            },
        )]

    # ------------------------------------------------------------------ #
    # Heuristic: unnatural pause placement
    # ------------------------------------------------------------------ #
    def _detect_unnatural_pause(
        self, segment: TranscriptSegment, participant_id: str
    ) -> List[EvidencePacket]:
        """Flag a long pause right before a polished answer.

        The segment itself does not carry pause information; we infer the pause
        from the gap between this segment's start and the previous segment's end
        for the same speaker.
        """
        last_end = self._last_segment_end.get(participant_id)
        if last_end is None:
            return []

        pause_seconds = (segment.start_time - last_end).total_seconds()
        if pause_seconds <= 0:
            return []

        text = segment.text.strip()
        fluency = self._structural_fluency(text)

        # The meaningful pattern: long pause (>4s) followed by highly fluent text.
        if pause_seconds < 4.0 or fluency < 0.75:
            return []

        # Scale severity by pause length and fluency.
        score = min(1.0, (pause_seconds / 15.0) * fluency)
        delta = -(score - 0.3) * 2.0
        confidence = min(0.9, score)

        return [EvidencePacket(
            source=SignalSource.UNNATURAL_PAUSE,
            axis=SignalAxis.AUTHENTICITY,
            target_participant_id=participant_id,
            delta_log_odds=delta,
            confidence=confidence,
            severity=FlagSeverity.WARNING,
            flag_type="unnatural_pause",
            recommendation="Long pause before a polished answer may indicate the candidate looked up a response.",
            rationale=(
                f"Pause of {pause_seconds:.1f}s before a structurally fluent answer "
                f"(fluency={fluency:.2f}) — decreased authenticity"
            ),
            timestamp=segment.start_time,
            metadata={
                "pause_seconds": round(pause_seconds, 2),
                "answer_fluency": round(fluency, 4),
                "segment_text": text[:200],
            },
        )]

    # ------------------------------------------------------------------ #
    # Optional LLM weak signal
    # ------------------------------------------------------------------ #
    def _llm_ai_signal(
        self, segment: TranscriptSegment, participant_id: str
    ) -> Optional[EvidencePacket]:
        if self.llm_client is None:
            return None

        text = segment.text.strip()
        if len(text.split()) < 10:
            return None

        system_prompt = (
            "You are a weak sensor in a larger fusion engine. "
            "Estimate whether the following spoken answer sounds like it was read from "
            "AI-generated text or a hidden screen. Return ONLY a JSON object with keys: "
            "ai_likelihood (float 0-1), reading_likelihood (float 0-1), one_line_reason. "
            "Be conservative; low confidence should map to low scores."
        )
        prompt = f"Spoken answer:\n{text[:500]}\n\nProvide the JSON object."

        try:
            raw = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=256,
            )
            # Simple JSON extraction.
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if not match:
                return None
            import json
            result = json.loads(match.group(0))
            ai_score = float(result.get("ai_likelihood", 0.0))
            reading_score = float(result.get("reading_likelihood", 0.0))
        except Exception as exc:
            logger.debug("LLM text authenticity signal skipped: %s", exc)
            return None

        combined = max(ai_score, reading_score)
        if combined < 0.5:
            return None

        delta = -(combined - 0.4) * 1.5
        return EvidencePacket(
            source=SignalSource.AI_GENERATED_TEXT,
            axis=SignalAxis.AUTHENTICITY,
            target_participant_id=participant_id,
            delta_log_odds=delta,
            confidence=min(0.85, combined),
            severity=FlagSeverity.INFO,
            flag_type="llm_ai_text_signal",
            recommendation="LLM sensor flagged this segment; corroborate with other signals.",
            rationale=f"LLM sensor: {result.get('one_line_reason', 'AI/read pattern suspected')}",
            timestamp=segment.start_time,
            metadata={
                "ai_likelihood": ai_score,
                "reading_likelihood": reading_score,
                "segment_text": text[:200],
            },
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _disfluency_rate(text: str) -> float:
        lower = text.lower()
        words = re.sub(r"[^\w\s']", " ", lower).split()
        if not words:
            return 0.0
        fillers = sum(1 for w in words if w in FILLER_WORDS)
        # Also count "you know" bigram once.
        fillers += lower.count("you know")
        return fillers / len(words)

    @staticmethod
    def _structural_fluency(text: str) -> float:
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) < 2:
            return 0.5
        lengths = [len(s.split()) for s in sentences]
        cv = TextAuthenticityPipeline._coefficient_of_variation(lengths)
        return 1.0 / (1.0 + cv)

    @staticmethod
    def _coefficient_of_variation(values: List[int]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        if mean == 0:
            return 0.0
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = math.sqrt(variance)
        return std / mean

    @staticmethod
    def _repeated_phrase_score(text: str) -> float:
        """Detect repeated 4-grams (common when reading bullet points)."""
        words = re.sub(r"[^\w\s']", " ", text.lower()).split()
        if len(words) < 8:
            return 0.0
        ngrams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
        if not ngrams:
            return 0.0
        unique = set(ngrams)
        repeated = len(ngrams) - len(unique)
        return min(1.0, repeated / max(len(ngrams) * 0.3, 1.0))
