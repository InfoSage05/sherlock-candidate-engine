from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ..llm_client import OpenRouterClient, LLMClient
from ..models import (
    EvidencePacket,
    MeetingContext,
    SignalAxis,
    SignalSource,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)


BATCH_WINDOW_SECONDS = 30
MIN_SEGMENT_LENGTH = 20
MAX_CONFIDENCE = 0.95
MIN_CONFIDENCE = 0.5


def confidence_to_log_odds(confidence: float, role: str) -> float:
    confidence = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, confidence))
    log_odds = math.log(confidence / (1 - confidence))

    if role == "interviewer":
        log_odds = -log_odds

    return log_odds


ROLE_CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "role_guess": {
            "type": "string",
            "enum": ["candidate", "interviewer"],
            "description": "The classified role of the speaker",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence in the classification (0-1)",
        },
        "rationale": {
            "type": "string",
            "description": "One-sentence explanation for the classification",
        },
    },
    "required": ["role_guess", "confidence", "rationale"],
}


class SemanticSignalExtractor:
    def __init__(
        self,
        context: MeetingContext,
        llm_client: Optional[LLMClient] = None,
        enabled: bool = True,
        batch_window_seconds: int = BATCH_WINDOW_SECONDS,
    ):
        self.context = context
        self.llm_client = llm_client
        self.enabled = enabled
        self.batch_window_seconds = batch_window_seconds
        self.transcript_segments: List[TranscriptSegment] = []
        self._processed_until: Optional[datetime] = None
        self._failed_call_count = 0

    def add_transcript_segment(self, segment: TranscriptSegment) -> None:
        self.transcript_segments.append(segment)

    def _build_system_prompt(self) -> str:
        candidate_name = self.context.candidate_name or "unknown"
        interviewer_names = (
            ", ".join(self.context.interviewer_names)
            if self.context.interviewer_names
            else "none provided"
        )

        return f"""You are analyzing a job interview transcript to determine the role of each speaker.

Context:
- Candidate name (from calendar): {candidate_name}
- Known interviewer names: {interviewer_names}
- The candidate's display name may NOT match their calendar name (they could use a nickname, device name, etc.)

Classification guidelines:
- CANDIDATE: Answers questions, explains experience, solves problems, describes projects, responds to technical challenges
- INTERVIEWER: Asks questions, evaluates responses, guides conversation, provides instructions, gives feedback

Important: Base your classification on the CONTENT and FUNCTION of what is said, not just the speaker's name. A person named "Alice" could be the interviewer if she's asking questions and evaluating."""

    def _build_batch_prompt(self, segments: List[TranscriptSegment]) -> str:
        participant = self.context.participants.get(segments[0].participant_id)
        participant_name = participant.display_name if participant else segments[0].participant_id

        transcript_text = "\n\n".join(
            f"[{seg.start_time.strftime('%H:%M:%S')} - {seg.end_time.strftime('%H:%M:%S')}]: {seg.text}"
            for seg in segments
        )

        return f"""Classify the role of the following speaker based on their transcript segments.

Speaker: {participant_name} (ID: {segments[0].participant_id})

Transcript segments:
{transcript_text}

Analyze the content, tone, and function of these segments to determine if this speaker is the CANDIDATE or an INTERVIEWER.

Use the role_classification tool to provide your answer."""

    def _batch_segments_by_window(
        self, segments: List[TranscriptSegment]
    ) -> Dict[str, List[List[TranscriptSegment]]]:
        participant_batches: Dict[str, List[List[TranscriptSegment]]] = defaultdict(list)

        for participant_id in sorted(set(seg.participant_id for seg in segments)):
            participant_segs = sorted(
                [s for s in segments if s.participant_id == participant_id],
                key=lambda s: s.start_time,
            )

            if not participant_segs:
                continue

            current_batch: List[TranscriptSegment] = []
            batch_start_time = participant_segs[0].start_time

            for seg in participant_segs:
                if not seg.text or len(seg.text.strip()) < MIN_SEGMENT_LENGTH:
                    continue

                time_since_batch_start = (seg.start_time - batch_start_time).total_seconds()

                if time_since_batch_start > self.batch_window_seconds and current_batch:
                    participant_batches[participant_id].append(current_batch)
                    current_batch = []
                    batch_start_time = seg.start_time

                current_batch.append(seg)

            if current_batch:
                participant_batches[participant_id].append(current_batch)

        return dict(participant_batches)

    def _classify_batch(
        self, batch: List[TranscriptSegment]
    ) -> Optional[Dict[str, Any]]:
        if not self.llm_client:
            logger.warning("No LLM client configured, skipping classification")
            return None

        if not self.enabled:
            logger.info("LLM signal disabled via config, skipping classification")
            return None

        prompt = self._build_batch_prompt(batch)
        system_prompt = self._build_system_prompt()

        try:
            result = self.llm_client.generate_structured(
                prompt=prompt,
                response_schema=ROLE_CLASSIFICATION_SCHEMA,
                system_prompt=system_prompt,
            )

            if not result:
                logger.warning(f"Empty response from LLM for batch starting at {batch[0].start_time}")
                self._failed_call_count += 1
                return None

            role_guess = result.get("role_guess")
            confidence = result.get("confidence")
            rationale = result.get("rationale", "")

            if role_guess not in ["candidate", "interviewer"]:
                logger.warning(f"Invalid role_guess from LLM: {role_guess}")
                self._failed_call_count += 1
                return None

            if confidence is None or not isinstance(confidence, (int, float)):
                logger.warning(f"Invalid confidence from LLM: {confidence}")
                self._failed_call_count += 1
                return None

            confidence = float(confidence)
            if confidence < MIN_CONFIDENCE:
                logger.info(
                    f"LLM confidence {confidence:.2f} below threshold {MIN_CONFIDENCE}, "
                    f"skipping weak signal"
                )
                return None

            return {
                "role_guess": role_guess,
                "confidence": confidence,
                "rationale": rationale,
            }

        except Exception as e:
            logger.error(
                f"LLM classification failed for batch starting at {batch[0].start_time}: {e}"
            )
            self._failed_call_count += 1
            return None

    def _create_evidence_packet(
        self,
        participant_id: str,
        classification: Dict[str, Any],
        batch: List[TranscriptSegment],
    ) -> EvidencePacket:
        role = classification["role_guess"]
        confidence = classification["confidence"]
        rationale = classification["rationale"]

        delta_log_odds = confidence_to_log_odds(confidence, role)

        segment_texts = [seg.text[:50] for seg in batch[:3]]
        segment_summary = " | ".join(segment_texts)

        return EvidencePacket(
            source=SignalSource.LLM_ROLE_CLASSIFIER,
            axis=SignalAxis.IDENTITY,
            target_participant_id=participant_id,
            delta_log_odds=delta_log_odds,
            confidence=confidence,
            rationale=f"LLM classified as {role}: {rationale}",
            timestamp=datetime.utcnow(),
            metadata={
                "role_guess": role,
                "llm_confidence": confidence,
                "batch_size": len(batch),
                "batch_start": batch[0].start_time.isoformat(),
                "batch_end": batch[-1].end_time.isoformat(),
                "segment_preview": segment_summary,
            },
        )

    def extract_llm_role_classification(self) -> List[EvidencePacket]:
        if not self.enabled:
            logger.info("LLM signal disabled via config")
            return []

        if not self.llm_client:
            logger.warning("No LLM client configured")
            return []

        evidence = []

        batches = self._batch_segments_by_window(self.transcript_segments)

        for participant_id, participant_batches in batches.items():
            for batch in participant_batches:
                if not batch:
                    continue

                classification = self._classify_batch(batch)

                if classification is None:
                    continue

                packet = self._create_evidence_packet(participant_id, classification, batch)
                evidence.append(packet)

        if evidence:
            self._processed_until = max(
                seg.end_time for seg in self.transcript_segments
            )

        return evidence

    def extract_all(self) -> List[EvidencePacket]:
        return self.extract_llm_role_classification()

    def get_failure_stats(self) -> Dict[str, int]:
        return {
            "failed_calls": self._failed_call_count,
            "total_segments": len(self.transcript_segments),
        }
