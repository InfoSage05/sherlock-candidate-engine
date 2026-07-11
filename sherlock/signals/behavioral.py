from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from ..models import (
    EvidencePacket,
    MeetingContext,
    SignalAxis,
    SignalSource,
    SpeakingEvent,
    TranscriptSegment,
)


class BehavioralSignalExtractor:
    def __init__(self, context: MeetingContext):
        self.context = context
        self.speaking_events: List[SpeakingEvent] = []
        self.transcript_segments: List[TranscriptSegment] = []

    def add_speaking_event(self, event: SpeakingEvent) -> None:
        self.speaking_events.append(event)

    def add_transcript_segment(self, segment: TranscriptSegment) -> None:
        self.transcript_segments.append(segment)

    def _is_known_interviewer(self, participant) -> bool:
        """Check if a participant matches any known interviewer name or email."""
        name_lower = participant.display_name.lower() if participant.display_name else ""
        email_lower = participant.email.lower() if participant.email else ""
        for iname in self.context.interviewer_names:
            if iname.lower() in name_lower or name_lower in iname.lower():
                return True
        for iemail in self.context.interviewer_emails:
            if iemail.lower() == email_lower:
                return True
        return False

    def extract_turn_taking(self) -> List[EvidencePacket]:
        evidence = []
        now = datetime.utcnow()

        if not self.transcript_segments:
            return evidence

        in_degree: Dict[str, int] = {pid: 0 for pid in self.context.participants}
        out_degree: Dict[str, int] = {pid: 0 for pid in self.context.participants}

        questions = [s for s in self.transcript_segments if s.is_question]

        for question in questions:
            for segment in self.transcript_segments:
                if segment.participant_id == question.participant_id:
                    continue
                if segment.start_time > question.end_time:
                    time_gap = (segment.start_time - question.end_time).total_seconds()
                    if time_gap < 30:
                        in_degree[segment.participant_id] = in_degree.get(
                            segment.participant_id, 0
                        ) + 1
                        out_degree[question.participant_id] = out_degree.get(
                            question.participant_id, 0
                        ) + 1
                        break

        max_in_degree = max(in_degree.values()) if in_degree else 1
        if max_in_degree == 0:
            max_in_degree = 1

        for pid, count in in_degree.items():
            if count == 0:
                continue

            normalized_score = count / max_in_degree
            delta = normalized_score * 2.0

            evidence.append(EvidencePacket(
                source=SignalSource.TURN_TAKING,
                axis=SignalAxis.IDENTITY,
                target_participant_id=pid,
                delta_log_odds=delta,
                confidence=min(normalized_score, 1.0),
                rationale=(
                    f"Received {count} responses to questions "
                    f"(in-degree: {count}, out-degree: {out_degree.get(pid, 0)})"
                ),
                timestamp=now,
                metadata={
                    "in_degree": count,
                    "out_degree": out_degree.get(pid, 0),
                },
            ))

        return evidence

    def extract_speaking_ratio(self) -> List[EvidencePacket]:
        evidence = []
        now = datetime.utcnow()

        if not self.speaking_events:
            return evidence

        total_duration: Dict[str, float] = {pid: 0.0 for pid in self.context.participants}
        reactive_duration: Dict[str, float] = {pid: 0.0 for pid in self.context.participants}

        for event in self.speaking_events:
            if event.participant_id in total_duration:
                total_duration[event.participant_id] += event.duration_seconds
                if event.is_response_to_question:
                    reactive_duration[event.participant_id] += event.duration_seconds

        grand_total = sum(total_duration.values())
        if grand_total == 0:
            return evidence

        for pid, duration in total_duration.items():
            if duration == 0:
                continue

            speaking_ratio = duration / grand_total
            reactive_ratio = (
                reactive_duration[pid] / duration if duration > 0 else 0.0
            )

            combined_score = speaking_ratio * 0.6 + reactive_ratio * 0.4
            delta = combined_score * 1.5

            evidence.append(EvidencePacket(
                source=SignalSource.SPEAKING_RATIO,
                axis=SignalAxis.IDENTITY,
                target_participant_id=pid,
                delta_log_odds=delta,
                confidence=min(combined_score, 1.0),
                rationale=(
                    f"Speaking ratio: {speaking_ratio:.2f} of total time, "
                    f"reactive ratio: {reactive_ratio:.2f} of own speech"
                ),
                timestamp=now,
                metadata={
                    "speaking_ratio": speaking_ratio,
                    "reactive_ratio": reactive_ratio,
                    "total_duration_seconds": duration,
                },
            ))

        return evidence

    def extract_screen_share(self) -> List[EvidencePacket]:
        evidence = []
        now = datetime.utcnow()

        for pid, participant in self.context.participants.items():
            if participant.is_screen_sharing and not self._is_known_interviewer(participant):
                evidence.append(EvidencePacket(
                    source=SignalSource.SCREEN_SHARE,
                    axis=SignalAxis.IDENTITY,
                    target_participant_id=pid,
                    delta_log_odds=1.5,
                    confidence=0.7,
                    rationale=f"Currently sharing screen (participant: {participant.display_name})",
                    timestamp=now,
                ))

        return evidence

    def extract_webcam_state(self) -> List[EvidencePacket]:
        evidence = []
        now = datetime.utcnow()

        for pid, participant in self.context.participants.items():
            if participant.webcam_on and not self._is_known_interviewer(participant):
                evidence.append(EvidencePacket(
                    source=SignalSource.WEBCAM_STATE,
                    axis=SignalAxis.IDENTITY,
                    target_participant_id=pid,
                    delta_log_odds=0.3,
                    confidence=0.3,
                    rationale=f"Webcam is on (participant: {participant.display_name})",
                    timestamp=now,
                ))

        return evidence

    def extract_all(self) -> List[EvidencePacket]:
        evidence = []
        evidence.extend(self.extract_turn_taking())
        evidence.extend(self.extract_speaking_ratio())
        evidence.extend(self.extract_screen_share())
        evidence.extend(self.extract_webcam_state())
        return evidence
