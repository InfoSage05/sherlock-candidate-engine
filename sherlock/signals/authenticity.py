from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ..models import (
    CodingEvent,
    EvidencePacket,
    FlagSeverity,
    GazeEvent,
    MeetingContext,
    SignalAxis,
    SignalSource,
    SpeakingEvent,
    TranscriptSegment,
)


FILLER_WORDS = {
    "um", "uh", "er", "ah", "like", "you know", "so", "well",
    "basically", "actually", "i mean", "right", "okay",
}


class AuthenticitySignalExtractor:
    def __init__(self, context: MeetingContext):
        self.context = context
        self.speaking_events: List[SpeakingEvent] = []
        self.transcript_segments: List[TranscriptSegment] = []
        self.coding_events: List[CodingEvent] = []
        self.gaze_events: List[GazeEvent] = []
        self._participant_baselines: Dict[str, Dict] = {}

    def add_speaking_event(self, event: SpeakingEvent) -> None:
        self.speaking_events.append(event)

    def add_transcript_segment(self, segment: TranscriptSegment) -> None:
        self.transcript_segments.append(segment)

    def add_coding_event(self, event: CodingEvent) -> None:
        self.coding_events.append(event)

    def add_gaze_event(self, event: GazeEvent) -> None:
        self.gaze_events.append(event)

    def _compute_disfluency_rate(self, text: str) -> float:
        import re
        words = re.sub(r'[^\w\s]', '', text.lower()).split()
        if not words:
            return 0.0
        filler_count = sum(1 for w in words if w in FILLER_WORDS)
        return filler_count / len(words)

    def _compute_structural_fluency(self, text: str) -> float:
        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        if not sentences:
            return 0.0

        sentence_lengths = [len(s.split()) for s in sentences]
        if not sentence_lengths:
            return 0.0

        mean_len = sum(sentence_lengths) / len(sentence_lengths)
        variance = sum((l - mean_len) ** 2 for l in sentence_lengths) / len(sentence_lengths)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0

        cv = std_dev / mean_len if mean_len > 0 else 0.0
        fluency = 1.0 / (1.0 + cv)
        return fluency

    def _get_or_compute_baseline(self, participant_id: str) -> Dict:
        if participant_id in self._participant_baselines:
            return self._participant_baselines[participant_id]

        participant_segments = [
            s for s in self.transcript_segments
            if s.participant_id == participant_id
        ]

        if not participant_segments:
            return {"disfluency_rate": 0.0, "structural_fluency": 0.5}

        disfluency_rates = [self._compute_disfluency_rate(s.text) for s in participant_segments]
        fluency_scores = [self._compute_structural_fluency(s.text) for s in participant_segments]

        baseline = {
            "disfluency_rate": sum(disfluency_rates) / len(disfluency_rates) if disfluency_rates else 0.0,
            "structural_fluency": sum(fluency_scores) / len(fluency_scores) if fluency_scores else 0.5,
        }

        self._participant_baselines[participant_id] = baseline
        return baseline

    def extract_disfluency_anomaly(self) -> List[EvidencePacket]:
        evidence = []
        now = datetime.utcnow()

        participant_segments: Dict[str, List[TranscriptSegment]] = {}
        for segment in self.transcript_segments:
            if segment.participant_id not in participant_segments:
                participant_segments[segment.participant_id] = []
            participant_segments[segment.participant_id].append(segment)

        for pid, segments in participant_segments.items():
            if len(segments) < 3:
                continue

            baseline = self._get_or_compute_baseline(pid)
            baseline_disfluency = baseline["disfluency_rate"]

            segment_disfluencies = [self._compute_disfluency_rate(s.text) for s in segments]
            recent_disfluency = sum(segment_disfluencies[-3:]) / 3

            if baseline_disfluency > 0.02:
                anomaly_ratio = recent_disfluency / baseline_disfluency
            else:
                anomaly_ratio = 1.0 if recent_disfluency < 0.01 else 0.0

            if anomaly_ratio < 0.3 and baseline_disfluency > 0.02:
                delta = -(1.0 - anomaly_ratio) * 1.5
                evidence.append(EvidencePacket(
                    source=SignalSource.DISFLUENCY_ANOMALY,
                    axis=SignalAxis.AUTHENTICITY,
                    target_participant_id=pid,
                    delta_log_odds=delta,
                    confidence=min(1.0 - anomaly_ratio, 1.0),
                    severity=FlagSeverity.WARNING,
                    flag_type="disfluency_anomaly",
                    recommendation="Check if the candidate is reading from a script or another screen.",
                    rationale=(
                        f"Disfluency rate dropped significantly (decreased authenticity): "
                        f"baseline={baseline_disfluency:.3f}, recent={recent_disfluency:.3f} "
                        f"(ratio: {anomaly_ratio:.2f})"
                    ),
                    timestamp=now,
                    metadata={
                        "baseline_disfluency": baseline_disfluency,
                        "recent_disfluency": recent_disfluency,
                        "anomaly_ratio": anomaly_ratio,
                    },
                ))

        return evidence

    def extract_pause_fluency_pattern(self) -> List[EvidencePacket]:
        evidence = []
        now = datetime.utcnow()

        questions = [s for s in self.transcript_segments if s.is_question]

        for question in questions:
            for segment in self.transcript_segments:
                if segment.participant_id == question.participant_id:
                    continue

                time_gap = (segment.start_time - question.end_time).total_seconds()
                if 5 < time_gap < 30:
                    fluency = self._compute_structural_fluency(segment.text)

                    if fluency > 0.85 and time_gap > 10:
                        delta = -(fluency - 0.5) * 2.0
                        evidence.append(EvidencePacket(
                            source=SignalSource.PAUSE_FLUENCY_PATTERN,
                            axis=SignalAxis.AUTHENTICITY,
                            target_participant_id=segment.participant_id,
                            delta_log_odds=delta,
                            confidence=min(fluency, 1.0),
                            severity=FlagSeverity.WARNING,
                            flag_type="pause_fluency_pattern",
                            recommendation="Long pause followed by polished answer may indicate AI assistance.",
                            rationale=(
                                f"Long pause ({time_gap:.1f}s) after hard question, "
                                f"followed by highly fluent answer (fluency: {fluency:.2f})"
                                f" — decreased authenticity"
                            ),
                            timestamp=now,
                            metadata={
                                "pause_duration": time_gap,
                                "answer_fluency": fluency,
                                "question_text": question.text[:50],
                            },
                        ))
                    break

        return evidence

    def extract_coding_telemetry(self) -> List[EvidencePacket]:
        evidence = []
        now = datetime.utcnow()

        participant_events: Dict[str, List[CodingEvent]] = {}
        for event in self.coding_events:
            if event.participant_id not in participant_events:
                participant_events[event.participant_id] = []
            participant_events[event.participant_id].append(event)

        for pid, events in participant_events.items():
            paste_events = [e for e in events if e.event_type == "paste"]
            burst_events = [e for e in events if e.event_type == "burst" and e.content_length > 50]

            if paste_events:
                for paste in paste_events:
                    evidence.append(EvidencePacket(
                        source=SignalSource.CODING_TELEMETRY,
                        axis=SignalAxis.AUTHENTICITY,
                        target_participant_id=pid,
                        delta_log_odds=-1.5,
                        confidence=0.7,
                        severity=FlagSeverity.CRITICAL,
                        flag_type="paste_event",
                        recommendation="Large paste may indicate copy-pasted code. Verify candidate's understanding.",
                        rationale=(
                            f"Paste event detected (decreased authenticity): {paste.content_length} characters "
                            f"at {paste.timestamp.isoformat()}"
                        ),
                        timestamp=now,
                        metadata={
                            "event_type": "paste",
                            "content_length": paste.content_length,
                        },
                    ))

            if burst_events:
                for burst in burst_events:
                    if burst.keystroke_interval_ms and burst.keystroke_interval_ms < 10:
                        evidence.append(EvidencePacket(
                            source=SignalSource.CODING_TELEMETRY,
                            axis=SignalAxis.AUTHENTICITY,
                            target_participant_id=pid,
                            delta_log_odds=-1.0,
                            confidence=0.6,
                            severity=FlagSeverity.WARNING,
                            flag_type="rapid_burst",
                            recommendation="Unnaturally fast typing may indicate pre-written code.",
                            rationale=(
                                f"Rapid burst (decreased authenticity): {burst.content_length} chars with "
                                f"keystroke interval {burst.keystroke_interval_ms:.1f}ms"
                            ),
                            timestamp=now,
                            metadata={
                                "event_type": "burst",
                                "content_length": burst.content_length,
                                "keystroke_interval_ms": burst.keystroke_interval_ms,
                            },
                        ))

        return evidence

    def extract_gaze_detection(self) -> List[EvidencePacket]:
        evidence = []
        now = datetime.utcnow()

        participant_gaze: Dict[str, List[GazeEvent]] = {}
        for event in self.gaze_events:
            if event.participant_id not in participant_gaze:
                participant_gaze[event.participant_id] = []
            participant_gaze[event.participant_id].append(event)

        for pid, events in participant_gaze.items():
            off_screen_events = [e for e in events if e.is_off_screen and e.saccade_periodicity]
            if not off_screen_events:
                continue

            periodicities = [e.saccade_periodicity for e in off_screen_events if e.saccade_periodicity]
            if not periodicities:
                continue

            avg_periodicity = sum(periodicities) / len(periodicities)

            if avg_periodicity < 2.0:
                delta = -(2.0 - avg_periodicity) * 0.75
                evidence.append(EvidencePacket(
                    source=SignalSource.GAZE_DETECTION,
                    axis=SignalAxis.AUTHENTICITY,
                    target_participant_id=pid,
                    delta_log_odds=delta,
                    confidence=min(1.0, delta / -1.5),
                    severity=FlagSeverity.WARNING,
                    flag_type="periodic_gaze",
                    recommendation="Rhythmic off-screen gaze pattern suggests reading from another source.",
                    rationale=(
                        f"Periodic off-screen gaze pattern detected (decreased authenticity): "
                        f"avg saccade periodicity {avg_periodicity:.2f}s "
                        f"(reading pattern signature)"
                    ),
                    timestamp=now,
                    metadata={
                        "avg_periodicity": avg_periodicity,
                        "off_screen_event_count": len(off_screen_events),
                    },
                ))

        return evidence

    def extract_all(self) -> List[EvidencePacket]:
        evidence = []
        evidence.extend(self.extract_disfluency_anomaly())
        evidence.extend(self.extract_pause_fluency_pattern())
        evidence.extend(self.extract_coding_telemetry())
        evidence.extend(self.extract_gaze_detection())
        return evidence
