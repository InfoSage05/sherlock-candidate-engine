from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .models import (
    BeliefState,
    EvidencePacket,
    FlagSeverity,
    IdentificationResult,
    MeetingContext,
    ParticipantMetrics,
    SignalAxis,
    SignalWeight,
    SnapshotEntry,
)


AMBIGUITY_THRESHOLD = 0.10


class FusionEngine:
    def __init__(self, context: MeetingContext, ambiguity_threshold: float = AMBIGUITY_THRESHOLD):
        self.context = context
        self.ambiguity_threshold = ambiguity_threshold
        self.beliefs: Dict[str, BeliefState] = {}
        self.identity_log_odds_sum: float = 0.0
        self.evidence_ledger: List[EvidencePacket] = []
        self.signal_weights: Dict[Tuple[str, str], SignalWeight] = {}
        self.snapshot_history: List[SnapshotEntry] = []
        self._init_beliefs()
        self._init_weights()

    def _init_beliefs(self) -> None:
        for pid in self.context.participants:
            self.beliefs[pid] = BeliefState(
                participant_id=pid,
                identity_log_odds=0.0,
                authenticity_log_odds=0.0,
                last_updated=datetime.utcnow(),
            )

    def _init_weights(self) -> None:
        default_weights = {
            ("calendar_match", "identity"): 3.0,
            ("interviewer_negative", "identity"): -2.5,
            ("email_domain", "identity"): 1.0,
            ("turn_taking", "identity"): 2.0,
            ("speaking_ratio", "identity"): 1.5,
            ("screen_share", "identity"): 1.5,
            ("llm_role_classifier", "identity"): 1.5,
            ("display_name_change", "identity"): 0.5,
            ("join_timing", "identity"): 1.0,
            ("webcam_state", "identity"): 0.5,
            ("disfluency_anomaly", "authenticity"): 1.5,
            ("pause_fluency_pattern", "authenticity"): 1.5,
            ("coding_telemetry", "authenticity"): 2.0,
            ("gaze_detection", "authenticity"): 1.5,
        }
        for (source, axis), weight in default_weights.items():
            self.signal_weights[(source, axis)] = SignalWeight(
                source=source,
                axis=axis,
                weight=weight,
            )

    def add_participant(self, participant_id: str) -> None:
        if participant_id not in self.beliefs:
            self.beliefs[participant_id] = BeliefState(
                participant_id=participant_id,
                last_updated=datetime.utcnow(),
            )

    def ingest(self, evidence: EvidencePacket) -> None:
        if evidence.target_participant_id not in self.beliefs:
            self.add_participant(evidence.target_participant_id)

        weight_key = (evidence.source.value, evidence.axis.value)
        weight = self.signal_weights.get(weight_key)
        calibrated_delta = evidence.delta_log_odds
        if weight and weight.calibration_count > 0:
            calibrated_delta = evidence.delta_log_odds * (weight.calibrated_weight / weight.weight)

        belief = self.beliefs[evidence.target_participant_id]

        if evidence.axis == SignalAxis.IDENTITY:
            belief.identity_log_odds += calibrated_delta
            belief.identity_evidence.append(evidence)
        else:
            belief.authenticity_log_odds += calibrated_delta
            belief.authenticity_evidence.append(evidence)

        belief.last_updated = evidence.timestamp
        self.evidence_ledger.append(evidence)

    def ingest_batch(self, evidence_list: List[EvidencePacket]) -> None:
        for ev in sorted(evidence_list, key=lambda e: e.timestamp):
            self.ingest(ev)

    def get_result(self, timestamp: Optional[datetime] = None) -> IdentificationResult:
        ts = timestamp or datetime.utcnow()

        if not self.beliefs:
            return IdentificationResult(
                meeting_id=self.context.meeting_id,
                timestamp=ts,
                top_candidate_id=None,
                top_candidate_probability=0.0,
                status="no_participants",
                all_hypotheses={},
                ambiguity_gap=0.0,
                rationale="No participants detected in the meeting.",
            )

        normalized = self._normalize_identity_beliefs()
        sorted_hypotheses = sorted(
            normalized.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        top_id, top_prob = sorted_hypotheses[0]
        second_prob = sorted_hypotheses[1][1] if len(sorted_hypotheses) > 1 else 0.0
        gap = top_prob - second_prob

        if gap < self.ambiguity_threshold:
            status = "ambiguous"
            rationale = self._build_ambiguous_rationale(sorted_hypotheses, gap)
        else:
            status = "identified"
            rationale = self._build_identified_rationale(top_id, top_prob, gap)

        result = IdentificationResult(
            meeting_id=self.context.meeting_id,
            timestamp=ts,
            top_candidate_id=top_id,
            top_candidate_probability=top_prob,
            status=status,
            all_hypotheses=dict(self.beliefs),
            ambiguity_gap=gap,
            rationale=rationale,
        )

        self._record_snapshot(result, ts, normalized)
        return result

    def _compute_participant_metrics(self, normalized: Dict[str, float], ts: datetime) -> Dict[str, ParticipantMetrics]:
        metrics = {}
        for pid, belief in self.beliefs.items():
            id_evidence_count = len(belief.identity_evidence) + len(belief.authenticity_evidence)
            metrics[pid] = ParticipantMetrics(
                participant_id=pid,
                identity_probability=normalized.get(pid, 0.0),
                authenticity_probability=belief.authenticity_probability,
                evidence_count=id_evidence_count,
            )
        return metrics

    def _collect_active_flags(self) -> List[EvidencePacket]:
        return [
            ep for ep in self.evidence_ledger
            if ep.severity not in (FlagSeverity.NONE,)
        ][-20:]

    def _compute_elapsed(self, ts: datetime) -> float:
        if not self.context.scheduled_start:
            return 0.0
        return (ts - self.context.scheduled_start).total_seconds()

    def _record_snapshot(self, result: IdentificationResult, ts: datetime,
                         normalized: Dict[str, float]) -> None:
        elapsed = self._compute_elapsed(ts)
        metrics = self._compute_participant_metrics(normalized, ts)
        flags = self._collect_active_flags()
        self.snapshot_history.append(SnapshotEntry(
            timestamp=ts,
            elapsed_seconds=elapsed,
            top_candidate_id=result.top_candidate_id,
            top_candidate_probability=result.top_candidate_probability,
            status=result.status,
            ambiguity_gap=result.ambiguity_gap,
            participants_metrics=metrics,
            active_flags=flags,
        ))

    def get_snapshot_history(self) -> List[SnapshotEntry]:
        return list(self.snapshot_history)

    def get_participant_timeline(self, participant_id: str, field: str = "identity_probability") -> List[Tuple[datetime, float]]:
        points = []
        for entry in self.snapshot_history:
            pm = entry.participants_metrics.get(participant_id)
            if pm:
                points.append((entry.timestamp, getattr(pm, field, 0.0)))
        return points

    def _normalize_identity_beliefs(self) -> Dict[str, float]:
        if not self.beliefs:
            return {}

        max_log_odds = max(b.identity_log_odds for b in self.beliefs.values())
        exp_values = {
            pid: math.exp(b.identity_log_odds - max_log_odds)
            for pid, b in self.beliefs.items()
        }
        total = sum(exp_values.values())
        if total == 0:
            return {pid: 1.0 / len(self.beliefs) for pid in self.beliefs}
        return {pid: v / total for pid, v in exp_values.items()}

    def _build_identified_rationale(
        self, top_id: str, top_prob: float, gap: float
    ) -> str:
        participant = self.context.participants.get(top_id)
        name = participant.display_name if participant else top_id
        belief = self.beliefs[top_id]

        lines = [
            f"Identified {name} (ID: {top_id}) as the candidate with "
            f"{top_prob:.1%} confidence (gap: {gap:.1%}).",
            f"Identity log-odds: {belief.identity_log_odds:.3f}",
            "",
            "Top contributing evidence:",
        ]

        top_evidence = sorted(
            belief.identity_evidence,
            key=lambda e: abs(e.delta_log_odds),
            reverse=True,
        )[:5]

        for ev in top_evidence:
            direction = "+" if ev.delta_log_odds > 0 else "-"
            lines.append(
                f"  [{direction}] {ev.source.value}: "
                f"delta_log_odds={ev.delta_log_odds:+.3f} | {ev.rationale}"
            )

        return "\n".join(lines)

    def _build_ambiguous_rationale(
        self, sorted_hypotheses: List[Tuple[str, float]], gap: float
    ) -> str:
        lines = [
            f"AMBIGUOUS — top hypotheses within {gap:.1%} of each other.",
            "Cannot reliably identify the candidate. Leading hypotheses:",
        ]
        for pid, prob in sorted_hypotheses[:3]:
            participant = self.context.participants.get(pid)
            name = participant.display_name if participant else pid
            belief = self.beliefs[pid]
            lines.append(
                f"  - {name} (ID: {pid}): {prob:.1%} "
                f"(log-odds: {belief.identity_log_odds:.3f}, "
                f"evidence count: {len(belief.identity_evidence)})"
            )
        lines.append("")
        lines.append("Recommendation: wait for more evidence or request interviewer confirmation.")
        return "\n".join(lines)

    def get_authenticity_report(self, participant_id: str) -> str:
        if participant_id not in self.beliefs:
            return f"No belief state for participant {participant_id}"

        belief = self.beliefs[participant_id]
        participant = self.context.participants.get(participant_id)
        name = participant.display_name if participant else participant_id

        lines = [
            f"Authenticity Report for {name} (ID: {participant_id})",
            f"Authenticity probability: {belief.authenticity_probability:.1%}",
            f"Authenticity log-odds: {belief.authenticity_log_odds:.3f}",
            "",
            "Evidence trail:",
        ]

        for ev in belief.authenticity_evidence:
            direction = "+" if ev.delta_log_odds > 0 else "-"
            lines.append(
                f"  [{direction}] {ev.source.value}: "
                f"delta_log_odds={ev.delta_log_odds:+.3f} | {ev.rationale}"
            )

        return "\n".join(lines)

    def get_evidence_ledger(self) -> List[EvidencePacket]:
        return list(self.evidence_ledger)

    def get_belief_history(self) -> List[Dict]:
        history = []
        for ev in self.evidence_ledger:
            if ev.axis != SignalAxis.IDENTITY:
                continue
            belief = self.beliefs.get(ev.target_participant_id)
            if not belief:
                continue
            normalized = self._normalize_identity_beliefs()
            history.append({
                "timestamp": ev.timestamp,
                "source": ev.source.value,
                "target": ev.target_participant_id,
                "delta_log_odds": ev.delta_log_odds,
                "posterior_probability": normalized.get(ev.target_participant_id, 0.0),
                "rationale": ev.rationale,
            })
        return history
