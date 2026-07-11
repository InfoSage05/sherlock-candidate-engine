from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from .fusion import FusionEngine
from .models import BeliefState, EvidencePacket, IdentificationResult, SignalAxis


class ExplanationLayer:
    def __init__(self, engine: FusionEngine):
        self.engine = engine

    def explain_identity(self, participant_id: Optional[str] = None) -> str:
        result = self.engine.get_result()

        if participant_id is None:
            return result.rationale

        if participant_id not in self.engine.beliefs:
            return f"No belief state for participant {participant_id}"

        belief = self.engine.beliefs[participant_id]
        participant = self.engine.context.participants.get(participant_id)
        name = participant.display_name if participant else participant_id

        lines = [
            f"Identity Analysis for {name} (ID: {participant_id})",
            f"{'=' * 60}",
            f"Identity log-odds: {belief.identity_log_odds:.3f}",
            f"Identity probability: {belief.identity_probability:.1%}",
            f"Evidence count: {len(belief.identity_evidence)}",
            "",
        ]

        positive_evidence = sorted(
            [e for e in belief.identity_evidence if e.delta_log_odds > 0],
            key=lambda e: e.delta_log_odds,
            reverse=True,
        )

        negative_evidence = sorted(
            [e for e in belief.identity_evidence if e.delta_log_odds < 0],
            key=lambda e: e.delta_log_odds,
        )

        if positive_evidence:
            lines.append("POSITIVE EVIDENCE (supports candidate hypothesis):")
            lines.append("-" * 60)
            for ev in positive_evidence[:5]:
                lines.append(f"  [{ev.source.value}]")
                lines.append(f"    Delta log-odds: {ev.delta_log_odds:+.3f}")
                lines.append(f"    Confidence: {ev.confidence:.2f}")
                lines.append(f"    Rationale: {ev.rationale}")
                lines.append(f"    Timestamp: {ev.timestamp.isoformat()}")
                lines.append("")

        if negative_evidence:
            lines.append("NEGATIVE EVIDENCE (against candidate hypothesis):")
            lines.append("-" * 60)
            for ev in negative_evidence[:5]:
                lines.append(f"  [{ev.source.value}]")
                lines.append(f"    Delta log-odds: {ev.delta_log_odds:+.3f}")
                lines.append(f"    Confidence: {ev.confidence:.2f}")
                lines.append(f"    Rationale: {ev.rationale}")
                lines.append(f"    Timestamp: {ev.timestamp.isoformat()}")
                lines.append("")

        return "\n".join(lines)

    def explain_authenticity(self, participant_id: str) -> str:
        return self.engine.get_authenticity_report(participant_id)

    def explain_comparison(self, participant_ids: List[str]) -> str:
        lines = [
            "Comparative Analysis",
            "=" * 60,
            "",
        ]

        beliefs = []
        for pid in participant_ids:
            if pid in self.engine.beliefs:
                belief = self.engine.beliefs[pid]
                participant = self.engine.context.participants.get(pid)
                name = participant.display_name if participant else pid
                beliefs.append((name, pid, belief))

        beliefs.sort(key=lambda x: x[2].identity_probability, reverse=True)

        lines.append(f"{'Participant':<30} {'Probability':<15} {'Log-Odds':<15} {'Evidence':<10}")
        lines.append("-" * 60)

        for name, pid, belief in beliefs:
            lines.append(
                f"{name:<30} {belief.identity_probability:>13.1%} "
                f"{belief.identity_log_odds:>13.3f} {len(belief.identity_evidence):>10}"
            )

        lines.append("")
        lines.append("TOP EVIDENCE PER PARTICIPANT:")
        lines.append("=" * 60)

        for name, pid, belief in beliefs[:3]:
            lines.append(f"\n{name} (ID: {pid}):")
            lines.append("-" * 60)

            top_evidence = sorted(
                belief.identity_evidence,
                key=lambda e: abs(e.delta_log_odds),
                reverse=True,
            )[:3]

            for ev in top_evidence:
                direction = "↑" if ev.delta_log_odds > 0 else "↓"
                lines.append(
                    f"  {direction} {ev.source.value}: {ev.delta_log_odds:+.3f} | {ev.rationale[:80]}"
                )

        return "\n".join(lines)

    def explain_timeline(self) -> str:
        history = self.engine.get_belief_history()

        if not history:
            return "No identity evidence recorded yet."

        lines = [
            "Identity Belief Timeline",
            "=" * 80,
            "",
            f"{'Timestamp':<25} {'Source':<25} {'Target':<15} {'Delta':<10} {'Posterior':<10}",
            "-" * 80,
        ]

        for entry in history[-20:]:
            lines.append(
                f"{entry['timestamp'].strftime('%H:%M:%S'):<25} "
                f"{entry['source']:<25} "
                f"{entry['target'][:12]:<15} "
                f"{entry['delta_log_odds']:>+8.3f} "
                f"{entry['posterior_probability']:>8.1%}"
            )

        return "\n".join(lines)

    def get_full_report(self) -> str:
        result = self.engine.get_result()

        lines = [
            "SHERLOCK CANDIDATE IDENTIFICATION REPORT",
            "=" * 80,
            f"Meeting ID: {result.meeting_id}",
            f"Timestamp: {result.timestamp.isoformat()}",
            f"Status: {result.status.upper()}",
            "",
        ]

        if result.top_candidate_id:
            participant = self.engine.context.participants.get(result.top_candidate_id)
            name = participant.display_name if result.top_candidate_id else result.top_candidate_id
            lines.append(f"TOP CANDIDATE: {name}")
            lines.append(f"  Probability: {result.top_candidate_probability:.1%}")
            lines.append(f"  Ambiguity gap: {result.ambiguity_gap:.1%}")
            lines.append("")

        lines.append("ALL HYPOTHESES:")
        lines.append("-" * 80)

        normalized = self.engine._normalize_identity_beliefs()
        sorted_hypotheses = sorted(
            normalized.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for pid, prob in sorted_hypotheses:
            participant = self.engine.context.participants.get(pid)
            name = participant.display_name if participant else pid
            belief = self.engine.beliefs[pid]
            lines.append(
                f"  {name:<30} {prob:>6.1%}  "
                f"(log-odds: {belief.identity_log_odds:+.3f}, "
                f"evidence: {len(belief.identity_evidence)})"
            )

        lines.append("")
        lines.append("RATIONALE:")
        lines.append("-" * 80)
        lines.append(result.rationale)

        return "\n".join(lines)
