from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .fusion import FusionEngine
from .models import (
    EvidencePacket,
    FeedbackEvent,
    MeetingContext,
    SignalAxis,
    SignalSource,
    SignalWeight,
)


class FeedbackLoop:
    def __init__(self, engine: FusionEngine):
        self.engine = engine
        self.feedback_history: List[FeedbackEvent] = []

    def record_confirmation(self, participant_id: str, notes: str = "") -> None:
        feedback = FeedbackEvent(
            meeting_id=self.engine.context.meeting_id,
            timestamp=datetime.utcnow(),
            feedback_type="identity_confirmation",
            participant_id=participant_id,
            correct=True,
            notes=notes,
        )
        self.feedback_history.append(feedback)
        self._recalibrate_weights(feedback)

    def record_correction(self, correct_participant_id: str, notes: str = "") -> None:
        feedback = FeedbackEvent(
            meeting_id=self.engine.context.meeting_id,
            timestamp=datetime.utcnow(),
            feedback_type="identity_correction",
            participant_id=correct_participant_id,
            correct=False,
            notes=notes,
        )
        self.feedback_history.append(feedback)
        self._recalibrate_weights(feedback)

    def record_authenticity_flag(self, participant_id: str, confirmed: bool, notes: str = "") -> None:
        feedback_type = "authenticity_confirmed" if confirmed else "authenticity_dismissed"
        feedback = FeedbackEvent(
            meeting_id=self.engine.context.meeting_id,
            timestamp=datetime.utcnow(),
            feedback_type=feedback_type,
            participant_id=participant_id,
            correct=confirmed,
            notes=notes,
        )
        self.feedback_history.append(feedback)

    def _recalibrate_weights(self, feedback: FeedbackEvent) -> None:
        if feedback.feedback_type == "identity_confirmation":
            self._boost_supporting_signals(feedback.participant_id, factor=1.05)

        elif feedback.feedback_type == "identity_correction":
            current_top = self.engine.get_result().top_candidate_id
            if current_top and current_top != feedback.participant_id:
                self._boost_supporting_signals(feedback.participant_id, factor=1.1)
                self._penalize_conflicting_signals(current_top, factor=0.9)

    def _boost_supporting_signals(self, participant_id: str, factor: float) -> None:
        belief = self.engine.beliefs.get(participant_id)
        if not belief:
            return

        for evidence in belief.identity_evidence:
            weight_key = (evidence.source.value, evidence.axis.value)
            if weight_key in self.engine.signal_weights:
                weight = self.engine.signal_weights[weight_key]
                if evidence.delta_log_odds > 0:
                    weight.calibration_sum += evidence.delta_log_odds * factor
                    weight.calibration_count += 1

    def _penalize_conflicting_signals(self, participant_id: str, factor: float) -> None:
        belief = self.engine.beliefs.get(participant_id)
        if not belief:
            return

        for evidence in belief.identity_evidence:
            weight_key = (evidence.source.value, evidence.axis.value)
            if weight_key in self.engine.signal_weights:
                weight = self.engine.signal_weights[weight_key]
                if evidence.delta_log_odds > 0:
                    weight.calibration_count += 1

    def get_feedback_summary(self) -> str:
        if not self.feedback_history:
            return "No feedback recorded yet."

        lines = [
            "FEEDBACK SUMMARY",
            "=" * 80,
            f"Total feedback events: {len(self.feedback_history)}",
            "",
        ]

        confirmations = sum(1 for f in self.feedback_history if f.correct)
        corrections = sum(1 for f in self.feedback_history if not f.correct)

        lines.append(f"Confirmations: {confirmations}")
        lines.append(f"Corrections: {corrections}")
        lines.append("")

        lines.append("SIGNAL WEIGHT CALIBRATION:")
        lines.append("-" * 80)

        for (source, axis), weight in sorted(self.engine.signal_weights.items()):
            if weight.calibration_count > 0:
                lines.append(
                    f"  {source:<30} {axis:<15} "
                    f"original: {weight.weight:.2f}  "
                    f"calibrated: {weight.calibrated_weight:.2f}  "
                    f"(n={weight.calibration_count})"
                )

        return "\n".join(lines)

    def get_calibration_report(self) -> Dict:
        report = {
            "total_feedback": len(self.feedback_history),
            "confirmations": sum(1 for f in self.feedback_history if f.correct),
            "corrections": sum(1 for f in self.feedback_history if not f.correct),
            "signal_weights": {},
        }

        for (source, axis), weight in self.engine.signal_weights.items():
            key = f"{source.value}_{axis.value}"
            report["signal_weights"][key] = {
                "original_weight": weight.weight,
                "calibrated_weight": weight.calibrated_weight,
                "calibration_count": weight.calibration_count,
            }

        return report
