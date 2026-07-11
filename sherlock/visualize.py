from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from .fusion import FusionEngine
from .models import BeliefState, EvidencePacket, SignalAxis


def generate_ascii_belief_chart(
    engine: FusionEngine,
    width: int = 60,
    height: int = 10,
) -> str:
    normalized = engine._normalize_identity_beliefs()

    if not normalized:
        return "No participants to display."

    sorted_participants = sorted(
        normalized.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    lines = []
    lines.append("IDENTITY BELIEF DISTRIBUTION")
    lines.append("=" * (width + 30))

    for pid, prob in sorted_participants:
        participant = engine.context.participants.get(pid)
        name = participant.display_name if participant else pid
        name = name[:20]

        bar_length = int(prob * width)
        bar = "█" * bar_length + "░" * (width - bar_length)

        lines.append(f"{name:<20} |{bar}| {prob:>6.1%}")

    return "\n".join(lines)


def generate_ascii_timeline(
    engine: FusionEngine,
    participant_id: str,
    width: int = 60,
) -> str:
    history = engine.get_belief_history()
    participant_history = [h for h in history if h["target"] == participant_id]

    if not participant_history:
        return f"No history for participant {participant_id}"

    lines = []
    lines.append(f"BELIEF TIMELINE FOR {participant_id}")
    lines.append("=" * (width + 20))

    probs = [h["posterior_probability"] for h in participant_history]
    max_prob = max(probs) if probs else 1.0

    for entry in participant_history[-20:]:
        prob = entry["posterior_probability"]
        bar_length = int((prob / max_prob) * width) if max_prob > 0 else 0
        bar = "▓" * bar_length

        timestamp = entry["timestamp"].strftime("%H:%M:%S")
        lines.append(
            f"{timestamp} |{bar:<{width}}| {prob:>6.1%} ({entry['source']})"
        )

    return "\n".join(lines)


def generate_evidence_heatmap(
    engine: FusionEngine,
) -> str:
    lines = []
    lines.append("EVIDENCE HEATMAP (log-odds contribution)")
    lines.append("=" * 80)

    sources = set()
    participants = list(engine.beliefs.keys())

    for belief in engine.beliefs.values():
        for ev in belief.identity_evidence:
            sources.add(ev.source.value)

    sources = sorted(sources)

    header = f"{'Source':<30}"
    for pid in participants:
        participant = engine.context.participants.get(pid)
        name = participant.display_name[:10] if participant else pid[:10]
        header += f" {name:>10}"
    lines.append(header)
    lines.append("-" * 80)

    for source in sources:
        row = f"{source:<30}"
        for pid in participants:
            belief = engine.beliefs.get(pid)
            if not belief:
                row += f" {'N/A':>10}"
                continue

            total_delta = sum(
                ev.delta_log_odds
                for ev in belief.identity_evidence
                if ev.source.value == source
            )

            if abs(total_delta) < 0.01:
                row += f" {'0.0':>10}"
            elif total_delta > 0:
                row += f" {'+' + f'{total_delta:.1f}':>10}"
            else:
                row += f" {total_delta:>10.1f}"

        lines.append(row)

    return "\n".join(lines)


def generate_confidence_gauge(
    probability: float,
    label: str = "Confidence",
) -> str:
    gauge_width = 40
    filled = int(probability * gauge_width)
    empty = gauge_width - filled

    gauge = "█" * filled + "░" * empty

    lines = [
        f"{label}: {probability:.1%}",
        f"[{gauge}]",
    ]

    if probability > 0.8:
        lines.append("Status: HIGH CONFIDENCE")
    elif probability > 0.6:
        lines.append("Status: MODERATE CONFIDENCE")
    elif probability > 0.4:
        lines.append("Status: LOW CONFIDENCE")
    else:
        lines.append("Status: AMBIGUOUS")

    return "\n".join(lines)


def generate_full_visualization(engine: FusionEngine) -> str:
    result = engine.get_result()

    sections = []

    sections.append(generate_ascii_belief_chart(engine))

    sections.append("")
    sections.append(generate_confidence_gauge(
        result.top_candidate_probability,
        "Top Candidate Confidence"
    ))

    sections.append("")
    sections.append(generate_evidence_heatmap(engine))

    if result.top_candidate_id:
        sections.append("")
        sections.append(generate_ascii_timeline(engine, result.top_candidate_id))

    return "\n".join(sections)


def print_visualization(engine: FusionEngine) -> None:
    print("\n" + generate_full_visualization(engine))
