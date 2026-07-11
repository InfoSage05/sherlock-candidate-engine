"""Shared test helpers."""

from datetime import datetime

from sherlock.fusion import FusionEngine
from sherlock.models import (
    EvidencePacket,
    MeetingContext,
    Participant,
    SignalAxis,
    SignalSource,
)


def make_context(participant_ids, candidate_name="Alice"):
    participants = {
        pid: Participant(id=pid, display_name=pid.capitalize())
        for pid in participant_ids
    }
    return MeetingContext(
        meeting_id="test-meeting",
        candidate_name=candidate_name,
        participants=participants,
        scheduled_start=datetime.utcnow(),
    )


def make_identified_engine(participant_ids, candidate_id, interviewer_id):
    """Build an engine where ``candidate_id`` is clearly identified."""
    context = make_context(participant_ids)
    engine = FusionEngine(context)
    engine.ingest(EvidencePacket(
        source=SignalSource.CALENDAR_MATCH, axis=SignalAxis.IDENTITY,
        target_participant_id=candidate_id, delta_log_odds=2.0, confidence=0.8,
        rationale="calendar match", timestamp=datetime.utcnow(),
    ))
    engine.ingest(EvidencePacket(
        source=SignalSource.INTERVIEWER_NEGATIVE, axis=SignalAxis.IDENTITY,
        target_participant_id=interviewer_id, delta_log_odds=-2.5, confidence=0.9,
        rationale="known interviewer", timestamp=datetime.utcnow(),
    ))
    return context, engine
