import math
from datetime import datetime, timedelta

from sherlock.models import (
    EvidencePacket,
    MeetingContext,
    Participant,
    SignalAxis,
    SignalSource,
    SpeakingEvent,
    TranscriptSegment,
)
from sherlock.fusion import FusionEngine
from sherlock.signals.identity import IdentitySignalExtractor
from sherlock.signals.behavioral import BehavioralSignalExtractor
from sherlock.signals.authenticity import AuthenticitySignalExtractor


def test_basic_identity_detection():
    context = MeetingContext(
        meeting_id="test-001",
        candidate_name="Alice Johnson",
        candidate_email="alice@email.com",
        interviewer_names=["Bob Smith"],
        interviewer_emails=["bob@company.com"],
    )

    alice = Participant(
        id="alice",
        display_name="Alice Johnson",
        email="alice@email.com",
    )
    bob = Participant(
        id="bob",
        display_name="Bob Smith",
        email="bob@company.com",
    )
    context.participants = {"alice": alice, "bob": bob}

    engine = FusionEngine(context)
    extractor = IdentitySignalExtractor(context)
    evidence = extractor.extract_all()
    engine.ingest_batch(evidence)

    result = engine.get_result()
    assert result.top_candidate_id == "alice"
    assert result.top_candidate_probability > 0.6
    assert result.status == "identified"


def test_device_name_handling():
    context = MeetingContext(
        meeting_id="test-002",
        candidate_name="Charlie Brown",
        candidate_email="charlie@email.com",
        interviewer_names=["Diana"],
        interviewer_emails=["diana@company.com"],
    )

    charlie = Participant(
        id="charlie",
        display_name="MacBook Pro",
        email="charlie@email.com",
    )
    diana = Participant(
        id="diana",
        display_name="Diana",
        email="diana@company.com",
    )
    context.participants = {"charlie": charlie, "diana": diana}

    engine = FusionEngine(context)
    extractor = IdentitySignalExtractor(context)
    evidence = extractor.extract_all()
    engine.ingest_batch(evidence)

    result = engine.get_result()
    assert result.top_candidate_id == "charlie"
    assert result.top_candidate_probability > 0.5


def test_interviewer_negative_prior():
    context = MeetingContext(
        meeting_id="test-003",
        candidate_name="Eve",
        candidate_email="eve@email.com",
        interviewer_names=["Frank"],
        interviewer_emails=["frank@company.com"],
    )

    eve = Participant(id="eve", display_name="Eve", email="eve@email.com")
    frank = Participant(id="frank", display_name="Frank", email="frank@company.com")
    context.participants = {"eve": eve, "frank": frank}

    engine = FusionEngine(context)
    extractor = IdentitySignalExtractor(context)
    evidence = extractor.extract_all()
    engine.ingest_batch(evidence)

    result = engine.get_result()
    assert result.top_candidate_id == "eve"

    frank_belief = engine.beliefs["frank"]
    assert frank_belief.identity_log_odds < 0


def test_multiple_participants():
    context = MeetingContext(
        meeting_id="test-004",
        candidate_name="Grace",
        candidate_email="grace@email.com",
        interviewer_names=["Henry", "Ivy"],
        interviewer_emails=["henry@company.com", "ivy@company.com"],
    )

    grace = Participant(id="grace", display_name="Grace", email="grace@email.com")
    henry = Participant(id="henry", display_name="Henry", email="henry@company.com")
    ivy = Participant(id="ivy", display_name="Ivy", email="ivy@company.com")
    observer = Participant(id="observer", display_name="Observer", email="observer@company.com")
    context.participants = {
        "grace": grace,
        "henry": henry,
        "ivy": ivy,
        "observer": observer,
    }

    engine = FusionEngine(context)
    extractor = IdentitySignalExtractor(context)
    evidence = extractor.extract_all()
    engine.ingest_batch(evidence)

    result = engine.get_result()
    assert result.top_candidate_id == "grace"

    normalized = engine._normalize_identity_beliefs()
    assert sum(normalized.values()) > 0.99


def test_ambiguous_detection():
    context = MeetingContext(
        meeting_id="test-005",
        candidate_name=None,
        candidate_email=None,
        interviewer_names=[],
        interviewer_emails=[],
    )

    p1 = Participant(id="p1", display_name="Person One")
    p2 = Participant(id="p2", display_name="Person Two")
    context.participants = {"p1": p1, "p2": p2}

    engine = FusionEngine(context)

    engine.ingest(EvidencePacket(
        source=SignalSource.SPEAKING_RATIO,
        axis=SignalAxis.IDENTITY,
        target_participant_id="p1",
        delta_log_odds=0.5,
        confidence=0.5,
        rationale="test",
        timestamp=datetime.utcnow(),
    ))

    engine.ingest(EvidencePacket(
        source=SignalSource.SPEAKING_RATIO,
        axis=SignalAxis.IDENTITY,
        target_participant_id="p2",
        delta_log_odds=0.5,
        confidence=0.5,
        rationale="test",
        timestamp=datetime.utcnow(),
    ))

    result = engine.get_result()
    assert result.status == "ambiguous"


def test_log_odds_accumulation():
    context = MeetingContext(
        meeting_id="test-006",
        candidate_name="Test",
        candidate_email="test@email.com",
    )

    p = Participant(id="p1", display_name="Test", email="test@email.com")
    context.participants = {"p1": p}

    engine = FusionEngine(context)

    engine.ingest(EvidencePacket(
        source=SignalSource.CALENDAR_MATCH,
        axis=SignalAxis.IDENTITY,
        target_participant_id="p1",
        delta_log_odds=1.0,
        confidence=0.8,
        rationale="name match",
        timestamp=datetime.utcnow(),
    ))

    engine.ingest(EvidencePacket(
        source=SignalSource.SPEAKING_RATIO,
        axis=SignalAxis.IDENTITY,
        target_participant_id="p1",
        delta_log_odds=0.5,
        confidence=0.6,
        rationale="speaking ratio",
        timestamp=datetime.utcnow(),
    ))

    belief = engine.beliefs["p1"]
    assert abs(belief.identity_log_odds - 1.5) < 0.01


def test_evidence_ledger():
    context = MeetingContext(
        meeting_id="test-007",
        candidate_name="Test",
    )

    p = Participant(id="p1", display_name="Test")
    context.participants = {"p1": p}

    engine = FusionEngine(context)

    ev1 = EvidencePacket(
        source=SignalSource.CALENDAR_MATCH,
        axis=SignalAxis.IDENTITY,
        target_participant_id="p1",
        delta_log_odds=1.0,
        confidence=0.8,
        rationale="name match",
        timestamp=datetime.utcnow(),
    )
    ev2 = EvidencePacket(
        source=SignalSource.SPEAKING_RATIO,
        axis=SignalAxis.IDENTITY,
        target_participant_id="p1",
        delta_log_odds=0.5,
        confidence=0.6,
        rationale="speaking ratio",
        timestamp=datetime.utcnow(),
    )

    engine.ingest(ev1)
    engine.ingest(ev2)

    ledger = engine.get_evidence_ledger()
    assert len(ledger) == 2
    assert ledger[0].source == SignalSource.CALENDAR_MATCH
    assert ledger[1].source == SignalSource.SPEAKING_RATIO


def test_speaking_ratio_signal():
    context = MeetingContext(
        meeting_id="test-008",
        candidate_name="Alice",
    )

    alice = Participant(id="alice", display_name="Alice")
    bob = Participant(id="bob", display_name="Bob")
    context.participants = {"alice": alice, "bob": bob}

    extractor = BehavioralSignalExtractor(context)

    for _ in range(5):
        extractor.add_speaking_event(SpeakingEvent(
            participant_id="alice",
            start_time=datetime.utcnow() - timedelta(minutes=2),
            end_time=datetime.utcnow() - timedelta(minutes=1),
            is_response_to_question=True,
        ))

    extractor.add_speaking_event(SpeakingEvent(
        participant_id="bob",
        start_time=datetime.utcnow() - timedelta(minutes=3),
        end_time=datetime.utcnow() - timedelta(minutes=2, seconds=50),
        is_response_to_question=False,
    ))

    evidence = extractor.extract_speaking_ratio()
    assert len(evidence) > 0

    alice_evidence = [e for e in evidence if e.target_participant_id == "alice"]
    assert len(alice_evidence) > 0
    assert alice_evidence[0].delta_log_odds > 0


def test_turn_taking_signal():
    context = MeetingContext(
        meeting_id="test-009",
        candidate_name="Alice",
    )

    alice = Participant(id="alice", display_name="Alice")
    bob = Participant(id="bob", display_name="Bob")
    context.participants = {"alice": alice, "bob": bob}

    extractor = BehavioralSignalExtractor(context)

    for i in range(5):
        extractor.add_transcript_segment(TranscriptSegment(
            participant_id="bob",
            text=f"Question {i+1}?",
            start_time=datetime.utcnow() - timedelta(minutes=5-i),
            end_time=datetime.utcnow() - timedelta(minutes=5-i, seconds=5),
            is_question=True,
        ))
        extractor.add_transcript_segment(TranscriptSegment(
            participant_id="alice",
            text=f"Answer {i+1}",
            start_time=datetime.utcnow() - timedelta(minutes=4-i, seconds=50),
            end_time=datetime.utcnow() - timedelta(minutes=4-i, seconds=30),
            is_question=False,
        ))

    evidence = extractor.extract_turn_taking()
    assert len(evidence) > 0

    alice_evidence = [e for e in evidence if e.target_participant_id == "alice"]
    assert len(alice_evidence) > 0
    assert alice_evidence[0].delta_log_odds > 0


def test_authenticity_disfluency():
    context = MeetingContext(
        meeting_id="test-010",
        candidate_name="Alice",
    )

    alice = Participant(id="alice", display_name="Alice")
    context.participants = {"alice": alice}

    extractor = AuthenticitySignalExtractor(context)

    for i in range(8):
        extractor.add_transcript_segment(TranscriptSegment(
            participant_id="alice",
            text=f"Um, well, like, basically, you know, answer {i+1} with lots of filler words and stuff",
            start_time=datetime.utcnow() - timedelta(minutes=10-i),
            end_time=datetime.utcnow() - timedelta(minutes=10-i, seconds=30),
        ))

    extractor._get_or_compute_baseline("alice")

    for i in range(3):
        extractor.add_transcript_segment(TranscriptSegment(
            participant_id="alice",
            text=f"Perfectly structured answer {i+1} without any filler words at all",
            start_time=datetime.utcnow() - timedelta(minutes=2-i),
            end_time=datetime.utcnow() - timedelta(minutes=2-i, seconds=30),
        ))

    evidence = extractor.extract_disfluency_anomaly()
    assert len(evidence) > 0
    assert evidence[0].delta_log_odds < 0


def test_probability_normalization():
    context = MeetingContext(
        meeting_id="test-011",
        candidate_name="Test",
    )

    p1 = Participant(id="p1", display_name="P1")
    p2 = Participant(id="p2", display_name="P2")
    p3 = Participant(id="p3", display_name="P3")
    context.participants = {"p1": p1, "p2": p2, "p3": p3}

    engine = FusionEngine(context)

    engine.ingest(EvidencePacket(
        source=SignalSource.CALENDAR_MATCH,
        axis=SignalAxis.IDENTITY,
        target_participant_id="p1",
        delta_log_odds=2.0,
        confidence=0.9,
        rationale="strong match",
        timestamp=datetime.utcnow(),
    ))

    engine.ingest(EvidencePacket(
        source=SignalSource.CALENDAR_MATCH,
        axis=SignalAxis.IDENTITY,
        target_participant_id="p2",
        delta_log_odds=0.5,
        confidence=0.5,
        rationale="weak match",
        timestamp=datetime.utcnow(),
    ))

    normalized = engine._normalize_identity_beliefs()
    total = sum(normalized.values())
    assert abs(total - 1.0) < 0.01


def test_separate_authenticity_stream():
    context = MeetingContext(
        meeting_id="test-012",
        candidate_name="Test",
    )

    p = Participant(id="p1", display_name="Test")
    context.participants = {"p1": p}

    engine = FusionEngine(context)

    engine.ingest(EvidencePacket(
        source=SignalSource.CALENDAR_MATCH,
        axis=SignalAxis.IDENTITY,
        target_participant_id="p1",
        delta_log_odds=2.0,
        confidence=0.9,
        rationale="identity signal",
        timestamp=datetime.utcnow(),
    ))

    engine.ingest(EvidencePacket(
        source=SignalSource.DISFLUENCY_ANOMALY,
        axis=SignalAxis.AUTHENTICITY,
        target_participant_id="p1",
        delta_log_odds=1.5,
        confidence=0.7,
        rationale="authenticity signal",
        timestamp=datetime.utcnow(),
    ))

    belief = engine.beliefs["p1"]
    assert belief.identity_log_odds > 0
    assert belief.authenticity_log_odds > 0
    assert len(belief.identity_evidence) == 1
    assert len(belief.authenticity_evidence) == 1


if __name__ == "__main__":
    test_basic_identity_detection()
    test_device_name_handling()
    test_interviewer_negative_prior()
    test_multiple_participants()
    test_ambiguous_detection()
    test_log_odds_accumulation()
    test_evidence_ledger()
    test_speaking_ratio_signal()
    test_turn_taking_signal()
    test_authenticity_disfluency()
    test_probability_normalization()
    test_separate_authenticity_stream()
    print("All tests passed!")
