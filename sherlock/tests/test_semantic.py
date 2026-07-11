import math
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from sherlock.llm_client import MockLLMClient
from sherlock.models import (
    EvidencePacket,
    MeetingContext,
    Participant,
    SignalAxis,
    SignalSource,
    TranscriptSegment,
)
from sherlock.signals.semantic import (
    BATCH_WINDOW_SECONDS,
    MIN_CONFIDENCE,
    SemanticSignalExtractor,
    confidence_to_log_odds,
)


def test_logit_conversion_candidate_high_confidence():
    log_odds = confidence_to_log_odds(0.9, "candidate")
    assert log_odds > 0
    assert abs(log_odds - math.log(0.9 / 0.1)) < 0.01


def test_logit_conversion_interviewer_high_confidence():
    log_odds = confidence_to_log_odds(0.9, "interviewer")
    assert log_odds < 0
    assert abs(log_odds - (-math.log(0.9 / 0.1))) < 0.01


def test_logit_conversion_50_percent():
    log_odds = confidence_to_log_odds(0.5, "candidate")
    assert abs(log_odds) < 0.01


def test_logit_conversion_clamps_low_confidence():
    log_odds = confidence_to_log_odds(0.1, "candidate")
    expected = math.log(MIN_CONFIDENCE / (1 - MIN_CONFIDENCE))
    assert abs(log_odds - expected) < 0.01


def test_logit_conversion_clamps_high_confidence():
    log_odds = confidence_to_log_odds(0.99, "candidate")
    expected = math.log(0.95 / 0.05)
    assert abs(log_odds - expected) < 0.01


def test_logit_conversion_symmetric():
    candidate_log_odds = confidence_to_log_odds(0.8, "candidate")
    interviewer_log_odds = confidence_to_log_odds(0.8, "interviewer")
    assert abs(candidate_log_odds + interviewer_log_odds) < 0.01


def test_evidence_packet_schema():
    packet = EvidencePacket(
        source=SignalSource.LLM_ROLE_CLASSIFIER,
        axis=SignalAxis.IDENTITY,
        target_participant_id="test-participant",
        delta_log_odds=1.5,
        confidence=0.8,
        rationale="Test rationale",
        timestamp=datetime.utcnow(),
        metadata={"test": "data"},
    )

    assert packet.source == SignalSource.LLM_ROLE_CLASSIFIER
    assert packet.axis == SignalAxis.IDENTITY
    assert packet.target_participant_id == "test-participant"
    assert packet.delta_log_odds == 1.5
    assert packet.confidence == 0.8
    assert packet.rationale == "Test rationale"
    assert packet.evidence_id
    assert packet.delta_probability > 0


def test_semantic_extractor_disabled():
    context = MeetingContext(
        meeting_id="test",
        candidate_name="Alice",
    )
    participant = Participant(id="alice", display_name="Alice")
    context.participants = {"alice": participant}

    mock_client = MockLLMClient()
    extractor = SemanticSignalExtractor(context, llm_client=mock_client, enabled=False)

    extractor.add_transcript_segment(TranscriptSegment(
        participant_id="alice",
        text="I have 5 years of experience in software engineering.",
        start_time=datetime.utcnow() - timedelta(minutes=1),
        end_time=datetime.utcnow(),
    ))

    evidence = extractor.extract_all()
    assert len(evidence) == 0


def test_semantic_extractor_no_client():
    context = MeetingContext(
        meeting_id="test",
        candidate_name="Alice",
    )
    participant = Participant(id="alice", display_name="Alice")
    context.participants = {"alice": participant}

    extractor = SemanticSignalExtractor(context, llm_client=None)

    extractor.add_transcript_segment(TranscriptSegment(
        participant_id="alice",
        text="I have 5 years of experience in software engineering.",
        start_time=datetime.utcnow() - timedelta(minutes=1),
        end_time=datetime.utcnow(),
    ))

    evidence = extractor.extract_all()
    assert len(evidence) == 0


def test_semantic_extractor_short_segments_filtered():
    context = MeetingContext(
        meeting_id="test",
        candidate_name="Alice",
    )
    participant = Participant(id="alice", display_name="Alice")
    context.participants = {"alice": participant}

    mock_client = MockLLMClient([
        {"structured": {"role_guess": "candidate", "confidence": 0.8, "rationale": "test"}}
    ])
    extractor = SemanticSignalExtractor(context, llm_client=mock_client)

    extractor.add_transcript_segment(TranscriptSegment(
        participant_id="alice",
        text="Short",
        start_time=datetime.utcnow() - timedelta(minutes=1),
        end_time=datetime.utcnow(),
    ))

    evidence = extractor.extract_all()
    assert len(evidence) == 0


def test_semantic_extractor_batching():
    context = MeetingContext(
        meeting_id="test",
        candidate_name="Alice",
    )
    participant = Participant(id="alice", display_name="Alice")
    context.participants = {"alice": participant}

    mock_client = MockLLMClient([
        {"structured": {"role_guess": "candidate", "confidence": 0.85, "rationale": "Answering questions"}},
        {"structured": {"role_guess": "candidate", "confidence": 0.9, "rationale": "Technical discussion"}},
    ])
    extractor = SemanticSignalExtractor(context, llm_client=mock_client, batch_window_seconds=30)

    for i in range(6):
        extractor.add_transcript_segment(TranscriptSegment(
            participant_id="alice",
            text=f"This is a longer transcript segment number {i+1} with enough words to pass the minimum length filter.",
            start_time=datetime.utcnow() - timedelta(minutes=5-i),
            end_time=datetime.utcnow() - timedelta(minutes=4-i),
        ))

    evidence = extractor.extract_all()
    assert len(evidence) >= 1
    assert all(e.source == SignalSource.LLM_ROLE_CLASSIFIER for e in evidence)
    assert all(e.axis == SignalAxis.IDENTITY for e in evidence)
    assert all(e.target_participant_id == "alice" for e in evidence)


def test_semantic_extractor_low_confidence_filtered():
    context = MeetingContext(
        meeting_id="test",
        candidate_name="Alice",
    )
    participant = Participant(id="alice", display_name="Alice")
    context.participants = {"alice": participant}

    mock_client = MockLLMClient([
        {"structured": {"role_guess": "candidate", "confidence": 0.3, "rationale": "Uncertain"}}
    ])
    extractor = SemanticSignalExtractor(context, llm_client=mock_client)

    extractor.add_transcript_segment(TranscriptSegment(
        participant_id="alice",
        text="I have extensive experience with distributed systems and cloud architecture.",
        start_time=datetime.utcnow() - timedelta(minutes=1),
        end_time=datetime.utcnow(),
    ))

    evidence = extractor.extract_all()
    assert len(evidence) == 0


def test_semantic_extractor_invalid_role_filtered():
    context = MeetingContext(
        meeting_id="test",
        candidate_name="Alice",
    )
    participant = Participant(id="alice", display_name="Alice")
    context.participants = {"alice": participant}

    mock_client = MockLLMClient([
        {"structured": {"role_guess": "observer", "confidence": 0.8, "rationale": "test"}}
    ])
    extractor = SemanticSignalExtractor(context, llm_client=mock_client)

    extractor.add_transcript_segment(TranscriptSegment(
        participant_id="alice",
        text="I have extensive experience with distributed systems and cloud architecture.",
        start_time=datetime.utcnow() - timedelta(minutes=1),
        end_time=datetime.utcnow(),
    ))

    evidence = extractor.extract_all()
    assert len(evidence) == 0


def test_semantic_extractor_api_error_handling():
    context = MeetingContext(
        meeting_id="test",
        candidate_name="Alice",
    )
    participant = Participant(id="alice", display_name="Alice")
    context.participants = {"alice": participant}

    class FailingLLMClient:
        def generate_structured(self, *args, **kwargs):
            raise Exception("API rate limit exceeded")

    extractor = SemanticSignalExtractor(context, llm_client=FailingLLMClient())

    extractor.add_transcript_segment(TranscriptSegment(
        participant_id="alice",
        text="I have extensive experience with distributed systems and cloud architecture.",
        start_time=datetime.utcnow() - timedelta(minutes=1),
        end_time=datetime.utcnow(),
    ))

    evidence = extractor.extract_all()
    assert len(evidence) == 0

    stats = extractor.get_failure_stats()
    assert stats["failed_calls"] == 1


def test_semantic_extractor_evidence_packet_values():
    context = MeetingContext(
        meeting_id="test",
        candidate_name="Alice",
    )
    participant = Participant(id="alice", display_name="Alice")
    context.participants = {"alice": participant}

    mock_client = MockLLMClient([
        {"structured": {"role_guess": "candidate", "confidence": 0.8, "rationale": "Answering technical questions"}}
    ])
    extractor = SemanticSignalExtractor(context, llm_client=mock_client)

    extractor.add_transcript_segment(TranscriptSegment(
        participant_id="alice",
        text="I have extensive experience with distributed systems and cloud architecture.",
        start_time=datetime.utcnow() - timedelta(minutes=1),
        end_time=datetime.utcnow(),
    ))

    evidence = extractor.extract_all()
    assert len(evidence) == 1

    packet = evidence[0]
    assert packet.source == SignalSource.LLM_ROLE_CLASSIFIER
    assert packet.axis == SignalAxis.IDENTITY
    assert packet.target_participant_id == "alice"
    assert packet.delta_log_odds > 0
    assert packet.confidence == 0.8
    assert "candidate" in packet.rationale.lower()
    assert "batch_size" in packet.metadata
    assert packet.metadata["batch_size"] == 1


def test_semantic_extractor_interviewer_classification():
    context = MeetingContext(
        meeting_id="test",
        candidate_name="Alice",
    )
    participant = Participant(id="bob", display_name="Bob")
    context.participants = {"bob": participant}

    mock_client = MockLLMClient([
        {"structured": {"role_guess": "interviewer", "confidence": 0.85, "rationale": "Asking questions"}}
    ])
    extractor = SemanticSignalExtractor(context, llm_client=mock_client)

    extractor.add_transcript_segment(TranscriptSegment(
        participant_id="bob",
        text="Can you tell me about your experience with machine learning and how you've applied it in previous roles?",
        start_time=datetime.utcnow() - timedelta(minutes=1),
        end_time=datetime.utcnow(),
    ))

    evidence = extractor.extract_all()
    assert len(evidence) == 1

    packet = evidence[0]
    assert packet.delta_log_odds < 0
    assert packet.confidence == 0.85


def test_semantic_extractor_multiple_participants():
    context = MeetingContext(
        meeting_id="test",
        candidate_name="Alice",
        interviewer_names=["Bob"],
    )
    alice = Participant(id="alice", display_name="Alice")
    bob = Participant(id="bob", display_name="Bob")
    context.participants = {"alice": alice, "bob": bob}

    mock_client = MockLLMClient([
        {"structured": {"role_guess": "candidate", "confidence": 0.8, "rationale": "Answering"}},
        {"structured": {"role_guess": "interviewer", "confidence": 0.9, "rationale": "Questioning"}},
    ])
    extractor = SemanticSignalExtractor(context, llm_client=mock_client)

    extractor.add_transcript_segment(TranscriptSegment(
        participant_id="alice",
        text="I have extensive experience with distributed systems and cloud architecture.",
        start_time=datetime.utcnow() - timedelta(minutes=2),
        end_time=datetime.utcnow() - timedelta(minutes=1),
    ))

    extractor.add_transcript_segment(TranscriptSegment(
        participant_id="bob",
        text="Can you tell me about your experience with machine learning and how you've applied it?",
        start_time=datetime.utcnow() - timedelta(minutes=1),
        end_time=datetime.utcnow(),
    ))

    evidence = extractor.extract_all()
    assert len(evidence) == 2

    alice_evidence = [e for e in evidence if e.target_participant_id == "alice"]
    bob_evidence = [e for e in evidence if e.target_participant_id == "bob"]

    assert len(alice_evidence) == 1
    assert alice_evidence[0].delta_log_odds > 0

    assert len(bob_evidence) == 1
    assert bob_evidence[0].delta_log_odds < 0


if __name__ == "__main__":
    test_logit_conversion_candidate_high_confidence()
    test_logit_conversion_interviewer_high_confidence()
    test_logit_conversion_50_percent()
    test_logit_conversion_clamps_low_confidence()
    test_logit_conversion_clamps_high_confidence()
    test_logit_conversion_symmetric()
    test_evidence_packet_schema()
    test_semantic_extractor_disabled()
    test_semantic_extractor_no_client()
    test_semantic_extractor_short_segments_filtered()
    test_semantic_extractor_batching()
    test_semantic_extractor_low_confidence_filtered()
    test_semantic_extractor_invalid_role_filtered()
    test_semantic_extractor_api_error_handling()
    test_semantic_extractor_evidence_packet_values()
    test_semantic_extractor_interviewer_classification()
    test_semantic_extractor_multiple_participants()
    print("All semantic signal tests passed!")
