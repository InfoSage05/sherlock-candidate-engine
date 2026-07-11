"""Tests for the candidate stream gate (Prompt 10.3.2)."""

from datetime import datetime

from sherlock.gate import CandidateStreamGate
from sherlock.models import EvidencePacket, RawMediaFrame, SignalAxis, SignalSource
from sherlock.tests._helpers import make_context, make_identified_engine


def test_gate_passes_only_candidate_frames():
    _, engine = make_identified_engine(["alice", "bob"], "alice", "bob")
    gate = CandidateStreamGate(engine)

    cand = RawMediaFrame(participant_id="alice", video_frame=None, timestamp_ms=1000)
    other = RawMediaFrame(participant_id="bob", video_frame=None, timestamp_ms=1000)

    out_cand = gate.process(cand)
    assert out_cand is not None
    assert out_cand.candidate_id == "alice"

    out_other = gate.process(other)
    assert out_other is None
    assert gate.dropped_frames_non_candidate == 1


def test_gate_emits_warning_when_identity_ambiguous():
    # Fresh engine with no evidence -> status "ambiguous".
    context = make_context(["alice", "bob"])
    from sherlock.fusion import FusionEngine

    engine = FusionEngine(context)
    gate = CandidateStreamGate(engine)

    out = gate.process(RawMediaFrame(participant_id="alice", timestamp_ms=0))
    assert out is None
    flags = gate.consume_flags()
    assert len(flags) == 1
    assert flags[0].severity.value == "warning"
    assert flags[0].source == SignalSource.IDENTITY_UNCERTAIN


def test_gate_switches_candidate_mid_call():
    context, engine = make_identified_engine(["alice", "bob"], "alice", "bob")
    gate = CandidateStreamGate(engine)

    # Initially alice is the candidate.
    assert gate.process(RawMediaFrame(participant_id="alice", timestamp_ms=0)) is not None

    # Operator corrects: bob is actually the candidate.
    engine.ingest(EvidencePacket(
        source=SignalSource.CALENDAR_MATCH, axis=SignalAxis.IDENTITY,
        target_participant_id="bob", delta_log_odds=3.0, confidence=0.9,
        rationale="operator correction", timestamp=datetime.utcnow(),
    ))

    out_bob = gate.process(RawMediaFrame(participant_id="bob", timestamp_ms=10))
    assert out_bob is not None
    assert out_bob.candidate_id == "bob"
    flags = gate.consume_flags()
    assert any(f.source == SignalSource.JOIN_TIMING for f in flags)
