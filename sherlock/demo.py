from __future__ import annotations

import random
from datetime import datetime, timedelta

from sherlock import FusionEngine, ExplanationLayer, FeedbackLoop
from sherlock.models import (
    CodingEvent,
    GazeEvent,
    MeetingContext,
    Participant,
    SpeakingEvent,
    TranscriptSegment,
)
from sherlock.signals import (
    AuthenticitySignalExtractor,
    BehavioralSignalExtractor,
    IdentitySignalExtractor,
    SemanticSignalExtractor,
)


def create_scenario_1() -> MeetingContext:
    print("\n" + "=" * 80)
    print("SCENARIO 1: Normal Interview")
    print("Candidate: Alice Johnson (alice.johnson@email.com)")
    print("Interviewer: Bob Smith")
    print("=" * 80)

    context = MeetingContext(
        meeting_id="meeting-001",
        candidate_name="Alice Johnson",
        candidate_email="alice.johnson@email.com",
        interviewer_names=["Bob Smith"],
        interviewer_emails=["bob.smith@company.com"],
        scheduled_start=datetime.utcnow() - timedelta(minutes=5),
        scheduled_end=datetime.utcnow() + timedelta(minutes=55),
    )

    alice = Participant(
        id="alice-001",
        display_name="Alice Johnson",
        email="alice.johnson@email.com",
        join_time=datetime.utcnow() - timedelta(minutes=4),
        webcam_on=True,
    )

    bob = Participant(
        id="bob-001",
        display_name="Bob Smith",
        email="bob.smith@company.com",
        join_time=datetime.utcnow() - timedelta(minutes=5),
        webcam_on=True,
    )

    context.participants = {"alice-001": alice, "bob-001": bob}
    return context


def create_scenario_2() -> MeetingContext:
    print("\n" + "=" * 80)
    print("SCENARIO 2: Candidate Joins as Device Name")
    print("Candidate: Charlie Brown (joins as 'MacBook Pro')")
    print("Interviewer: Diana Prince")
    print("=" * 80)

    context = MeetingContext(
        meeting_id="meeting-002",
        candidate_name="Charlie Brown",
        candidate_email="charlie.brown@email.com",
        interviewer_names=["Diana Prince"],
        interviewer_emails=["diana.prince@company.com"],
        scheduled_start=datetime.utcnow() - timedelta(minutes=3),
        scheduled_end=datetime.utcnow() + timedelta(minutes=57),
    )

    charlie = Participant(
        id="charlie-001",
        display_name="MacBook Pro",
        email="charlie.brown@email.com",
        join_time=datetime.utcnow() - timedelta(minutes=2),
        webcam_on=True,
        device_name="MacBook Pro",
    )

    diana = Participant(
        id="diana-001",
        display_name="Diana Prince",
        email="diana.prince@company.com",
        join_time=datetime.utcnow() - timedelta(minutes=3),
        webcam_on=True,
    )

    context.participants = {"charlie-001": charlie, "diana-001": diana}
    return context


def create_scenario_3() -> MeetingContext:
    print("\n" + "=" * 80)
    print("SCENARIO 3: Multiple Interviewers + Silent Observer")
    print("Candidate: Eve Davis")
    print("Interviewers: Frank Miller, Grace Lee")
    print("Silent Observer: Henry Wilson")
    print("=" * 80)

    context = MeetingContext(
        meeting_id="meeting-003",
        candidate_name="Eve Davis",
        candidate_email="eve.davis@email.com",
        interviewer_names=["Frank Miller", "Grace Lee"],
        interviewer_emails=["frank.miller@company.com", "grace.lee@company.com"],
        scheduled_start=datetime.utcnow() - timedelta(minutes=10),
        scheduled_end=datetime.utcnow() + timedelta(minutes=50),
    )

    eve = Participant(
        id="eve-001",
        display_name="Eve Davis",
        email="eve.davis@email.com",
        join_time=datetime.utcnow() - timedelta(minutes=9),
        webcam_on=True,
    )

    frank = Participant(
        id="frank-001",
        display_name="Frank Miller",
        email="frank.miller@company.com",
        join_time=datetime.utcnow() - timedelta(minutes=10),
        webcam_on=True,
    )

    grace = Participant(
        id="grace-001",
        display_name="Grace Lee",
        email="grace.lee@company.com",
        join_time=datetime.utcnow() - timedelta(minutes=10),
        webcam_on=False,
    )

    henry = Participant(
        id="henry-001",
        display_name="Henry Wilson",
        email="henry.wilson@company.com",
        join_time=datetime.utcnow() - timedelta(minutes=8),
        webcam_on=False,
    )

    context.participants = {
        "eve-001": eve,
        "frank-001": frank,
        "grace-001": grace,
        "henry-001": henry,
    }
    return context


def create_scenario_4() -> MeetingContext:
    print("\n" + "=" * 80)
    print("SCENARIO 4: Candidate Changes Display Name Mid-Call")
    print("Candidate: Ivan Petrov (changes name to 'Ivan P.')")
    print("Interviewer: Julia Chen")
    print("=" * 80)

    context = MeetingContext(
        meeting_id="meeting-004",
        candidate_name="Ivan Petrov",
        candidate_email="ivan.petrov@email.com",
        interviewer_names=["Julia Chen"],
        interviewer_emails=["julia.chen@company.com"],
        scheduled_start=datetime.utcnow() - timedelta(minutes=15),
        scheduled_end=datetime.utcnow() + timedelta(minutes=45),
    )

    ivan = Participant(
        id="ivan-001",
        display_name="Ivan P.",
        email="ivan.petrov@email.com",
        join_time=datetime.utcnow() - timedelta(minutes=14),
        webcam_on=True,
        display_name_history=[
            {"old_name": "Ivan Petrov", "new_name": "Ivan P.", "timestamp": datetime.utcnow() - timedelta(minutes=10)},
        ],
    )

    julia = Participant(
        id="julia-001",
        display_name="Julia Chen",
        email="julia.chen@company.com",
        join_time=datetime.utcnow() - timedelta(minutes=15),
        webcam_on=True,
    )

    context.participants = {"ivan-001": ivan, "julia-001": julia}
    return context


def run_scenario(context: MeetingContext, scenario_name: str) -> None:
    engine = FusionEngine(context)
    explanation = ExplanationLayer(engine)

    identity_extractor = IdentitySignalExtractor(context)
    behavioral_extractor = BehavioralSignalExtractor(context)

    identity_evidence = identity_extractor.extract_all()
    engine.ingest_batch(identity_evidence)

    print(f"\nAfter identity signals:")
    result = engine.get_result()
    print(f"Status: {result.status.upper()}")
    if result.top_candidate_id:
        participant = context.participants.get(result.top_candidate_id)
        name = participant.display_name if participant else result.top_candidate_id
        print(f"Top candidate: {name} ({result.top_candidate_probability:.1%})")
    print(f"Ambiguity gap: {result.ambiguity_gap:.1%}")

    if scenario_name == "scenario-1":
        for i in range(5):
            behavioral_extractor.add_transcript_segment(TranscriptSegment(
                participant_id="bob-001",
                text=f"Question {i+1}: Can you tell me about your experience with Python?",
                start_time=datetime.utcnow() - timedelta(minutes=4, seconds=30-i*30),
                end_time=datetime.utcnow() - timedelta(minutes=4, seconds=25-i*30),
                is_question=True,
            ))
            behavioral_extractor.add_transcript_segment(TranscriptSegment(
                participant_id="alice-001",
                text=f"Yes, I've been working with Python for 5 years. I've used it for data analysis, web development, and machine learning projects.",
                start_time=datetime.utcnow() - timedelta(minutes=4, seconds=20-i*30),
                end_time=datetime.utcnow() - timedelta(minutes=3, seconds=50-i*30),
                is_question=False,
            ))

        behavioral_extractor.add_speaking_event(SpeakingEvent(
            participant_id="alice-001",
            start_time=datetime.utcnow() - timedelta(minutes=3),
            end_time=datetime.utcnow() - timedelta(minutes=2),
            is_response_to_question=True,
        ))

        behavioral_extractor.add_speaking_event(SpeakingEvent(
            participant_id="bob-001",
            start_time=datetime.utcnow() - timedelta(minutes=4),
            end_time=datetime.utcnow() - timedelta(minutes=3, seconds=30),
            is_response_to_question=False,
        ))

    elif scenario_name == "scenario-2":
        for i in range(3):
            behavioral_extractor.add_transcript_segment(TranscriptSegment(
                participant_id="diana-001",
                text=f"Let's discuss your background in machine learning.",
                start_time=datetime.utcnow() - timedelta(minutes=2, seconds=30-i*20),
                end_time=datetime.utcnow() - timedelta(minutes=2, seconds=25-i*20),
                is_question=True,
            ))
            behavioral_extractor.add_transcript_segment(TranscriptSegment(
                participant_id="charlie-001",
                text=f"I've worked on several ML projects including recommendation systems and NLP applications.",
                start_time=datetime.utcnow() - timedelta(minutes=2, seconds=20-i*20),
                end_time=datetime.utcnow() - timedelta(minutes=2, seconds=0-i*20),
                is_question=False,
            ))

        behavioral_extractor.add_speaking_event(SpeakingEvent(
            participant_id="charlie-001",
            start_time=datetime.utcnow() - timedelta(minutes=1),
            end_time=datetime.utcnow(),
            is_response_to_question=True,
        ))

    elif scenario_name == "scenario-3":
        for i in range(4):
            behavioral_extractor.add_transcript_segment(TranscriptSegment(
                participant_id="frank-001",
                text=f"Tell me about a challenging project you've worked on.",
                start_time=datetime.utcnow() - timedelta(minutes=8, seconds=30-i*25),
                end_time=datetime.utcnow() - timedelta(minutes=8, seconds=25-i*25),
                is_question=True,
            ))
            behavioral_extractor.add_transcript_segment(TranscriptSegment(
                participant_id="eve-001",
                text=f"I led a team of 5 engineers to build a real-time data pipeline that processed 10M events per day.",
                start_time=datetime.utcnow() - timedelta(minutes=8, seconds=20-i*25),
                end_time=datetime.utcnow() - timedelta(minutes=7, seconds=50-i*25),
                is_question=False,
            ))

        behavioral_extractor.add_speaking_event(SpeakingEvent(
            participant_id="eve-001",
            start_time=datetime.utcnow() - timedelta(minutes=5),
            end_time=datetime.utcnow() - timedelta(minutes=3),
            is_response_to_question=True,
        ))

    elif scenario_name == "scenario-4":
        for i in range(3):
            behavioral_extractor.add_transcript_segment(TranscriptSegment(
                participant_id="julia-001",
                text=f"What's your experience with distributed systems?",
                start_time=datetime.utcnow() - timedelta(minutes=12, seconds=30-i*30),
                end_time=datetime.utcnow() - timedelta(minutes=12, seconds=25-i*30),
                is_question=True,
            ))
            behavioral_extractor.add_transcript_segment(TranscriptSegment(
                participant_id="ivan-001",
                text=f"I've built microservices architectures handling millions of requests per day using Kubernetes and Kafka.",
                start_time=datetime.utcnow() - timedelta(minutes=12, seconds=20-i*30),
                end_time=datetime.utcnow() - timedelta(minutes=11, seconds=50-i*30),
                is_question=False,
            ))

        behavioral_extractor.add_speaking_event(SpeakingEvent(
            participant_id="ivan-001",
            start_time=datetime.utcnow() - timedelta(minutes=8),
            end_time=datetime.utcnow() - timedelta(minutes=6),
            is_response_to_question=True,
        ))

    behavioral_evidence = behavioral_extractor.extract_all()
    engine.ingest_batch(behavioral_evidence)

    print(f"\nAfter behavioral signals:")
    result = engine.get_result()
    print(f"Status: {result.status.upper()}")
    if result.top_candidate_id:
        participant = context.participants.get(result.top_candidate_id)
        name = participant.display_name if participant else result.top_candidate_id
        print(f"Top candidate: {name} ({result.top_candidate_probability:.1%})")
    print(f"Ambiguity gap: {result.ambiguity_gap:.1%}")

    print("\n" + "=" * 80)
    print("FULL REPORT:")
    print("=" * 80)
    print(explanation.get_full_report())

    if result.top_candidate_id:
        print("\n" + "=" * 80)
        print("DETAILED IDENTITY EXPLANATION:")
        print("=" * 80)
        print(explanation.explain_identity(result.top_candidate_id))

    print("\n" + "=" * 80)
    print("BELIEF TIMELINE:")
    print("=" * 80)
    print(explanation.explain_timeline())

    return engine, explanation


def demo_authenticity_detection() -> None:
    print("\n\n" + "=" * 80)
    print("AUTHENTICITY DETECTION DEMO")
    print("=" * 80)

    context = create_scenario_1()
    engine = FusionEngine(context)
    explanation = ExplanationLayer(engine)

    identity_extractor = IdentitySignalExtractor(context)
    behavioral_extractor = BehavioralSignalExtractor(context)
    authenticity_extractor = AuthenticitySignalExtractor(context)

    identity_evidence = identity_extractor.extract_all()
    engine.ingest_batch(identity_evidence)

    behavioral_extractor.add_transcript_segment(TranscriptSegment(
        participant_id="bob-001",
        text="Can you explain how backpropagation works?",
        start_time=datetime.utcnow() - timedelta(minutes=3),
        end_time=datetime.utcnow() - timedelta(minutes=2, seconds=55),
        is_question=True,
    ))

    behavioral_extractor.add_transcript_segment(TranscriptSegment(
        participant_id="alice-001",
        text="Um, so backpropagation is basically the algorithm used to, you know, compute gradients in neural networks. It works by, like, applying the chain rule from calculus backwards through the layers.",
        start_time=datetime.utcnow() - timedelta(minutes=2, seconds=50),
        end_time=datetime.utcnow() - timedelta(minutes=2, seconds=30),
        is_question=False,
    ))

    for i in range(5):
        behavioral_extractor.add_transcript_segment(TranscriptSegment(
            participant_id="alice-001",
            text=f"Answer {i+1} with natural filler words and self-corrections, um, like this.",
            start_time=datetime.utcnow() - timedelta(minutes=2-i, seconds=30),
            end_time=datetime.utcnow() - timedelta(minutes=2-i, seconds=10),
            is_question=False,
        ))

    behavioral_evidence = behavioral_extractor.extract_all()
    engine.ingest_batch(behavioral_evidence)

    for i in range(5):
        authenticity_extractor.add_transcript_segment(TranscriptSegment(
            participant_id="alice-001",
            text=f"Answer {i+1} with natural filler words and self-corrections, um, like this.",
            start_time=datetime.utcnow() - timedelta(minutes=2-i, seconds=30),
            end_time=datetime.utcnow() - timedelta(minutes=2-i, seconds=10),
            is_question=False,
        ))

    authenticity_extractor.add_coding_event(CodingEvent(
        participant_id="alice-001",
        timestamp=datetime.utcnow() - timedelta(minutes=1),
        event_type="paste",
        content_length=200,
    ))

    authenticity_extractor.add_gaze_event(GazeEvent(
        participant_id="alice-001",
        timestamp=datetime.utcnow() - timedelta(seconds=30),
        gaze_vector=(0.5, -0.3, 0.8),
        is_off_screen=True,
        saccade_periodicity=1.2,
    ))

    authenticity_evidence = authenticity_extractor.extract_all()
    engine.ingest_batch(authenticity_evidence)

    print("\n" + "=" * 80)
    print("AUTHENTICITY REPORT FOR ALICE:")
    print("=" * 80)
    print(explanation.explain_authenticity("alice-001"))


def demo_feedback_loop() -> None:
    print("\n\n" + "=" * 80)
    print("FEEDBACK LOOP DEMO")
    print("=" * 80)

    context = create_scenario_2()
    engine = FusionEngine(context)
    explanation = ExplanationLayer(engine)
    feedback_loop = FeedbackLoop(engine)

    identity_extractor = IdentitySignalExtractor(context)
    behavioral_extractor = BehavioralSignalExtractor(context)

    identity_evidence = identity_extractor.extract_all()
    engine.ingest_batch(identity_evidence)

    behavioral_extractor.add_transcript_segment(TranscriptSegment(
        participant_id="diana-001",
        text="Tell me about your experience.",
        start_time=datetime.utcnow() - timedelta(minutes=2),
        end_time=datetime.utcnow() - timedelta(minutes=1, seconds=55),
        is_question=True,
    ))
    behavioral_extractor.add_transcript_segment(TranscriptSegment(
        participant_id="charlie-001",
        text="I have 5 years of experience in software engineering.",
        start_time=datetime.utcnow() - timedelta(minutes=1, seconds=50),
        end_time=datetime.utcnow() - timedelta(minutes=1, seconds=30),
        is_question=False,
    ))

    behavioral_evidence = behavioral_extractor.extract_all()
    engine.ingest_batch(behavioral_evidence)

    print("\nBefore feedback:")
    result = engine.get_result()
    if result.top_candidate_id:
        participant = context.participants.get(result.top_candidate_id)
        name = participant.display_name if participant else result.top_candidate_id
        print(f"Top candidate: {name} ({result.top_candidate_probability:.1%})")

    print("\nInterviewer confirms: 'Yes, MacBook Pro is actually Charlie Brown'")
    feedback_loop.record_confirmation("charlie-001", notes="Interviewer confirmed identity")

    print("\nAfter feedback:")
    print(feedback_loop.get_feedback_summary())


def main():
    print("\n" + "=" * 80)
    print("SHERLOCK CANDIDATE IDENTIFICATION ENGINE - DEMO")
    print("=" * 80)

    scenarios = [
        ("scenario-1", create_scenario_1),
        ("scenario-2", create_scenario_2),
        ("scenario-3", create_scenario_3),
        ("scenario-4", create_scenario_4),
    ]

    for scenario_name, scenario_fn in scenarios:
        context = scenario_fn()
        run_scenario(context, scenario_name)

    demo_authenticity_detection()
    demo_feedback_loop()

    print("\n\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
