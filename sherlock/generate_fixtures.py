"""
Generate JSON fixtures for all edge-case scenarios.

Run this script to regenerate all fixtures:
    python -m sherlock.generate_fixtures
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from sherlock.models import (
    EvidencePacket,
    MeetingContext,
    Participant,
    SignalAxis,
    SignalSource,
    TranscriptSegment,
)
from sherlock.signals.identity import IdentitySignalExtractor
from sherlock.signals.behavioral import BehavioralSignalExtractor
from sherlock.session_replay import SessionFixture


def generate_normal_interview() -> SessionFixture:
    """Scenario: Normal interview with clear candidate identification."""
    base_time = datetime(2024, 1, 15, 10, 0, 0)
    
    context = MeetingContext(
        meeting_id="normal-interview",
        candidate_name="Alice Johnson",
        candidate_email="alice.johnson@example.com",
        interviewer_names=["Bob Smith"],
        interviewer_emails=["bob.smith@company.com"],
        scheduled_start=base_time,
        participants={
            "alice": Participant(
                id="alice",
                display_name="Alice Johnson",
                email="alice.johnson@example.com",
                join_time=base_time - timedelta(seconds=30),
                webcam_on=True,
            ),
            "bob": Participant(
                id="bob",
                display_name="Bob Smith",
                email="bob.smith@company.com",
                join_time=base_time - timedelta(seconds=60),
                webcam_on=True,
            ),
        },
    )
    
    evidence_packets = []
    transcript_segments = []
    
    # Identity signals
    identity_extractor = IdentitySignalExtractor(context)
    evidence_packets.extend(identity_extractor.extract_all())
    
    # Transcript and behavioral signals
    behavioral_extractor = BehavioralSignalExtractor(context)
    
    # Interview Q&A
    qa_pairs = [
        ("bob", "Can you tell me about your experience with machine learning?", True),
        ("alice", "I've been working with ML for about 5 years. At my last company, I led a team that built a recommendation system that improved user engagement by 30%.", False),
        ("bob", "That's impressive. What frameworks do you typically use?", True),
        ("alice", "I primarily use PyTorch for deep learning work, and scikit-learn for traditional ML. I'm also familiar with TensorFlow from my earlier projects.", False),
        ("bob", "How do you handle overfitting in your models?", True),
        ("alice", "I use several techniques: cross-validation, regularization, dropout for neural networks, and early stopping. I also make sure to have a proper train-validation-test split.", False),
    ]
    
    current_time = base_time + timedelta(seconds=60)
    for speaker, text, is_question in qa_pairs:
        segment = TranscriptSegment(
            participant_id=speaker,
            text=text,
            start_time=current_time,
            end_time=current_time + timedelta(seconds=8),
            is_question=is_question,
        )
        transcript_segments.append(segment)
        behavioral_extractor.add_transcript_segment(segment)
        current_time += timedelta(seconds=15)
    
    # Extract behavioral signals
    evidence_packets.extend(behavioral_extractor.extract_all())
    
    # Sort by timestamp
    evidence_packets.sort(key=lambda e: e.timestamp)
    
    fixture = SessionFixture(
        meeting_context=context,
        evidence_packets=evidence_packets,
        transcript_segments=transcript_segments,
        metadata={
            "name": "Normal Interview",
            "description": "Standard interview with clear candidate identification",
            "difficulty": "easy",
        },
    )
    
    return normalize_timestamps(fixture, base_time)


def generate_device_name_scenario() -> SessionFixture:
    """Scenario: Candidate joins as 'MacBook Pro' instead of their name."""
    base_time = datetime(2024, 1, 15, 11, 0, 0)
    
    context = MeetingContext(
        meeting_id="device-name",
        candidate_name="Charlie Brown",
        candidate_email="charlie.brown@example.com",
        interviewer_names=["Diana Prince"],
        interviewer_emails=["diana.prince@company.com"],
        scheduled_start=base_time,
        participants={
            "charlie": Participant(
                id="charlie",
                display_name="MacBook Pro",
                email="charlie.brown@example.com",
                join_time=base_time - timedelta(seconds=20),
                webcam_on=True,
                device_name="MacBook Pro",
            ),
            "diana": Participant(
                id="diana",
                display_name="Diana Prince",
                email="diana.prince@company.com",
                join_time=base_time - timedelta(seconds=60),
                webcam_on=True,
            ),
        },
    )
    
    evidence_packets = []
    transcript_segments = []
    
    # Identity signals
    identity_extractor = IdentitySignalExtractor(context)
    evidence_packets.extend(identity_extractor.extract_all())
    
    # Transcript
    behavioral_extractor = BehavioralSignalExtractor(context)
    
    qa_pairs = [
        ("diana", "Welcome! I see you're joining from a MacBook. Let's start with your background.", True),
        ("charlie", "Thanks! I'm a software engineer with 7 years of experience, specializing in backend systems and distributed computing.", False),
        ("diana", "Can you walk me through a challenging project you've worked on?", True),
        ("charlie", "Sure. At my current role, I architected a microservices platform that handles 10 million requests per day. The main challenge was ensuring consistency across services while maintaining high availability.", False),
    ]
    
    current_time = base_time + timedelta(seconds=60)
    for speaker, text, is_question in qa_pairs:
        segment = TranscriptSegment(
            participant_id=speaker,
            text=text,
            start_time=current_time,
            end_time=current_time + timedelta(seconds=10),
            is_question=is_question,
        )
        transcript_segments.append(segment)
        behavioral_extractor.add_transcript_segment(segment)
        current_time += timedelta(seconds=20)
    
    evidence_packets.extend(behavioral_extractor.extract_all())
    evidence_packets.sort(key=lambda e: e.timestamp)
    
    return SessionFixture(
        meeting_context=context,
        evidence_packets=evidence_packets,
        transcript_segments=transcript_segments,
        metadata={
            "name": "Device Name Join",
            "description": "Candidate joins as 'MacBook Pro' instead of their real name",
            "difficulty": "medium",
        },
    )


def generate_multiple_interviewers_scenario() -> SessionFixture:
    """Scenario: Multiple interviewers and a silent observer."""
    base_time = datetime(2024, 1, 15, 14, 0, 0)
    
    context = MeetingContext(
        meeting_id="multiple-interviewers",
        candidate_name="Eve Davis",
        candidate_email="eve.davis@example.com",
        interviewer_names=["Frank Miller", "Grace Lee"],
        interviewer_emails=["frank.miller@company.com", "grace.lee@company.com"],
        scheduled_start=base_time,
        participants={
            "eve": Participant(
                id="eve",
                display_name="Eve Davis",
                email="eve.davis@example.com",
                join_time=base_time - timedelta(seconds=30),
                webcam_on=True,
            ),
            "frank": Participant(
                id="frank",
                display_name="Frank Miller",
                email="frank.miller@company.com",
                join_time=base_time - timedelta(seconds=60),
                webcam_on=True,
            ),
            "grace": Participant(
                id="grace",
                display_name="Grace Lee",
                email="grace.lee@company.com",
                join_time=base_time - timedelta(seconds=60),
                webcam_on=True,
            ),
            "henry": Participant(
                id="henry",
                display_name="Henry Wilson",
                email="henry.wilson@company.com",
                join_time=base_time - timedelta(seconds=45),
                webcam_on=False,
            ),
        },
    )
    
    evidence_packets = []
    transcript_segments = []
    
    identity_extractor = IdentitySignalExtractor(context)
    evidence_packets.extend(identity_extractor.extract_all())
    
    behavioral_extractor = BehavioralSignalExtractor(context)
    
    qa_pairs = [
        ("frank", "Let's start with your technical background. What's your experience with cloud architecture?", True),
        ("eve", "I've been working with AWS for the past 4 years. I'm certified as a Solutions Architect Professional and have designed several large-scale distributed systems.", False),
        ("grace", "That's great. Can you give me an example of how you've handled a production incident?", True),
        ("eve", "Last month, we had a critical database outage during peak traffic. I led the incident response, coordinated with the DBA team, and we restored service within 15 minutes. Afterward, I implemented better monitoring and failover mechanisms.", False),
        ("frank", "Excellent. How do you approach system design for high availability?", True),
        ("eve", "I focus on redundancy, circuit breakers, and graceful degradation. I also believe in chaos engineering - regularly testing failure scenarios in staging environments.", False),
    ]
    
    current_time = base_time + timedelta(seconds=60)
    for speaker, text, is_question in qa_pairs:
        segment = TranscriptSegment(
            participant_id=speaker,
            text=text,
            start_time=current_time,
            end_time=current_time + timedelta(seconds=10),
            is_question=is_question,
        )
        transcript_segments.append(segment)
        behavioral_extractor.add_transcript_segment(segment)
        current_time += timedelta(seconds=18)
    
    evidence_packets.extend(behavioral_extractor.extract_all())
    evidence_packets.sort(key=lambda e: e.timestamp)
    
    return SessionFixture(
        meeting_context=context,
        evidence_packets=evidence_packets,
        transcript_segments=transcript_segments,
        metadata={
            "name": "Multiple Interviewers + Silent Observer",
            "description": "Two interviewers, one candidate, and a silent observer",
            "difficulty": "hard",
        },
    )


def generate_nickname_scenario() -> SessionFixture:
    """Scenario: Candidate joins under a nickname."""
    base_time = datetime(2024, 1, 15, 15, 0, 0)
    
    context = MeetingContext(
        meeting_id="nickname",
        candidate_name="Alexander Thompson",
        candidate_email="alex.thompson@example.com",
        interviewer_names=["Sarah Connor"],
        interviewer_emails=["sarah.connor@company.com"],
        scheduled_start=base_time,
        participants={
            "alex": Participant(
                id="alex",
                display_name="Alex T.",
                email="alex.thompson@example.com",
                join_time=base_time - timedelta(seconds=25),
                webcam_on=True,
            ),
            "sarah": Participant(
                id="sarah",
                display_name="Sarah Connor",
                email="sarah.connor@company.com",
                join_time=base_time - timedelta(seconds=60),
                webcam_on=True,
            ),
        },
    )
    
    evidence_packets = []
    transcript_segments = []
    
    identity_extractor = IdentitySignalExtractor(context)
    evidence_packets.extend(identity_extractor.extract_all())
    
    behavioral_extractor = BehavioralSignalExtractor(context)
    
    qa_pairs = [
        ("sarah", "Hi Alex! Thanks for joining. Let's talk about your frontend experience.", True),
        ("alex", "Sure! I've been doing frontend development for 6 years, mostly with React and TypeScript. I'm passionate about performance optimization and accessibility.", False),
        ("sarah", "How do you approach performance optimization in React applications?", True),
        ("alex", "I start with profiling using React DevTools to identify bottlenecks. Then I focus on code splitting, memoization with useMemo and useCallback, and virtualizing long lists. I also pay attention to bundle size and lazy loading.", False),
    ]
    
    current_time = base_time + timedelta(seconds=60)
    for speaker, text, is_question in qa_pairs:
        segment = TranscriptSegment(
            participant_id=speaker,
            text=text,
            start_time=current_time,
            end_time=current_time + timedelta(seconds=10),
            is_question=is_question,
        )
        transcript_segments.append(segment)
        behavioral_extractor.add_transcript_segment(segment)
        current_time += timedelta(seconds=20)
    
    evidence_packets.extend(behavioral_extractor.extract_all())
    evidence_packets.sort(key=lambda e: e.timestamp)
    
    return SessionFixture(
        meeting_context=context,
        evidence_packets=evidence_packets,
        transcript_segments=transcript_segments,
        metadata={
            "name": "Nickname Join",
            "description": "Candidate joins as 'Alex T.' but calendar says 'Alexander Thompson'",
            "difficulty": "medium",
        },
    )


def generate_wrong_name_scenario() -> SessionFixture:
    """Scenario: Interviewer enters wrong candidate name in calendar."""
    base_time = datetime(2024, 1, 15, 16, 0, 0)
    
    context = MeetingContext(
        meeting_id="wrong-name",
        candidate_name="Jessica Williams",  # Wrong name in calendar
        candidate_email="maria.garcia@example.com",  # But correct email
        interviewer_names=["Tom Anderson"],
        interviewer_emails=["tom.anderson@company.com"],
        scheduled_start=base_time,
        participants={
            "maria": Participant(
                id="maria",
                display_name="Maria Garcia",
                email="maria.garcia@example.com",
                join_time=base_time - timedelta(seconds=30),
                webcam_on=True,
            ),
            "tom": Participant(
                id="tom",
                display_name="Tom Anderson",
                email="tom.anderson@company.com",
                join_time=base_time - timedelta(seconds=60),
                webcam_on=True,
            ),
        },
    )
    
    evidence_packets = []
    transcript_segments = []
    
    identity_extractor = IdentitySignalExtractor(context)
    evidence_packets.extend(identity_extractor.extract_all())
    
    behavioral_extractor = BehavioralSignalExtractor(context)
    
    qa_pairs = [
        ("tom", "Hi Jessica! Welcome to the interview.", True),
        ("maria", "Actually, my name is Maria. There might have been a mix-up with the calendar invite.", False),
        ("tom", "Oh, I apologize Maria! Let me correct that. Let's get started with your background.", True),
        ("maria", "No problem! I'm a data scientist with 8 years of experience. I specialize in NLP and have built several production ML systems.", False),
    ]
    
    current_time = base_time + timedelta(seconds=60)
    for speaker, text, is_question in qa_pairs:
        segment = TranscriptSegment(
            participant_id=speaker,
            text=text,
            start_time=current_time,
            end_time=current_time + timedelta(seconds=10),
            is_question=is_question,
        )
        transcript_segments.append(segment)
        behavioral_extractor.add_transcript_segment(segment)
        current_time += timedelta(seconds=18)
    
    evidence_packets.extend(behavioral_extractor.extract_all())
    evidence_packets.sort(key=lambda e: e.timestamp)
    
    return SessionFixture(
        meeting_context=context,
        evidence_packets=evidence_packets,
        transcript_segments=transcript_segments,
        metadata={
            "name": "Wrong Name in Calendar",
            "description": "Calendar has wrong name but correct email - system must use email match",
            "difficulty": "hard",
        },
    )


def generate_display_name_change_scenario() -> SessionFixture:
    """Scenario: Candidate changes display name mid-call."""
    base_time = datetime(2024, 1, 15, 17, 0, 0)
    
    context = MeetingContext(
        meeting_id="name-change",
        candidate_name="David Kim",
        candidate_email="david.kim@example.com",
        interviewer_names=["Lisa Chen"],
        interviewer_emails=["lisa.chen@company.com"],
        scheduled_start=base_time,
        participants={
            "david": Participant(
                id="david",
                display_name="David Kim",
                email="david.kim@example.com",
                join_time=base_time - timedelta(seconds=30),
                webcam_on=True,
                display_name_history=[
                    {"old_name": "David Kim", "new_name": "Dave K.", "timestamp": base_time + timedelta(seconds=300)},
                ],
            ),
            "lisa": Participant(
                id="lisa",
                display_name="Lisa Chen",
                email="lisa.chen@company.com",
                join_time=base_time - timedelta(seconds=60),
                webcam_on=True,
            ),
        },
    )
    
    evidence_packets = []
    transcript_segments = []
    
    identity_extractor = IdentitySignalExtractor(context)
    evidence_packets.extend(identity_extractor.extract_all())
    
    behavioral_extractor = BehavioralSignalExtractor(context)
    
    # Early part of interview
    qa_pairs_early = [
        ("lisa", "Let's discuss your backend experience.", True),
        ("david", "I've been working with Java and Spring Boot for 5 years. I've built several REST APIs and microservices.", False),
    ]
    
    current_time = base_time + timedelta(seconds=60)
    for speaker, text, is_question in qa_pairs_early:
        segment = TranscriptSegment(
            participant_id=speaker,
            text=text,
            start_time=current_time,
            end_time=current_time + timedelta(seconds=8),
            is_question=is_question,
        )
        transcript_segments.append(segment)
        behavioral_extractor.add_transcript_segment(segment)
        current_time += timedelta(seconds=15)
    
    # Name change happens here
    name_change_time = base_time + timedelta(seconds=300)
    evidence_packets.append(EvidencePacket(
        source=SignalSource.DISPLAY_NAME_CHANGE,
        axis=SignalAxis.IDENTITY,
        target_participant_id="david",
        delta_log_odds=0.5,
        confidence=0.7,
        rationale="Display name changed from 'David Kim' to 'Dave K.'",
        timestamp=name_change_time,
        metadata={"old_name": "David Kim", "new_name": "Dave K."},
    ))
    
    # Later part of interview (after name change)
    qa_pairs_late = [
        ("lisa", "How do you handle database migrations in production?", True),
        ("david", "I use tools like Flyway for schema versioning. I always test migrations in staging first and have rollback scripts ready.", False),
    ]
    
    current_time = base_time + timedelta(seconds=360)
    for speaker, text, is_question in qa_pairs_late:
        segment = TranscriptSegment(
            participant_id=speaker,
            text=text,
            start_time=current_time,
            end_time=current_time + timedelta(seconds=10),
            is_question=is_question,
        )
        transcript_segments.append(segment)
        behavioral_extractor.add_transcript_segment(segment)
        current_time += timedelta(seconds=20)
    
    evidence_packets.extend(behavioral_extractor.extract_all())
    evidence_packets.sort(key=lambda e: e.timestamp)
    
    return SessionFixture(
        meeting_context=context,
        evidence_packets=evidence_packets,
        transcript_segments=transcript_segments,
        metadata={
            "name": "Display Name Change Mid-Call",
            "description": "Candidate changes display name from 'David Kim' to 'Dave K.' during interview",
            "difficulty": "hard",
        },
    )


def generate_silent_observer_scenario() -> SessionFixture:
    """Scenario: Silent observer who never speaks."""
    base_time = datetime(2024, 1, 15, 9, 0, 0)
    
    context = MeetingContext(
        meeting_id="silent-observer",
        candidate_name="Nina Patel",
        candidate_email="nina.patel@example.com",
        interviewer_names=["Mark Johnson"],
        interviewer_emails=["mark.johnson@company.com"],
        scheduled_start=base_time,
        participants={
            "nina": Participant(
                id="nina",
                display_name="Nina Patel",
                email="nina.patel@example.com",
                join_time=base_time - timedelta(seconds=30),
                webcam_on=True,
            ),
            "mark": Participant(
                id="mark",
                display_name="Mark Johnson",
                email="mark.johnson@company.com",
                join_time=base_time - timedelta(seconds=60),
                webcam_on=True,
            ),
            "olivia": Participant(
                id="olivia",
                display_name="Olivia Martinez",
                email="olivia.martinez@company.com",
                join_time=base_time - timedelta(seconds=45),
                webcam_on=False,
            ),
        },
    )
    
    evidence_packets = []
    transcript_segments = []
    
    identity_extractor = IdentitySignalExtractor(context)
    evidence_packets.extend(identity_extractor.extract_all())
    
    behavioral_extractor = BehavioralSignalExtractor(context)
    
    qa_pairs = [
        ("mark", "Tell me about your experience with DevOps practices.", True),
        ("nina", "I've been implementing CI/CD pipelines for the past 3 years. I work with Jenkins, GitLab CI, and GitHub Actions.", False),
        ("mark", "How do you ensure deployment reliability?", True),
        ("nina", "I use infrastructure as code with Terraform, implement blue-green deployments, and have comprehensive monitoring with Prometheus and Grafana.", False),
    ]
    
    current_time = base_time + timedelta(seconds=60)
    for speaker, text, is_question in qa_pairs:
        segment = TranscriptSegment(
            participant_id=speaker,
            text=text,
            start_time=current_time,
            end_time=current_time + timedelta(seconds=10),
            is_question=is_question,
        )
        transcript_segments.append(segment)
        behavioral_extractor.add_transcript_segment(segment)
        current_time += timedelta(seconds=20)
    
    evidence_packets.extend(behavioral_extractor.extract_all())
    evidence_packets.sort(key=lambda e: e.timestamp)
    
    return SessionFixture(
        meeting_context=context,
        evidence_packets=evidence_packets,
        transcript_segments=transcript_segments,
        metadata={
            "name": "Silent Observer",
            "description": "Three participants: candidate, interviewer, and a silent observer who never speaks",
            "difficulty": "medium",
        },
    )


def normalize_timestamps(fixture: SessionFixture, base_time: datetime) -> SessionFixture:
    """Normalize all timestamps to be relative to base_time."""
    # Spread evidence packets evenly over the interview duration
    # Assume a 30-minute interview
    interview_duration = timedelta(minutes=30)
    
    if fixture.evidence_packets:
        time_step = interview_duration / len(fixture.evidence_packets)
        for i, ep in enumerate(fixture.evidence_packets):
            ep.timestamp = base_time + (time_step * i)
    
    # Transcript segments already have correct timestamps relative to base_time
    # No adjustment needed
    
    return fixture


def save_fixture(fixture: SessionFixture, output_path: Path):
    """Save a session fixture to JSON file."""
    data = {
        "meeting_context": {
            "meeting_id": fixture.meeting_context.meeting_id,
            "candidate_name": fixture.meeting_context.candidate_name,
            "candidate_email": fixture.meeting_context.candidate_email,
            "interviewer_names": fixture.meeting_context.interviewer_names,
            "interviewer_emails": fixture.meeting_context.interviewer_emails,
            "scheduled_start": fixture.meeting_context.scheduled_start.isoformat() if fixture.meeting_context.scheduled_start else None,
            "participants": {
                pid: {
                    "display_name": p.display_name,
                    "email": p.email,
                    "join_time": p.join_time.isoformat() if p.join_time else None,
                    "webcam_on": p.webcam_on,
                    "is_screen_sharing": p.is_screen_sharing,
                }
                for pid, p in fixture.meeting_context.participants.items()
            },
        },
        "evidence_packets": [
            {
                "source": ep.source.value,
                "axis": ep.axis.value,
                "target_participant_id": ep.target_participant_id,
                "delta_log_odds": ep.delta_log_odds,
                "confidence": ep.confidence,
                "rationale": ep.rationale,
                "timestamp": ep.timestamp.isoformat(),
                "metadata": ep.metadata,
            }
            for ep in fixture.evidence_packets
        ],
        "transcript_segments": [
            {
                "participant_id": ts.participant_id,
                "text": ts.text,
                "start_time": ts.start_time.isoformat(),
                "end_time": ts.end_time.isoformat(),
                "is_question": ts.is_question,
            }
            for ts in fixture.transcript_segments
        ],
        "metadata": fixture.metadata,
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved: {output_path}")


def main():
    """Generate all fixtures."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)
    
    scenarios = [
        ("01_normal_interview.json", generate_normal_interview),
        ("02_device_name.json", generate_device_name_scenario),
        ("03_multiple_interviewers.json", generate_multiple_interviewers_scenario),
        ("04_nickname.json", generate_nickname_scenario),
        ("05_wrong_name.json", generate_wrong_name_scenario),
        ("06_display_name_change.json", generate_display_name_change_scenario),
        ("07_silent_observer.json", generate_silent_observer_scenario),
    ]
    
    print(f"Generating {len(scenarios)} fixtures in {fixtures_dir}...")
    
    for filename, generator in scenarios:
        fixture = generator()
        save_fixture(fixture, fixtures_dir / filename)
    
    print(f"\nGenerated {len(scenarios)} fixtures successfully!")


if __name__ == "__main__":
    main()
