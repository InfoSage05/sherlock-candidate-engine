"""
Session replay module for Sherlock demo.

Provides unified interface for replaying pre-recorded sessions or running live simulations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sherlock.fusion import FusionEngine
from sherlock.models import (
    BeliefState,
    EvidencePacket,
    FlagSeverity,
    MeetingContext,
    OperatorNote,
    Participant,
    SignalAxis,
    SignalSource,
    SnapshotEntry,
    TranscriptSegment,
)


@dataclass
class SessionSnapshot:
    """Snapshot of session state at a specific point in time."""
    timestamp: datetime
    elapsed_seconds: float
    beliefs: Dict[str, BeliefState]
    evidence_ledger: List[EvidencePacket]
    transcript_segments: List[TranscriptSegment]
    top_candidate_id: Optional[str]
    top_candidate_probability: float
    ambiguity_gap: float
    status: str  # "identified" or "ambiguous"
    current_speaker_id: Optional[str]


@dataclass
class SessionFixture:
    """Pre-recorded session data."""
    meeting_context: MeetingContext
    evidence_packets: List[EvidencePacket]
    transcript_segments: List[TranscriptSegment]
    metadata: Dict  # scenario name, description, etc.


class SessionReplay:
    """Replays a pre-recorded session or runs live simulation."""
    
    def __init__(self, fixture: Optional[SessionFixture] = None):
        self.fixture = fixture
        self.engine: Optional[FusionEngine] = None
        self.current_index = 0
        self.start_time: Optional[datetime] = None
        self.current_time: Optional[datetime] = None
        self.playback_speed = 1.0
        self.is_playing = False
        
        if fixture:
            self.engine = FusionEngine(fixture.meeting_context)
            self.start_time = self._get_start_time()
            self.current_time = self.start_time
    
    def _get_start_time(self) -> datetime:
        """Get the earliest timestamp in the session."""
        if not self.fixture:
            return datetime.utcnow()
        
        all_times = []
        if self.fixture.evidence_packets:
            all_times.extend([ep.timestamp for ep in self.fixture.evidence_packets])
        if self.fixture.transcript_segments:
            all_times.extend([ts.start_time for ts in self.fixture.transcript_segments])
        
        return min(all_times) if all_times else datetime.utcnow()
    
    def reset(self):
        """Reset session to beginning."""
        if self.fixture:
            self.engine = FusionEngine(self.fixture.meeting_context)
            self.current_index = 0
            self.current_time = self.start_time
            self.is_playing = False
    
    def step_forward(self) -> Optional[SessionSnapshot]:
        """Advance to next evidence packet and return snapshot."""
        if not self.fixture or not self.engine:
            return None
        
        if self.current_index >= len(self.fixture.evidence_packets):
            return None
        
        packet = self.fixture.evidence_packets[self.current_index]
        self.engine.ingest(packet)
        self.current_index += 1
        self.current_time = packet.timestamp
        
        return self.get_current_snapshot()
    
    def advance_to_time(self, target_time: datetime) -> SessionSnapshot:
        """Advance session to specific timestamp."""
        if not self.fixture or not self.engine:
            raise ValueError("No fixture loaded")
        
        # Reset and replay up to target time
        self.reset()
        
        while self.current_index < len(self.fixture.evidence_packets):
            packet = self.fixture.evidence_packets[self.current_index]
            if packet.timestamp > target_time:
                break
            self.engine.ingest(packet)
            self.current_index += 1
            self.current_time = packet.timestamp
        
        return self.get_current_snapshot()
    
    def get_current_snapshot(self) -> SessionSnapshot:
        """Get current session state."""
        if not self.engine or not self.current_time:
            raise ValueError("Session not initialized")
        
        result = self.engine.get_result(self.current_time)
        
        # Get transcript segments up to current time
        transcript_up_to_now = [
            ts for ts in (self.fixture.transcript_segments if self.fixture else [])
            if ts.start_time <= self.current_time
        ]
        
        # Determine current speaker (last transcript segment)
        current_speaker = None
        if transcript_up_to_now:
            current_speaker = transcript_up_to_now[-1].participant_id
        
        elapsed = (self.current_time - self.start_time).total_seconds() if self.start_time else 0
        
        return SessionSnapshot(
            timestamp=self.current_time,
            elapsed_seconds=elapsed,
            beliefs=dict(self.engine.beliefs),
            evidence_ledger=list(self.engine.evidence_ledger),
            transcript_segments=transcript_up_to_now,
            top_candidate_id=result.top_candidate_id,
            top_candidate_probability=result.top_candidate_probability,
            ambiguity_gap=result.ambiguity_gap,
            status=result.status,
            current_speaker_id=current_speaker,
        )
    
    def get_total_duration(self) -> float:
        """Get total session duration in seconds."""
        if not self.fixture:
            return 0
        
        all_times = []
        if self.fixture.evidence_packets:
            all_times.extend([ep.timestamp for ep in self.fixture.evidence_packets])
        if self.fixture.transcript_segments:
            all_times.extend([ts.start_time for ts in self.fixture.transcript_segments])
        
        if not all_times:
            return 0
        
        start = min(all_times)
        end = max(all_times)
        return (end - start).total_seconds()
    
    def get_progress(self) -> float:
        """Get current progress as fraction (0.0 to 1.0)."""
        total = self.get_total_duration()
        if total == 0 or not self.current_time or not self.start_time:
            return 0.0
        
        elapsed = (self.current_time - self.start_time).total_seconds()
        return min(elapsed / total, 1.0)
    
    def set_playback_speed(self, speed: float):
        """Set playback speed multiplier."""
        self.playback_speed = max(0.5, min(4.0, speed))

    def get_snapshot_history(self) -> List[SnapshotEntry]:
        """Get full snapshot history from the engine."""
        if not self.engine:
            return []
        return self.engine.get_snapshot_history()

    def get_participant_timeline(self, participant_id: str, field: str = "identity_probability") -> List:
        """Get a time series of a metric for a participant."""
        if not self.engine:
            return []
        return self.engine.get_participant_timeline(participant_id, field)

    def get_active_flags(self) -> List[EvidencePacket]:
        """Get evidence packets with non-NONE severity."""
        if not self.engine:
            return []
        return self.engine._collect_active_flags()


def load_fixture(fixture_path: Path) -> SessionFixture:
    """Load a session fixture from JSON file."""
    with open(fixture_path, 'r') as f:
        data = json.load(f)
    
    # Parse meeting context
    context_data = data['meeting_context']
    participants = {}
    for pid, pdata in context_data['participants'].items():
        participants[pid] = Participant(
            id=pid,
            display_name=pdata['display_name'],
            email=pdata.get('email'),
            join_time=datetime.fromisoformat(pdata['join_time']) if pdata.get('join_time') else None,
            webcam_on=pdata.get('webcam_on', False),
            is_screen_sharing=pdata.get('is_screen_sharing', False),
        )
    
    context = MeetingContext(
        meeting_id=context_data['meeting_id'],
        candidate_name=context_data.get('candidate_name'),
        candidate_email=context_data.get('candidate_email'),
        interviewer_names=context_data.get('interviewer_names', []),
        interviewer_emails=context_data.get('interviewer_emails', []),
        scheduled_start=datetime.fromisoformat(context_data['scheduled_start']) if context_data.get('scheduled_start') else None,
        participants=participants,
    )
    
    # Parse evidence packets
    evidence_packets = []
    AUTH_SOURCES = {SignalSource.DISFLUENCY_ANOMALY, SignalSource.PAUSE_FLUENCY_PATTERN,
                    SignalSource.CODING_TELEMETRY, SignalSource.GAZE_DETECTION}
    for ep_data in data['evidence_packets']:
        source = SignalSource(ep_data['source'])
        axis = SignalAxis(ep_data['axis'])
        severity = FlagSeverity.NONE
        if "severity" in ep_data and ep_data["severity"]:
            try:
                severity = FlagSeverity(ep_data["severity"])
            except ValueError:
                severity = FlagSeverity.NONE
        elif source in AUTH_SOURCES and axis == SignalAxis.AUTHENTICITY:
            severity = FlagSeverity.WARNING

        evidence_packets.append(EvidencePacket(
            source=source,
            axis=axis,
            target_participant_id=ep_data['target_participant_id'],
            delta_log_odds=ep_data['delta_log_odds'],
            confidence=ep_data['confidence'],
            rationale=ep_data['rationale'],
            timestamp=datetime.fromisoformat(ep_data['timestamp']),
            metadata=ep_data.get('metadata', {}),
            severity=severity,
            flag_type=ep_data.get("flag_type", f"{ep_data['source']}_auto" if source in AUTH_SOURCES else ""),
            recommendation=ep_data.get("recommendation", ""),
        ))
    
    # Parse transcript segments
    transcript_segments = []
    for ts_data in data.get('transcript_segments', []):
        transcript_segments.append(TranscriptSegment(
            participant_id=ts_data['participant_id'],
            text=ts_data['text'],
            start_time=datetime.fromisoformat(ts_data['start_time']),
            end_time=datetime.fromisoformat(ts_data['end_time']),
            is_question=ts_data.get('is_question', False),
        ))
    
    return SessionFixture(
        meeting_context=context,
        evidence_packets=evidence_packets,
        transcript_segments=transcript_segments,
        metadata=data.get('metadata', {}),
    )


def list_available_fixtures(fixtures_dir: Path) -> List[Dict]:
    """List all available fixture files with metadata."""
    fixtures = []
    for fixture_file in sorted(fixtures_dir.glob('*.json')):
        try:
            with open(fixture_file, 'r') as f:
                data = json.load(f)
            fixtures.append({
                'id': fixture_file.stem,
                'name': data.get('metadata', {}).get('name', fixture_file.stem),
                'description': data.get('metadata', {}).get('description', ''),
                'path': fixture_file,
            })
        except Exception as e:
            print(f"Warning: Could not load {fixture_file}: {e}")
    
    return fixtures
