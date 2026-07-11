from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class SignalSource(str, Enum):
    CALENDAR_MATCH = "calendar_match"
    INTERVIEWER_NEGATIVE = "interviewer_negative"
    EMAIL_DOMAIN = "email_domain"
    TURN_TAKING = "turn_taking"
    SPEAKING_RATIO = "speaking_ratio"
    SCREEN_SHARE = "screen_share"
    LLM_ROLE_CLASSIFIER = "llm_role_classifier"
    DISFLUENCY_ANOMALY = "disfluency_anomaly"
    PAUSE_FLUENCY_PATTERN = "pause_fluency_pattern"
    CODING_TELEMETRY = "coding_telemetry"
    GAZE_DETECTION = "gaze_detection"
    DISPLAY_NAME_CHANGE = "display_name_change"
    JOIN_TIMING = "join_timing"
    WEBCAM_STATE = "webcam_state"
    DEEPFAKE_VIDEO = "deepfake_video"
    VOICE_LIVENESS = "voice_liveness"
    AI_GENERATED_TEXT = "ai_generated_text"
    AI_GENERATED_SPEECH = "ai_generated_speech"
    READING_PATTERN = "reading_pattern"
    UNNATURAL_PAUSE = "unnatural_pause"
    IDENTITY_UNCERTAIN = "identity_uncertain"


class SignalAxis(str, Enum):
    IDENTITY = "identity"
    AUTHENTICITY = "authenticity"


class FlagSeverity(str, Enum):
    NONE = "none"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class MeetingStatus(str, Enum):
    ACTIVE = "active"
    CONCLUDED = "concluded"


@dataclass
class Participant:
    id: str
    display_name: str
    email: Optional[str] = None
    join_time: Optional[datetime] = None
    leave_time: Optional[datetime] = None
    webcam_on: bool = False
    is_screen_sharing: bool = False
    device_name: Optional[str] = None
    display_name_history: List[dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


@dataclass
class MeetingContext:
    meeting_id: str
    candidate_name: Optional[str] = None
    candidate_email: Optional[str] = None
    interviewer_names: List[str] = field(default_factory=list)
    interviewer_emails: List[str] = field(default_factory=list)
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None
    participants: Dict[str, Participant] = field(default_factory=dict)
    status: MeetingStatus = MeetingStatus.ACTIVE


@dataclass
class EvidencePacket:
    source: SignalSource
    axis: SignalAxis
    target_participant_id: str
    delta_log_odds: float
    confidence: float
    rationale: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    evidence_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    severity: FlagSeverity = FlagSeverity.NONE
    flag_type: str = ""
    recommendation: str = ""

    @property
    def delta_probability(self) -> float:
        odds = math.exp(self.delta_log_odds)
        return odds / (1 + odds)


@dataclass
class ParticipantMetrics:
    participant_id: str
    speaking_ratio: float = 0.0
    reactive_ratio: float = 0.0
    turn_in_degree: int = 0
    turn_out_degree: int = 0
    total_speaking_seconds: float = 0.0
    response_count: int = 0
    question_count: int = 0
    average_response_latency: float = 0.0
    identity_probability: float = 0.0
    authenticity_probability: float = 0.0
    evidence_count: int = 0


@dataclass
class SnapshotEntry:
    timestamp: datetime
    elapsed_seconds: float
    top_candidate_id: Optional[str] = None
    top_candidate_probability: float = 0.0
    status: str = "no_participants"
    ambiguity_gap: float = 0.0
    participants_metrics: Dict[str, ParticipantMetrics] = field(default_factory=dict)
    active_flags: List[EvidencePacket] = field(default_factory=list)


@dataclass
class OperatorNote:
    meeting_id: str
    timestamp: datetime
    author: str = "operator"
    text: str = ""
    linked_evidence_id: Optional[str] = None
    linked_participant_id: Optional[str] = None
    note_type: str = "general"  # general | correction_reason | flag_context | verdict
    note_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])


@dataclass
class RawMediaFrame:
    """A single audio/video sample from one participant at one moment in time."""
    participant_id: str
    audio_chunk: Optional[bytes] = None  # 16-bit PCM, 16 kHz, mono
    video_frame: Optional[Any] = None     # numpy array (H, W, 3)
    timestamp_ms: int = 0


@dataclass
class CandidateMediaFrame(RawMediaFrame):
    """A RawMediaFrame that has passed the candidate stream gate."""
    candidate_id: str = ""


@dataclass
class BeliefState:
    participant_id: str
    identity_log_odds: float = 0.0
    authenticity_log_odds: float = 0.0
    identity_evidence: List[EvidencePacket] = field(default_factory=list)
    authenticity_evidence: List[EvidencePacket] = field(default_factory=list)
    last_updated: Optional[datetime] = None

    @property
    def identity_probability(self) -> float:
        odds = math.exp(self.identity_log_odds)
        return odds / (1 + odds)

    @property
    def authenticity_probability(self) -> float:
        odds = math.exp(self.authenticity_log_odds)
        return odds / (1 + odds)


@dataclass
class IdentificationResult:
    meeting_id: str
    timestamp: datetime
    top_candidate_id: Optional[str]
    top_candidate_probability: float
    status: str
    all_hypotheses: Dict[str, BeliefState]
    ambiguity_gap: float
    rationale: str

    @property
    def is_ambiguous(self) -> bool:
        return self.status == "ambiguous"


@dataclass
class SpeakingEvent:
    participant_id: str
    start_time: datetime
    end_time: datetime
    transcript: Optional[str] = None
    is_response_to_question: bool = False

    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()


@dataclass
class TranscriptSegment:
    participant_id: str
    text: str
    start_time: datetime
    end_time: datetime
    is_question: bool = False


@dataclass
class CodingEvent:
    participant_id: str
    timestamp: datetime
    event_type: str
    content_length: int = 0
    keystroke_interval_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GazeEvent:
    participant_id: str
    timestamp: datetime
    gaze_vector: tuple
    is_off_screen: bool = False
    saccade_periodicity: Optional[float] = None


@dataclass
class SignalWeight:
    source: SignalSource
    axis: SignalAxis
    weight: float = 1.0
    calibration_count: int = 0
    calibration_sum: float = 0.0

    @property
    def calibrated_weight(self) -> float:
        if self.calibration_count == 0:
            return self.weight
        return self.calibration_sum / self.calibration_count


@dataclass
class FeedbackEvent:
    meeting_id: str
    timestamp: datetime
    feedback_type: str
    participant_id: Optional[str] = None
    correct: bool = False
    signal_source: Optional[SignalSource] = None
    notes: str = ""
