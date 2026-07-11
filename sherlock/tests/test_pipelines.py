"""Tests for the real-time fraud-detection pipelines (Prompts 10.3.3-10.3.6)."""

import numpy as np

from sherlock.models import CandidateMediaFrame
from sherlock.pipelines import (
    DeepfakeVideoPipeline,
    DeepfakeDetector,
    GazeBehavioralPipeline,
    GazeDetector,
    LiveTranscriptionPipeline,
    SpeakerStore,
    Transcriber,
    VoiceLivenessDetector,
    VoiceLivenessPipeline,
)
from sherlock.tests._helpers import make_context


def _candidate_frame(video=None, audio=None):
    return CandidateMediaFrame(
        participant_id="candidate", candidate_id="candidate",
        video_frame=video, audio_chunk=audio, timestamp_ms=0,
    )


# ----- Deepfake (10.3.3) ---------------------------------------------- #
class FakeDeepfake(DeepfakeDetector):
    def __init__(self, score):
        self.score = score

    def detect(self, frame):
        return self.score, {"texture_variance": 0.0}


def test_deepfake_flags_high_score():
    pipe = DeepfakeVideoPipeline(make_context(["candidate"]), detector=FakeDeepfake(0.9))
    pkt = pipe.process(_candidate_frame(video=np.zeros((10, 10, 3), dtype=np.uint8)))
    assert pkt is not None
    assert pkt.source.value == "deepfake_video"
    assert pkt.delta_log_odds < 0
    assert pkt.severity.value in ("warning", "critical")


def test_deepfake_ignores_low_score():
    pipe = DeepfakeVideoPipeline(make_context(["candidate"]), detector=FakeDeepfake(0.1))
    assert pipe.process(_candidate_frame(video=np.zeros((10, 10, 3), dtype=np.uint8))) is None


# ----- Voice liveness (10.3.4) ---------------------------------------- #
class FakeVoice(VoiceLivenessDetector):
    def __init__(self, score):
        self.score = score

    def detect(self, pcm):
        return self.score, {"peak": 1.0}


def test_voice_flags_high_score():
    pipe = VoiceLivenessPipeline(make_context(["candidate"]), detector=FakeVoice(0.9))
    pkt = pipe.process(_candidate_frame(audio=np.zeros(16000, dtype=np.int16).tobytes()))
    assert pkt is not None
    assert pkt.source.value == "voice_liveness"
    assert pkt.delta_log_odds < 0


def test_speaker_store_detects_drift():
    store = SpeakerStore()
    a = np.ones(5, dtype=np.float32)
    b = -np.ones(5, dtype=np.float32)
    store.update("p1", a)
    assert store.has_baseline("p1")
    assert store.drift("p1", a) == 0.0
    assert store.drift("p1", b) == 2.0  # opposite vectors -> max drift


# ----- Gaze / behavioral CV (10.3.6) ---------------------------------- #
class FakeGaze(GazeDetector):
    def detect(self, frame):
        return (0.9, 0.5), True, 1.0  # off-screen, periodic saccade


def test_gaze_flags_periodic_offscreen():
    ctx = make_context(["candidate"])
    pipe = GazeBehavioralPipeline(ctx, detector=FakeGaze())
    result = None
    for _ in range(5):
        out = pipe.process(_candidate_frame(video=np.zeros((10, 10, 3), dtype=np.uint8)))
        if out:
            result = out
    assert result is not None
    # extract_gaze_detection returns a list of packets.
    packets = result if isinstance(result, list) else [result]
    assert any(p.source.value == "gaze_detection" for p in packets)


# ----- Live transcription (10.3.5) ------------------------------------ #
class FakeTranscriber(Transcriber):
    def __init__(self, text):
        self.text = text

    def transcribe(self, pcm):
        return self.text


def test_transcription_produces_segment():
    pipe = LiveTranscriptionPipeline(
        context=make_context(["candidate"]),
        transcriber=FakeTranscriber("What is your name?"),
        buffer_seconds=1.0,
    )
    empty = pipe.process(_candidate_frame(audio=np.zeros(16000, dtype=np.int16).tobytes()))
    # First 1s buffer fills but does not yet exceed 1.0*16000? It equals it -> transcribes.
    assert empty is not None
    seg = empty[0]
    assert seg.participant_id == "candidate"
    assert seg.is_question is True


if __name__ == "__main__":
    test_deepfake_flags_high_score()
    test_deepfake_ignores_low_score()
    test_voice_flags_high_score()
    test_speaker_store_detects_drift()
    test_gaze_flags_periodic_offscreen()
    test_transcription_produces_segment()
    print("All pipeline tests passed!")
