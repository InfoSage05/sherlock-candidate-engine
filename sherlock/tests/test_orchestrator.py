"""End-to-end test for the streaming orchestrator (Prompt 10.3.7)."""

import asyncio

from sherlock.fusion import FusionEngine
from sherlock.ingestion import FileSource
from sherlock.models import SignalAxis, SignalSource
from sherlock.orchestrator import RealtimeInferenceOrchestrator
from sherlock.pipelines import (
    DeepfakeDetector,
    DeepfakeVideoPipeline,
    Transcriber,
)
from sherlock.tests._helpers import make_identified_engine


class _HighDeepfake(DeepfakeDetector):
    def detect(self, frame):
        return 0.9, {}


class _FakeTranscriber(Transcriber):
    def transcribe(self, pcm):
        return None  # no speech in synthetic frames


def test_orchestrator_routes_candidate_media_to_pipelines():
    async def run():
        context, engine = make_identified_engine(
            ["candidate", "interviewer"], "candidate", "interviewer"
        )
        source = FileSource(
            participant_id="candidate", generate_synthetic=True, max_frames=5
        )
        deepfake = DeepfakeVideoPipeline(context, detector=_HighDeepfake())
        orch = RealtimeInferenceOrchestrator(
            engine=engine, source=source, context=context, deepfake=deepfake,
            transcription=_FakeTranscriberTranscription(context),
        )
        orch._is_stale = lambda frame: False
        await orch.start()
        # Allow frames to flow through ingest + candidate loops.
        for _ in range(50):
            if orch.latency_ms_samples or not orch._running:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.2)
        await orch.stop()
        return orch, engine

    orch, engine = asyncio.run(run())

    # Non-candidate frames were never analyzed.
    assert orch.gate.dropped_frames_non_candidate == 0
    # At least one authenticity packet from the deepfake pipeline reached the ledger.
    sources = [
        ep.source
        for ep in engine.evidence_ledger
        if ep.axis == SignalAxis.AUTHENTICITY
    ]
    assert SignalSource.DEEPFAKE_VIDEO in sources
    # Latency was measured for processed candidate frames.
    assert len(orch.latency_ms_samples) >= 1


class _FakeTranscriberTranscription:
    """Thin wrapper reusing LiveTranscriptionPipeline with a fake transcriber."""
    def __new__(cls, context):
        from sherlock.pipelines import LiveTranscriptionPipeline

        return LiveTranscriptionPipeline(context, transcriber=_FakeTranscriber())


if __name__ == "__main__":
    test_orchestrator_routes_candidate_media_to_pipelines()
    print("All orchestrator tests passed!")
