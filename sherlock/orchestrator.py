"""Real-time inference orchestrator.

Wires the ingestion source, candidate gate, and all fraud-detection pipelines
together:

1. Reads ``RawMediaFrame`` objects from a ``MediaSource``.
2. Runs live transcription on **every** participant's audio (feeds the
   existing identity/behavioral signals).
3. Routes frames through the ``CandidateStreamGate`` so only the candidate's
   media reaches the A/V pipelines.
4. Runs deepfake / voice-liveness / gaze pipelines on candidate frames.
5. Ingests all resulting ``EvidencePacket`` objects into the ``FusionEngine``.

Designed for backpressure (bounded queues), a latency budget (stale frames are
dropped), and fault isolation (one crashing pipeline never takes down the
others).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Optional

from .fusion import FusionEngine
from .gate import CandidateStreamGate
from .ingestion.base import MediaSource
from .models import EvidencePacket, FlagSeverity, RawMediaFrame, SignalAxis, SignalSource
from .signals.behavioral import BehavioralSignalExtractor
from .pipelines.audio_authenticity import AudioAuthenticityPipeline
from .pipelines.deepfake import DeepfakeVideoPipeline
from .pipelines.gaze_cv import GazeBehavioralPipeline
from .pipelines.text_authenticity import TextAuthenticityPipeline
from .pipelines.transcription_live import LiveTranscriptionPipeline
from .pipelines.voice_liveness import VoiceLivenessPipeline

logger = logging.getLogger(__name__)

STALE_FRAME_MS = 1000  # drop frames older than this when the queue is backed up
MAX_QUEUE = 30


class RealtimeInferenceOrchestrator:
    def __init__(
        self,
        engine: FusionEngine,
        source: MediaSource,
        context,
        gate: Optional[CandidateStreamGate] = None,
        deepfake: Optional[DeepfakeVideoPipeline] = None,
        voice: Optional[VoiceLivenessPipeline] = None,
        gaze: Optional[GazeBehavioralPipeline] = None,
        transcription: Optional[LiveTranscriptionPipeline] = None,
        text_authenticity: Optional[TextAuthenticityPipeline] = None,
        audio_authenticity: Optional[AudioAuthenticityPipeline] = None,
        candidate_queue_size: int = MAX_QUEUE,
    ) -> None:
        self.engine = engine
        self.source = source
        self.context = context
        self.gate = gate or CandidateStreamGate(engine)
        self.deepfake = deepfake or DeepfakeVideoPipeline(context)
        self.voice = voice or VoiceLivenessPipeline(context)
        self.gaze = gaze or GazeBehavioralPipeline(context)
        self.audio_authenticity = audio_authenticity or AudioAuthenticityPipeline(context)
        self.transcription = transcription or LiveTranscriptionPipeline(context)
        self.text_authenticity = text_authenticity or self._build_text_authenticity(context)
        self._candidate_queue: asyncio.Queue = asyncio.Queue(maxsize=candidate_queue_size)
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self.latency_ms_samples: List[float] = []
        self.dropped_stale: int = 0
        self.transcript_segments = []
        self._behavioral = BehavioralSignalExtractor(context)

    @staticmethod
    def _build_text_authenticity(context):
        from .pipelines.text_authenticity import TextAuthenticityPipeline

        semantic_model = None
        try:
            from sentence_transformers import SentenceTransformer

            semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded sentence-transformers model for Q/A relevance.")
        except Exception as exc:
            logger.info("Sentence-transformers not available: %s", exc)

        return TextAuthenticityPipeline(context=context, semantic_model=semantic_model)

    # ----- public control --------------------------------------------- #
    async def start(self) -> None:
        await self.source.start()
        self._running = True
        self._tasks = [
            asyncio.create_task(self._ingest_loop()),
            asyncio.create_task(self._candidate_loop()),
        ]
        logger.info("Orchestrator started.")

    async def stop(self) -> None:
        self._running = False
        await self.source.stop()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("Orchestrator stopped.")

    # ----- internal loops ---------------------------------------------- #
    async def _ingest_loop(self) -> None:
        try:
            async for frame in self.source.frames():
                if not self._running:
                    break
                # Transcription runs on ALL participants' audio.
                await self._run_transcription(frame)
                # Gate routes only candidate frames downstream.
                candidate_frame = self.gate.process(frame)
                flags = self.gate.consume_flags()
                if flags:
                    self.engine.ingest_batch(flags)
                if candidate_frame is not None:
                    try:
                        self._candidate_queue.put_nowait(candidate_frame)
                    except asyncio.QueueFull:
                        # Drop the oldest stale frame.
                        try:
                            self._candidate_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        self._candidate_queue.put_nowait(candidate_frame)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Ingest loop crashed: %s", exc)

    async def _candidate_loop(self) -> None:
        try:
            while self._running:
                frame = await self._candidate_queue.get()
                if self._is_stale(frame):
                    self.dropped_stale += 1
                    continue
                await self._run_candidate_pipelines(frame)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Candidate loop crashed: %s", exc)

    # ----- pipeline execution ------------------------------------------ #
    async def _run_transcription(self, frame: RawMediaFrame) -> None:
        try:
            segments = self.transcription.process(frame)
            if segments:
                all_evidence: List[EvidencePacket] = []
                for seg in segments:
                    self.transcript_segments.append(seg)
                    self._behavioral.add_transcript_segment(seg)
                    # Text-level authenticity (AI-generated text, reading pattern, pauses).
                    all_evidence.extend(await self.text_authenticity.aprocess(seg))
                # Convert transcript into identity/behavioral evidence.
                all_evidence.extend(self._behavioral.extract_all())
                if all_evidence:
                    self.engine.ingest_batch(all_evidence)

                # Lightweight per-segment activity boost for the current speaker.
                # This keeps the scoreboard moving continuously while someone talks.
                for seg in segments:
                    if seg.participant_id and self.gate.candidate_id and seg.participant_id == self.gate.candidate_id:
                        self.engine.ingest_batch([EvidencePacket(
                            source=SignalSource.TURN_TAKING,
                            axis=SignalAxis.IDENTITY,
                            target_participant_id=seg.participant_id,
                            delta_log_odds=0.08,
                            confidence=0.25,
                            severity=FlagSeverity.NONE,
                            flag_type="candidate_speech_activity",
                            recommendation="",
                            rationale="Candidate is actively speaking.",
                            timestamp=seg.start_time,
                            metadata={"segment_text": seg.text[:80]},
                        )])
        except Exception as exc:
            logger.warning("Transcription pipeline error: %s", exc)

    async def _run_candidate_pipelines(self, frame) -> None:
        packets: List[EvidencePacket] = []
        t0 = time.perf_counter()
        loop = asyncio.get_event_loop()
        for pipeline in (self.deepfake, self.voice, self.gaze, self.audio_authenticity):
            try:
                # Run heavy CV/audio models in thread pool so the event loop stays alive.
                result = await loop.run_in_executor(None, pipeline.process, frame)
            except Exception as exc:
                logger.warning("%s crashed: %s", type(pipeline).__name__, exc)
                continue
            if result is None:
                continue
            if isinstance(result, list):
                packets.extend(result)
            else:
                packets.append(result)
        if packets:
            self.engine.ingest_batch(packets)
        self.latency_ms_samples.append((time.perf_counter() - t0) * 1000.0)

    # ----- helpers ----------------------------------------------------- #
    def _is_stale(self, frame) -> bool:
        age_ms = time.time() * 1000 - frame.timestamp_ms
        return age_ms > STALE_FRAME_MS

    def p95_latency_ms(self) -> float:
        if not self.latency_ms_samples:
            return 0.0
        ordered = sorted(self.latency_ms_samples)
        idx = min(len(ordered) - 1, int(0.95 * len(ordered)))
        return ordered[idx]
