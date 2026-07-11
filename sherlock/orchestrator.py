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
from .models import EvidencePacket, RawMediaFrame
from .signals.behavioral import BehavioralSignalExtractor
from .pipelines.deepfake import DeepfakeVideoPipeline
from .pipelines.gaze_cv import GazeBehavioralPipeline
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
        candidate_queue_size: int = MAX_QUEUE,
    ) -> None:
        self.engine = engine
        self.source = source
        self.context = context
        self.gate = gate or CandidateStreamGate(engine)
        self.deepfake = deepfake or DeepfakeVideoPipeline(context)
        self.voice = voice or VoiceLivenessPipeline(context)
        self.gaze = gaze or GazeBehavioralPipeline(context)
        self.transcription = transcription or LiveTranscriptionPipeline(context)
        self._candidate_queue: asyncio.Queue = asyncio.Queue(maxsize=candidate_queue_size)
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self.latency_ms_samples: List[float] = []
        self.dropped_stale: int = 0
        self.transcript_segments = []
        self._behavioral = BehavioralSignalExtractor(context)

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
                for seg in segments:
                    self.transcript_segments.append(seg)
                    self._behavioral.add_transcript_segment(seg)
                # Convert transcript into identity evidence immediately.
                evidence = self._behavioral.extract_all()
                if evidence:
                    self.engine.ingest_batch(evidence)
        except Exception as exc:
            logger.warning("Transcription pipeline error: %s", exc)

    async def _run_candidate_pipelines(self, frame) -> None:
        packets: List[EvidencePacket] = []
        t0 = time.perf_counter()
        for pipeline in (self.deepfake, self.voice, self.gaze):
            try:
                result = pipeline.process(frame)
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
