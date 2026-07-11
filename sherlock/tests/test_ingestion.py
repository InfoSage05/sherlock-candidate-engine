"""Tests for the ingestion adapter layer (Prompt 10.3.1)."""

import asyncio

import numpy as np

from sherlock.ingestion import FileSource, WebRTCSource


def test_filesource_synthetic_yields_frames():
    async def run():
        src = FileSource(participant_id="candidate", generate_synthetic=True, max_frames=10)
        await src.start()
        frames = []
        async for f in src.frames():
            frames.append(f)
        await src.stop()
        return frames

    frames = asyncio.run(run())
    assert len(frames) == 10
    f0 = frames[0]
    assert f0.participant_id == "candidate"
    assert isinstance(f0.video_frame, np.ndarray)
    assert f0.video_frame.shape == (240, 240, 3)
    assert isinstance(f0.audio_chunk, bytes)
    assert len(f0.audio_chunk) == 16000 * 2  # 1s @ 16kHz 16-bit
    assert f0.timestamp_ms < frames[-1].timestamp_ms


def test_filesource_rejects_missing_file():
    async def run():
        src = FileSource(path="/nonexistent/file.mp4")
        try:
            await src.start()
            return False
        except FileNotFoundError:
            return True

    assert asyncio.run(run())


def test_webrtc_source_importable_but_requires_aiortc():
    src = WebRTCSource(signaling_url="ws://x", room="r")
    assert hasattr(src, "start")

    async def run():
        try:
            await src.start()
            return False
        except RuntimeError:
            return True

    # aiortc is not installed in this environment.
    assert asyncio.run(run())
