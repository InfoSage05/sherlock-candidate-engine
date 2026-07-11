"""YouTube video download utility.

Downloads a YouTube video (or any URL supported by ``yt-dlp``) into a local
temporary file so the rest of the ingestion pipeline can process it as a normal
media file.  Caches downloads so the same URL is not re-downloaded on every
rerun.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(tempfile.gettempdir()) / "sherlock_yt_cache"


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


@dataclass
class YouTubeVideo:
    """Metadata + local file path for a downloaded YouTube video."""
    url: str
    title: str
    duration_seconds: float
    file_path: str
    thumbnail_url: str = ""


_YT_DLP_AVAILABLE: Optional[bool] = None


def _yt_dlp_available() -> bool:
    global _YT_DLP_AVAILABLE
    if _YT_DLP_AVAILABLE is None:
        _YT_DLP_AVAILABLE = (
            subprocess.run(
                ["yt-dlp", "--version"],
                capture_output=True, timeout=5,
            ).returncode == 0
        )
    return _YT_DLP_AVAILABLE


def _parse_duration(raw: str) -> float:
    """Parse ``HH:MM:SS`` or ``MM:SS`` or ``SS`` into seconds."""
    parts = raw.strip().split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except Exception:
        return 0.0


def download_youtube_video(
    url: str,
    max_height: int = 720,
    cache: bool = True,
) -> YouTubeVideo:
    """Download a YouTube video and return its metadata + local path.

    Parameters
    ----------
    url : str
        Full YouTube URL (or any URL ``yt-dlp`` supports).
    max_height : int
        Maximum video resolution to download.
    cache : bool
        If True and the file already exists in the cache, skip re-download.
    """
    if not _yt_dlp_available():
        raise RuntimeError(
            "yt-dlp is not installed. Install with: pip install yt-dlp"
        )

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url_id = _url_hash(url)
    cached_file = _CACHE_DIR / f"{url_id}.mp4"
    cached_meta = _CACHE_DIR / f"{url_id}.info.json"

    if cache and cached_file.exists() and cached_meta.exists():
        import json

        with open(cached_meta) as f:
            meta = json.load(f)
        logger.info("Using cached download: %s", cached_file)
        return YouTubeVideo(
            url=url,
            title=meta.get("title", "Unknown"),
            duration_seconds=meta.get("duration", 0),
            file_path=str(cached_file),
            thumbnail_url=meta.get("thumbnail", ""),
        )

    # --- live download ------------------------------------------------ #
    logger.info("Downloading YouTube video: %s", url)
    output_template = str(_CACHE_DIR / f"{url_id}.%(ext)s")
    info_json_path = str(_CACHE_DIR / f"{url_id}.info.json")

    cmd = [
        "yt-dlp",
        "-f", f"best[height<={max_height}]",
        "--write-info-json",
        "--no-playlist",
        "-o", output_template,
        url,
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr}")

    # Find the downloaded file (could be .mp4 / .webm / etc.)
    downloaded_file: Optional[Path] = None
    for ext in ("mp4", "webm", "mkv"):
        candidate = _CACHE_DIR / f"{url_id}.{ext}"
        if candidate.exists():
            downloaded_file = candidate
            break
    if downloaded_file is None:
        raise RuntimeError(f"Download succeeded but no video file found in {_CACHE_DIR}")

    # Rename to .mp4 for consistency.
    final_path = _CACHE_DIR / f"{url_id}.mp4"
    if downloaded_file != final_path:
        downloaded_file.rename(final_path)

    # Parse metadata.
    title = url
    duration = 0.0
    thumbnail = ""
    if Path(info_json_path).exists():
        import json

        with open(info_json_path) as f:
            meta = json.load(f)
        title = meta.get("title", url)
        duration = meta.get("duration", 0)
        thumbnail = meta.get("thumbnail", "")

    logger.info("Downloaded '%s' (%.0fs) -> %s", title, duration, final_path)
    return YouTubeVideo(
        url=url,
        title=title,
        duration_seconds=duration,
        file_path=str(final_path),
        thumbnail_url=thumbnail,
    )


def is_youtube_url(text: str) -> bool:
    """Return True if ``text`` looks like a YouTube URL."""
    return bool(re.match(
        r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)",
        text.strip(),
    ))
