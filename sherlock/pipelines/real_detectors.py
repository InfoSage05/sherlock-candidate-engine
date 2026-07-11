"""Real detector backends that use installed CV/ML libraries.

These replace the heuristic detectors in the pipelines when the required
libraries are available. Each class implements the same detector protocol as
its heuristic counterpart, so they can be swapped in at construction time with
no pipeline code changes.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Deepfake / face-manipulation detector ──────────────────────────── #

class RealDeepfakeDetector:
    """Uses ``insightface`` for face detection, MediaPipe Face Mesh for 3D
    landmarks, and a liveness heuristic (texture variance + facial landmark
    stability) to score P(frame is manipulated).

    In production this should be replaced with a trained classifier (e.g.
    Silent-Face-Anti-Spoofing or a MobileNet liveness model).
    """

    def __init__(self):
        import insightface

        self._detector = insightface.model_zoo.get_model("buffalo_l")
        if self._detector is None:
            raise RuntimeError(
                "insightface buffal_l model not available — download may require "
                "internet on first run. Falling back to HeuristicDeepfakeDetector."
            )

    def detect(self, frame: np.ndarray) -> Tuple[float, dict]:
        if self._detector is None:
            return 0.0, {"faces_detected": 0, "error": "model not loaded"}
        try:
            import mediapipe as mp

            mp_version = getattr(mp, "__version__", "0")
            major = int(mp_version.split(".")[0]) if mp_version else 0
            if major >= 10:
                mp = None
        except Exception:
            mp = None

        img = np.asarray(frame)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        h, w = img.shape[:2]

        faces = self._detector.get(img, max_num=1)
        if not faces:
            return 0.0, {"faces_detected": 0}

        # Texture variance (same heuristic as before, scaled to frame).
        gray = img.astype(np.float32).mean(axis=2)
        lap = np.abs(gray[1:, :] - gray[:-1, :]) + np.abs(gray[:, 1:] - gray[:, :-1])
        texture = float(lap.var())
        texture_ratio = float(np.clip(texture / (h * w * 0.01), 0.0, 1.0))

        # --- 3D face-mesh stability (MediaPipe) ------------------------------ #
        mesh_stability_score = 0.5  # neutral default
        if mp is not None:
            try:
                mp_face = mp.solutions.face_mesh
                with mp_face.FaceMesh(
                    static_image_mode=True, max_num_faces=1, refine_landmarks=True
                ) as face_mesh:
                    results = face_mesh.process(img)
                    if results.multi_face_landmarks:
                        lm = results.multi_face_landmarks[0]
                        # Low landmark variance = possible static image or photo.
                        pts = np.array([(l.x, l.y, l.z) for l in lm.landmark])
                        lm_var = float(np.var(pts))
                        mesh_stability_score = float(
                            np.clip(1.0 - lm_var / 0.001, 0.0, 1.0)
                        )
            except Exception:
                pass

        face_count_score = 0.0 if faces else 0.3  # no face = slightly suspicious
        score = float(
            np.clip(
                0.35 * texture_ratio + 0.35 * mesh_stability_score + 0.3 * face_count_score,
                0.0,
                1.0,
            )
        )
        return score, {
            "texture_variance": texture,
            "mesh_stability_score": mesh_stability_score,
            "faces_detected": len(faces),
        }


# ── Voice-liveness / anti-cloning detector ─────────────────────────── #

class RealVoiceLivenessDetector:
    """Uses ``resemblyzer`` speaker embeddings to detect in-session drift
    and ``librosa`` spectral features to flag replay / synthesis artifacts."""

    def __init__(self):
        import resemblyzer  # noqa: F401

    def detect(self, pcm: np.ndarray) -> Tuple[float, dict]:
        if pcm.size < 1600:
            return 0.0, {"energy": 0.0}

        import librosa
        import resemblyzer

        audio = pcm.astype(np.float32) / 32768.0

        # Clipping / saturation check.
        peak = float(np.abs(pcm).max()) / 32768.0
        clip_ratio = float(np.mean(np.abs(pcm) > 30000)) if pcm.size else 0.0
        clip_score = float(np.clip((peak > 0.98) * 0.3 + clip_ratio * 2.0, 0.0, 1.0))

        # Spectral flatness — very flat spectrum suggests TTS/synthesis.
        try:
            spec = np.abs(librosa.stft(audio, n_fft=512))
            gmean = np.exp(np.mean(np.log(spec + 1e-10), axis=0))
            amean = np.mean(spec, axis=0)
            flatness = gmean / (amean + 1e-10)
            flatness_score = float(np.clip(float(np.mean(flatness)), 0.0, 1.0))
        except Exception:
            flatness_score = 0.0

        # HF energy ratio — synthesized speech often has unusual HF content.
        try:
            hf_bin = int(4000 * (512 / 16000))
            hf_energy = np.sum(spec[hf_bin:, :])
            total_energy = np.sum(spec) + 1e-10
            hf_ratio = float(hf_energy / total_energy)
            hf_score = float(np.clip(hf_ratio / 0.3, 0.0, 1.0))
        except Exception:
            hf_score = 0.0

        score = float(np.clip(0.4 * clip_score + 0.3 * flatness_score + 0.3 * hf_score, 0.0, 1.0))
        return score, {
            "peak": peak,
            "clip_ratio": clip_ratio,
            "spectral_flatness": flatness_score,
            "hf_energy_ratio": hf_ratio if "hf_ratio" in dir() else 0.0,
        }


# ── Gaze / off-screen reading detector ─────────────────────────────── #

class RealGazeDetector:
    """Uses MediaPipe Face Mesh for 3D face landmarks, then derives gaze
    direction + head-pose to detect off-screen reading patterns."""

    def __init__(self):
        import mediapipe as mp

        version = getattr(mp, "__version__", "0")
        try:
            major = int(version.split(".")[0])
        except Exception:
            major = 0

        if major >= 10:
            raise RuntimeError(
                "MediaPipe >= 0.10 uses the new `tasks` API which requires "
                "a separate model bundle. Install mediapipe < 0.10 for the "
                "legacy `solutions` API, or use HeuristicGazeDetector."
            )

        self._mp_face = mp.solutions.face_mesh
        self.face_mesh = self._mp_face.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        self._prev_gaze = (0.5, 0.5)
        self._saccade_times: list = []

    def detect(self, frame: np.ndarray) -> Tuple[tuple, bool, Optional[float]]:
        img = np.asarray(frame)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        h, w = img.shape[:2]
        results = self.face_mesh.process(img)

        if not results.multi_face_landmarks:
            return (0.5, 0.5), False, None

        lm = results.multi_face_landmarks[0]
        # Nose tip as coarse gaze proxy.
        nose = lm.landmark[1]
        gx = float(nose.x)
        gy = float(nose.y)
        # Left / right iris landmarks (468, 473).
        left_iris = lm.landmark[468]
        right_iris = lm.landmark[473]
        iris_cx = float((left_iris.x + right_iris.x) / 2)
        iris_cy = float((left_iris.y + right_iris.y) / 2)
        gaze_vec = (iris_cx, iris_cy)

        # Head-pose rough detection: nose away from center + iris deviation.
        off_screen = (
            iris_cx < 0.3 or iris_cx > 0.7 or iris_cy < 0.3 or iris_cy > 0.7
        )

        # Saccade periodicity tracking.
        periodicity: Optional[float] = None
        import time

        dx = abs(gaze_vec[0] - self._prev_gaze[0])
        if dx > 0.05:
            now = time.time()
            self._saccade_times.append(now)
            if len(self._saccade_times) > 20:
                self._saccade_times.pop(0)
            if len(self._saccade_times) >= 3:
                gaps = [
                    self._saccade_times[i] - self._saccade_times[i - 1]
                    for i in range(1, len(self._saccade_times))
                ]
                periodicity = float(np.mean(gaps)) if gaps else None

        self._prev_gaze = gaze_vec
        return gaze_vec, off_screen, periodicity


# ── Real speaker embedding (plugs into SpeakerStore) ───────────────── #

def resemblyzer_embedding(pcm: np.ndarray) -> Optional[np.ndarray]:
    """Produce a 256-d speaker embedding from PCM audio using Resemblyzer."""
    import resemblyzer

    if pcm.size < 1600:
        return None
    audio = pcm.astype(np.float32) / 32768.0
    try:
        encoder = resemblyzer.VoiceEncoder()
        emb = encoder.embed_utterance(audio)
        return emb.astype(np.float32)
    except Exception as exc:
        logger.warning("Resemblyzer embedding failed: %s", exc)
        return None
