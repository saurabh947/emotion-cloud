"""
TorchServe custom handler for the emotion-detection-action Two-Tower model.

Stateful design: each robot session gets its own EmotionDetector instance,
which owns the 16-frame rolling buffer and temporal GRU state.  Sessions are
keyed by session_id (a string set by the reachy-emotion client on every frame).

TorchServe lifecycle (called by the model server process):
  initialize()  → load weights once, resolve the checkpoint path
  handle()      → per-batch entry point → preprocess → inference → postprocess
  cleanup()     → called on explicit "action": "cleanup" request

Session reaping: a background thread removes sessions idle for longer than
SESSION_TIMEOUT_SECONDS (default 300 s) to prevent unbounded memory growth.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

import numpy as np
import torch
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

SESSION_TIMEOUT_SECONDS = int(os.environ.get("SESSION_TIMEOUT_SECONDS", 300))
_REAPER_INTERVAL_SECONDS = 60


class EmotionHandler(BaseHandler):
    """
    TorchServe handler that wraps emotion-detection-action's EmotionDetector.

    One EmotionDetector per session_id keeps the rolling buffer and GRU state
    isolated — concurrent robot streams don't bleed into each other.
    """

    def __init__(self) -> None:
        super().__init__()
        # session_id → {"detector": EmotionDetector, "last_seen": float}
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._initialized = False
        self._detector_cls = None
        self._config_cls = None
        self._config_kwargs: dict[str, Any] = {}
        self._use_int8: bool = False

        # Start background reaper.
        reaper = threading.Thread(target=self._reap_idle_sessions, daemon=True)
        reaper.start()

    # ── TorchServe lifecycle ──────────────────────────────────────────────────

    def initialize(self, context) -> None:
        """Called once by TorchServe when the model worker starts."""
        properties = context.system_properties
        model_dir = properties.get("model_dir", "/opt/ml/model-store/weights")

        # Resolve compute device.
        if torch.cuda.is_available():
            device = "cuda"
        else:
            logger.warning("CUDA not available — falling back to CPU inference")
            device = "cpu"

        # SDK uses "Config", not "EmotionDetectorConfig".
        try:
            from emotion_detection_action import EmotionDetector, Config
            self._detector_cls = EmotionDetector
            self._config_cls = Config
        except ImportError as exc:
            raise RuntimeError(
                "emotion-detection-action SDK not found. "
                "Ensure it is listed in requirements.txt and baked into the .mar."
            ) from exc

        # Full path to the fine-tuned Two-Tower checkpoint.
        weights_file = os.environ.get("WEIGHTS_FILE_NAME", "phase2_last.pt")
        checkpoint_path = os.path.join(model_dir, weights_file)

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                f"Check WEIGHTS_FILE_NAME and that the init container ran successfully."
            )

        # Cache dir for HuggingFace + FunASR backbone downloads.
        # Stored inside model_dir so it survives on the same pod across
        # process restarts (though not pod reschedules on emptyDir).
        cache_dir = os.path.join(model_dir, ".backbone_cache")
        os.makedirs(cache_dir, exist_ok=True)

        self._use_int8 = os.environ.get("USE_INT8", "true").lower() == "true"

        # Config kwargs used to construct a new detector per session.
        # two_tower_model_path is a full file path (not a directory).
        self._config_kwargs = dict(
            two_tower_device=device,
            two_tower_model_path=checkpoint_path,
            cache_dir=cache_dir,
        )

        self._initialized = True
        logger.info(
            "EmotionHandler initialized | device=%s | checkpoint=%s | int8=%s",
            device,
            checkpoint_path,
            self._use_int8,
        )

    def handle(self, data: list[dict], context) -> list[str]:
        """TorchServe entry point — receives a micro-batch, returns JSON strings."""
        if not self._initialized:
            raise RuntimeError("Handler not initialized")

        requests = self.preprocess(data)
        outputs = self.inference(requests)
        return self.postprocess(outputs)

    # ── Handler phases ────────────────────────────────────────────────────────

    def preprocess(self, data: list[dict]) -> list[dict]:
        """Deserialize raw TorchServe request bodies into typed dicts."""
        parsed = []
        for item in data:
            body = item.get("body", item)
            if isinstance(body, (bytes, bytearray)):
                body = json.loads(body.decode("utf-8"))
            parsed.append(body)
        return parsed

    def inference(self, requests: list[dict]) -> list[tuple[str, Any, Any]]:
        """Run each request through the appropriate session's detector."""
        results = []
        for req in requests:
            # Explicit session cleanup (sent when the gRPC stream closes).
            if req.get("action") == "cleanup":
                self._remove_session(req.get("session_id", ""))
                results.append((req.get("session_id", ""), None, "cleanup"))
                continue

            session_id: str = req["session_id"]
            detector = self._get_or_create_session(session_id)

            # Decode video frame: hex-encoded raw RGB bytes → np.ndarray (H, W, 3).
            frame_hex: str = req["video_frame"]
            h: int = req["frame_height"]
            w: int = req["frame_width"]
            frame_rgb = np.frombuffer(bytes.fromhex(frame_hex), dtype=np.uint8).reshape(h, w, 3)

            # SDK's process_frame() expects BGR (OpenCV convention).
            # The proto contract sends RGB, so we flip here on the server side.
            frame_bgr = frame_rgb[..., ::-1].copy()

            # Decode audio (optional): hex-encoded float32 PCM → np.ndarray (N,).
            audio: np.ndarray | None = None
            if req.get("audio_chunk"):
                audio = np.frombuffer(bytes.fromhex(req["audio_chunk"]), dtype=np.float32)

            timestamp_ms: int = req.get("timestamp_ms", int(time.time() * 1000))
            # SDK uses seconds for its timestamp parameter.
            timestamp_s: float = timestamp_ms / 1000.0

            try:
                result = detector.process_frame(frame_bgr, audio, timestamp=timestamp_s)
                results.append((session_id, result, timestamp_ms))
            except Exception as exc:
                logger.error("Inference error for session %s: %s", session_id, exc, exc_info=True)
                results.append((session_id, exc, timestamp_ms))

        return results

    def postprocess(self, outputs: list[tuple]) -> list[str]:
        """Serialize NeuralEmotionResult to JSON strings."""
        serialized = []
        for session_id, result, meta in outputs:
            # Cleanup acknowledgement.
            if meta == "cleanup":
                serialized.append(json.dumps({"session_id": session_id, "action": "cleanup_ack"}))
                continue

            timestamp_ms: int = meta

            # Propagate inference errors as error responses.
            if isinstance(result, Exception):
                serialized.append(json.dumps({
                    "session_id": session_id,
                    "error": str(result),
                    "timestamp_ms": timestamp_ms,
                }))
                continue

            # None means the frame buffer is completely empty (first call edge case).
            if result is None:
                serialized.append(json.dumps({
                    "session_id": session_id,
                    "buffering": True,
                    "timestamp_ms": timestamp_ms,
                }))
                continue

            # NeuralEmotionResult field mapping (SDK → proto):
            #   result.emotion_scores       → confidence_scores
            #   result.confidence           → overall_confidence
            #   result.latent_embedding     → embedding (already a list[float])
            #   result.metrics["stress"]    → stress
            #   result.metrics["engagement"]→ engagement
            #   result.metrics["arousal"]   → arousal
            #   result.timestamp (seconds)  → timestamp_ms (milliseconds)
            serialized.append(json.dumps({
                "session_id": session_id,
                "dominant_emotion": result.dominant_emotion,
                "confidence_scores": result.emotion_scores,
                "stress": float(result.metrics.get("stress", 0.0)),
                "engagement": float(result.metrics.get("engagement", 0.0)),
                "arousal": float(result.metrics.get("arousal", 0.0)),
                "overall_confidence": float(result.confidence),
                "embedding": result.latent_embedding,
                "timestamp_ms": int(result.timestamp * 1000),
                "buffering": False,
                "error": "",
            }))

        return serialized

    # ── Session management ────────────────────────────────────────────────────

    def _get_or_create_session(self, session_id: str):
        with self._lock:
            if session_id not in self._sessions:
                config = self._config_cls(**self._config_kwargs)
                detector = self._detector_cls(config)
                # initialize() loads backbone weights (may download from HuggingFace
                # on first call if not cached). Subsequent sessions reuse the cache.
                detector.initialize()
                if self._use_int8:
                    detector.quantize("dynamic")
                self._sessions[session_id] = {
                    "detector": detector,
                    "last_seen": time.monotonic(),
                }
                logger.info(
                    "Session created: %s | total active: %d",
                    session_id,
                    len(self._sessions),
                )
            else:
                self._sessions[session_id]["last_seen"] = time.monotonic()

            return self._sessions[session_id]["detector"]

    def _remove_session(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(
                    "Session removed: %s | total active: %d",
                    session_id,
                    len(self._sessions),
                )

    def _reap_idle_sessions(self) -> None:
        """Background thread: evict sessions idle longer than SESSION_TIMEOUT_SECONDS."""
        while True:
            time.sleep(_REAPER_INTERVAL_SECONDS)
            cutoff = time.monotonic() - SESSION_TIMEOUT_SECONDS
            with self._lock:
                stale = [sid for sid, s in self._sessions.items() if s["last_seen"] < cutoff]
                for sid in stale:
                    del self._sessions[sid]
                    logger.info("Session reaped (idle timeout): %s", sid)

    @property
    def active_session_count(self) -> int:
        with self._lock:
            return len(self._sessions)
