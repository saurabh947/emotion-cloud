"""
Unit tests for models/handler.py.

TorchServe and the emotion-detection-action SDK are mocked so these tests run
in CI without a GPU or heavyweight dependencies.
"""
from __future__ import annotations

import json
import sys
import time
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Minimal stubs so `from ts.torch_handler.base_handler import BaseHandler`
# and `import torch` work without installing torchserve / pytorch.
# ---------------------------------------------------------------------------

def _install_stub_modules():
    # torch
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = MagicMock()
    torch_mod.cuda.is_available = MagicMock(return_value=False)
    sys.modules.setdefault("torch", torch_mod)

    # ts.torch_handler.base_handler
    ts_mod = types.ModuleType("ts")
    handler_pkg = types.ModuleType("ts.torch_handler")
    base_mod = types.ModuleType("ts.torch_handler.base_handler")

    class BaseHandler:
        pass

    base_mod.BaseHandler = BaseHandler
    sys.modules.setdefault("ts", ts_mod)
    sys.modules.setdefault("ts.torch_handler", handler_pkg)
    sys.modules.setdefault("ts.torch_handler.base_handler", base_mod)

    # emotion_detection_action
    sdk_mod = types.ModuleType("emotion_detection_action")
    sys.modules.setdefault("emotion_detection_action", sdk_mod)


_install_stub_modules()

# Now safe to import the handler.
from models.handler import EmotionHandler  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_context(model_dir: str = "/tmp/weights") -> MagicMock:
    ctx = MagicMock()
    ctx.system_properties = {"model_dir": model_dir}
    return ctx


def _make_handler(tmp_path) -> EmotionHandler:
    """Return an initialized EmotionHandler with a mocked detector class."""
    weights = tmp_path / "phase2_last.pt"
    weights.touch()

    handler = EmotionHandler()

    mock_detector_cls = MagicMock()
    mock_config_cls = MagicMock()

    with (
        patch("models.handler.os.environ.get", side_effect=lambda k, d="": {
            "WEIGHTS_FILE_NAME": "phase2_last.pt",
            "USE_INT8": "false",
            "SESSION_TIMEOUT_SECONDS": "300",
        }.get(k, d)),
        patch("models.handler.os.path.isfile", return_value=True),
        patch("models.handler.os.makedirs"),
    ):
        ctx = _make_context(str(tmp_path))
        handler._detector_cls = mock_detector_cls
        handler._config_cls = mock_config_cls
        # Manually replicate what initialize() sets after the import block.
        handler._config_kwargs = dict(
            two_tower_device="cpu",
            two_tower_model_path=str(weights),
            cache_dir=str(tmp_path / ".backbone_cache"),
        )
        handler._use_int8 = False
        handler._initialized = True

    return handler


def _rgb_frame_hex(h: int = 4, w: int = 4) -> str:
    return (np.zeros((h, w, 3), dtype=np.uint8)).tobytes().hex()


# ---------------------------------------------------------------------------
# Tests: preprocess
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_dict_body_passthrough(self, tmp_path):
        handler = _make_handler(tmp_path)
        data = [{"body": {"session_id": "s1", "video_frame": "aabb"}}]
        result = handler.preprocess(data)
        assert result == [{"session_id": "s1", "video_frame": "aabb"}]

    def test_bytes_body_decoded(self, tmp_path):
        handler = _make_handler(tmp_path)
        payload = json.dumps({"session_id": "s2", "action": "cleanup"}).encode()
        result = handler.preprocess([{"body": payload}])
        assert result[0]["action"] == "cleanup"

    def test_raw_dict_no_body_key(self, tmp_path):
        handler = _make_handler(tmp_path)
        result = handler.preprocess([{"session_id": "s3"}])
        assert result[0]["session_id"] == "s3"


# ---------------------------------------------------------------------------
# Tests: session management
# ---------------------------------------------------------------------------

class TestSessionManagement:
    def test_session_created_on_first_access(self, tmp_path):
        handler = _make_handler(tmp_path)
        mock_det = MagicMock()
        handler._detector_cls.return_value = mock_det

        det = handler._get_or_create_session("abc")

        assert det is mock_det
        mock_det.initialize.assert_called_once()
        assert handler.active_session_count == 1

    def test_same_session_reused(self, tmp_path):
        handler = _make_handler(tmp_path)
        mock_det = MagicMock()
        handler._detector_cls.return_value = mock_det

        det1 = handler._get_or_create_session("abc")
        det2 = handler._get_or_create_session("abc")

        assert det1 is det2
        assert handler._detector_cls.call_count == 1

    def test_different_sessions_isolated(self, tmp_path):
        handler = _make_handler(tmp_path)
        det_a, det_b = MagicMock(), MagicMock()
        handler._detector_cls.side_effect = [det_a, det_b]

        handler._get_or_create_session("a")
        handler._get_or_create_session("b")

        assert handler.active_session_count == 2

    def test_remove_session(self, tmp_path):
        handler = _make_handler(tmp_path)
        handler._detector_cls.return_value = MagicMock()
        handler._get_or_create_session("x")
        handler._remove_session("x")
        assert handler.active_session_count == 0

    def test_remove_nonexistent_session_is_noop(self, tmp_path):
        handler = _make_handler(tmp_path)
        handler._remove_session("no-such-session")  # should not raise

    def test_int8_quantize_called_when_enabled(self, tmp_path):
        handler = _make_handler(tmp_path)
        handler._use_int8 = True
        mock_det = MagicMock()
        handler._detector_cls.return_value = mock_det

        handler._get_or_create_session("q")
        mock_det.quantize.assert_called_once_with("dynamic")

    def test_int8_quantize_not_called_when_disabled(self, tmp_path):
        handler = _make_handler(tmp_path)
        handler._use_int8 = False
        mock_det = MagicMock()
        handler._detector_cls.return_value = mock_det

        handler._get_or_create_session("q")
        mock_det.quantize.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: postprocess
# ---------------------------------------------------------------------------

class TestPostprocess:
    def test_cleanup_ack(self, tmp_path):
        handler = _make_handler(tmp_path)
        out = handler.postprocess([("s1", None, "cleanup")])
        data = json.loads(out[0])
        assert data["action"] == "cleanup_ack"

    def test_buffering_response(self, tmp_path):
        handler = _make_handler(tmp_path)
        out = handler.postprocess([("s1", None, 1000)])
        data = json.loads(out[0])
        assert data["buffering"] is True
        assert data["timestamp_ms"] == 1000

    def test_error_response(self, tmp_path):
        handler = _make_handler(tmp_path)
        exc = RuntimeError("boom")
        out = handler.postprocess([("s1", exc, 2000)])
        data = json.loads(out[0])
        assert "boom" in data["error"]
        assert data["timestamp_ms"] == 2000

    def test_full_result(self, tmp_path):
        handler = _make_handler(tmp_path)

        mock_result = MagicMock()
        mock_result.dominant_emotion = "happy"
        mock_result.emotion_scores = {"happy": 0.9, "neutral": 0.1}
        mock_result.confidence = 0.9
        mock_result.metrics = {"stress": 0.1, "engagement": 0.8, "arousal": 0.5}
        mock_result.latent_embedding = [0.1] * 512
        mock_result.timestamp = 1.5  # seconds

        out = handler.postprocess([("s1", mock_result, 1500)])
        data = json.loads(out[0])

        assert data["dominant_emotion"] == "happy"
        assert data["confidence_scores"]["happy"] == pytest.approx(0.9)
        assert data["stress"] == pytest.approx(0.1)
        assert data["engagement"] == pytest.approx(0.8)
        assert data["overall_confidence"] == pytest.approx(0.9)
        assert len(data["embedding"]) == 512
        assert data["timestamp_ms"] == 1500
        assert data["buffering"] is False
        assert data["error"] == ""

    def test_missing_metrics_default_to_zero(self, tmp_path):
        handler = _make_handler(tmp_path)

        mock_result = MagicMock()
        mock_result.dominant_emotion = "neutral"
        mock_result.emotion_scores = {}
        mock_result.confidence = 0.5
        mock_result.metrics = {}  # no stress/engagement/arousal
        mock_result.latent_embedding = []
        mock_result.timestamp = 2.0

        out = handler.postprocess([("s1", mock_result, 2000)])
        data = json.loads(out[0])

        assert data["stress"] == 0.0
        assert data["engagement"] == 0.0
        assert data["arousal"] == 0.0


# ---------------------------------------------------------------------------
# Tests: inference → cleanup path
# ---------------------------------------------------------------------------

class TestInferenceCleanup:
    def test_cleanup_request_removes_session(self, tmp_path):
        handler = _make_handler(tmp_path)
        handler._detector_cls.return_value = MagicMock()
        handler._get_or_create_session("s1")
        assert handler.active_session_count == 1

        results = handler.inference([{"action": "cleanup", "session_id": "s1"}])
        assert results[0] == ("s1", None, "cleanup")
        assert handler.active_session_count == 0


# ---------------------------------------------------------------------------
# Tests: RGB → BGR flip in inference
# ---------------------------------------------------------------------------

class TestRGBToBGRFlip:
    def test_process_frame_receives_bgr(self, tmp_path):
        handler = _make_handler(tmp_path)
        mock_det = MagicMock()
        mock_det.process_frame.return_value = None  # triggers buffering path
        handler._detector_cls.return_value = mock_det
        handler._get_or_create_session("flip")

        h, w = 4, 4
        # Build a frame where R=1, G=2, B=3 so we can verify the flip.
        frame_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        frame_rgb[:, :, 0] = 10  # R
        frame_rgb[:, :, 1] = 20  # G
        frame_rgb[:, :, 2] = 30  # B
        frame_hex = frame_rgb.tobytes().hex()

        req = {
            "session_id": "flip",
            "video_frame": frame_hex,
            "frame_width": w,
            "frame_height": h,
            "timestamp_ms": 0,
        }
        handler.inference([req])

        called_frame = mock_det.process_frame.call_args[0][0]
        # After BGR flip: channel 0 = B = 30, channel 2 = R = 10
        assert called_frame[0, 0, 0] == 30  # B
        assert called_frame[0, 0, 2] == 10  # R
