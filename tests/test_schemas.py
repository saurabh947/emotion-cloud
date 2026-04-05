"""Unit tests for api/schemas.py."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas import InferenceRequest, InferenceResponse


class TestInferenceRequest:
    def test_minimal(self):
        req = InferenceRequest(
            session_id="s1",
            video_frame="aabbcc",
            frame_width=640,
            frame_height=480,
        )
        assert req.session_id == "s1"
        assert req.audio_chunk == ""
        assert req.audio_sample_rate == 16000
        assert req.timestamp_ms == 0
        assert req.action == ""

    def test_with_audio(self):
        req = InferenceRequest(
            session_id="s1",
            video_frame="aabbcc",
            frame_width=640,
            frame_height=480,
            audio_chunk="deadbeef",
            audio_sample_rate=44100,
            timestamp_ms=1000,
        )
        assert req.audio_chunk == "deadbeef"
        assert req.audio_sample_rate == 44100
        assert req.timestamp_ms == 1000

    def test_cleanup_action(self):
        req = InferenceRequest(
            session_id="s1",
            video_frame="",
            frame_width=0,
            frame_height=0,
            action="cleanup",
        )
        assert req.action == "cleanup"

    def test_model_dump_roundtrip(self):
        req = InferenceRequest(
            session_id="abc",
            video_frame="ff00",
            frame_width=320,
            frame_height=240,
        )
        d = req.model_dump()
        req2 = InferenceRequest(**d)
        assert req2 == req

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            InferenceRequest(session_id="s1")  # missing video_frame, width, height


class TestInferenceResponse:
    def test_minimal_buffering(self):
        resp = InferenceResponse(session_id="s1", buffering=True, timestamp_ms=500)
        assert resp.buffering is True
        assert resp.dominant_emotion == ""
        assert resp.error == ""

    def test_full_result(self):
        resp = InferenceResponse(
            session_id="s1",
            dominant_emotion="happy",
            confidence_scores={"happy": 0.9, "neutral": 0.1},
            stress=0.2,
            engagement=0.8,
            arousal=0.5,
            overall_confidence=0.9,
            embedding=[0.1] * 512,
            timestamp_ms=1234,
            buffering=False,
            error="",
        )
        assert resp.dominant_emotion == "happy"
        assert len(resp.embedding) == 512
        assert resp.confidence_scores["happy"] == pytest.approx(0.9)

    def test_error_response(self):
        resp = InferenceResponse(session_id="s1", error="model crashed", timestamp_ms=0)
        assert resp.error == "model crashed"

    def test_model_validate_from_dict(self):
        data = {
            "session_id": "x",
            "dominant_emotion": "sad",
            "confidence_scores": {"sad": 0.7},
            "stress": 0.3,
            "engagement": 0.4,
            "arousal": 0.2,
            "overall_confidence": 0.7,
            "embedding": [0.0] * 512,
            "timestamp_ms": 9999,
            "buffering": False,
            "error": "",
        }
        resp = InferenceResponse.model_validate(data)
        assert resp.dominant_emotion == "sad"

    def test_cleanup_ack(self):
        resp = InferenceResponse(session_id="s1", action="cleanup_ack")
        assert resp.action == "cleanup_ack"
