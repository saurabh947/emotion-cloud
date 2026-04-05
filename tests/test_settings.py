"""Unit tests for config/settings.py."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from config.settings import Settings


def test_defaults():
    s = Settings()
    assert s.grpc_port == 50051
    assert s.torchserve_host == "localhost"
    assert s.torchserve_inference_port == 8080
    assert s.model_name == "emotion-detector"
    assert s.device == "cuda"
    assert s.use_int8 is True
    assert s.weights_file_name == "phase2_last.pt"
    assert s.session_timeout_seconds == 300
    assert s.torchserve_startup_timeout == 120
    assert s.torchserve_startup_poll_interval == 2.0


def test_env_override(monkeypatch):
    monkeypatch.setenv("GRPC_PORT", "50052")
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("USE_INT8", "false")
    monkeypatch.setenv("SESSION_TIMEOUT_SECONDS", "60")
    s = Settings()
    assert s.grpc_port == 50052
    assert s.device == "cpu"
    assert s.use_int8 is False
    assert s.session_timeout_seconds == 60


def test_grpc_port_must_be_int(monkeypatch):
    monkeypatch.setenv("GRPC_PORT", "not-a-number")
    with pytest.raises(ValidationError):
        Settings()


def test_weights_gcs_uri_default():
    s = Settings()
    assert s.weights_gcs_uri.startswith("gs://")


def test_weights_local_path_default():
    s = Settings()
    assert s.weights_local_path == "/opt/ml/model-store/weights"
