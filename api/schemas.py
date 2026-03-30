"""
Internal Pydantic schemas used between the gRPC server and TorchServe.

These are NOT the Protobuf wire types — those are generated from emotion.proto.
These schemas represent the JSON payloads sent over the loopback HTTP connection
between the gRPC server and TorchServe's inference endpoint.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── Request to TorchServe ─────────────────────────────────────────────────────

class InferenceRequest(BaseModel):
    """
    Payload posted to POST /predictions/{model_name}.

    Binary fields (video_frame, audio_chunk) are hex-encoded so they survive
    JSON serialization without a custom encoder.
    """
    session_id: str
    # Hex-encoded raw RGB bytes (H × W × 3, uint8).
    video_frame: str
    frame_width: int
    frame_height: int
    # Hex-encoded float32 PCM samples.  Empty string → no audio this frame.
    audio_chunk: str = ""
    audio_sample_rate: int = 16000
    timestamp_ms: int = 0
    # Set to "cleanup" when the gRPC stream closes to release session state.
    action: str = ""


# ── Response from TorchServe ──────────────────────────────────────────────────

class InferenceResponse(BaseModel):
    """
    Parsed TorchServe response body.

    buffering=True  →  model hasn't accumulated 16 frames yet; skip this result.
    error non-empty →  inference failed; propagate as EmotionResponse.error.
    """
    session_id: str
    dominant_emotion: str = ""
    confidence_scores: dict[str, float] = Field(default_factory=dict)
    stress: float = 0.0
    engagement: float = 0.0
    arousal: float = 0.0
    overall_confidence: float = 0.0
    # 512-dim embedding as a list of floats.
    embedding: list[float] = Field(default_factory=list)
    timestamp_ms: int = 0
    buffering: bool = False
    error: str = ""
    # Returned only for cleanup acknowledgements.
    action: str = ""
