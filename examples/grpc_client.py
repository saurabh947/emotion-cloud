#!/usr/bin/env python3
"""
Manual end-to-end gRPC test client for emotion-cloud.

Connects to a running emotion-cloud server (local or GCE VM), opens a
StreamEmotion session, sends synthetic video + audio frames, and
prints the results.

Usage:
    # Against local docker-compose:
    python3 examples/grpc_client.py

    # Against the GCE VM (get IP with `make vm-ip`):
    python3 examples/grpc_client.py --host <VM-IP> --port 50051

    # Health check only:
    python3 examples/grpc_client.py --health-only

Prerequisites:
    pip3 install grpcio grpcio-tools numpy
    make proto   # generates emotion_pb2.py and emotion_pb2_grpc.py

Run from the repo root so the generated stubs are on the path.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid

import numpy as np

# Stubs must be on sys.path — run from repo root or add it explicitly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import grpc
    import emotion_pb2
    import emotion_pb2_grpc
except ImportError as e:
    print(f"Import error: {e}")
    print("Run `make proto` from the repo root, then retry.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Frame generators
# ---------------------------------------------------------------------------

def _synthetic_frame(width: int, height: int, frame_idx: int) -> bytes:
    """Generate a simple gradient frame that varies with frame_idx."""
    rng = np.random.default_rng(seed=frame_idx)
    frame = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    return frame.tobytes()


def _synthetic_audio(sample_rate: int = 16000, duration_ms: int = 100) -> bytes:
    """Generate a short silent audio chunk (float32 PCM)."""
    n_samples = int(sample_rate * duration_ms / 1000)
    audio = np.zeros(n_samples, dtype=np.float32)
    return audio.tobytes()


# ---------------------------------------------------------------------------
# Request iterator
# ---------------------------------------------------------------------------

def _request_stream(
    session_id: str,
    num_frames: int,
    width: int,
    height: int,
    fps: float,
    include_audio: bool,
):
    interval_s = 1.0 / fps
    for i in range(num_frames):
        frame_bytes = _synthetic_frame(width, height, i)
        audio_bytes = _synthetic_audio() if include_audio else b""
        yield emotion_pb2.EmotionRequest(
            session_id=session_id,
            video_frame=frame_bytes,
            frame_width=width,
            frame_height=height,
            audio_chunk=audio_bytes,
            audio_sample_rate=16000,
            timestamp_ms=int(time.time() * 1000),
        )
        time.sleep(interval_s)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def run_health_check(stub: emotion_pb2_grpc.EmotionDetectionStub) -> bool:
    print("\n── Health Check ─────────────────────────────────────────────────")
    try:
        resp = stub.HealthCheck(emotion_pb2.HealthRequest(), timeout=5)
        status = "✓ healthy" if resp.healthy else "✗ unhealthy"
        print(f"  {status}  |  model_status={resp.model_status}")
        return resp.healthy
    except grpc.RpcError as e:
        print(f"  ✗ RPC error: {e.code()} — {e.details()}")
        return False


# ---------------------------------------------------------------------------
# Streaming session
# ---------------------------------------------------------------------------

def run_stream(
    stub: emotion_pb2_grpc.EmotionDetectionStub,
    session_id: str,
    num_frames: int,
    width: int,
    height: int,
    fps: float,
    include_audio: bool,
) -> None:
    print(f"\n── StreamEmotion  session={session_id} ──────────────────────────")
    print(f"   {num_frames} frames @ {fps:.0f} fps  |  {width}×{height}  |  audio={'yes' if include_audio else 'no'}")
    print()

    received = 0
    buffering_count = 0
    errors = 0

    requests = _request_stream(session_id, num_frames, width, height, fps, include_audio)

    try:
        for resp in stub.StreamEmotion(requests, timeout=num_frames / fps + 30):
            received += 1
            if resp.error:
                errors += 1
                print(f"  [frame {received:03d}] ERROR: {resp.error}")
                continue
            if resp.buffering:
                buffering_count += 1
                print(f"  [frame {received:03d}] buffering... ({buffering_count}/16)")
                continue

            scores_str = "  ".join(
                f"{k}={v:.2f}" for k, v in sorted(resp.confidence_scores.items(), key=lambda x: -x[1])[:3]
            )
            print(
                f"  [frame {received:03d}] "
                f"{resp.dominant_emotion:<12}  conf={resp.overall_confidence:.2f}  "
                f"stress={resp.stress:.2f}  engage={resp.engagement:.2f}  "
                f"arousal={resp.arousal:.2f}  |  {scores_str}"
            )

    except grpc.RpcError as e:
        print(f"\n  ✗ Stream ended with RPC error: {e.code()} — {e.details()}")

    print(f"\n  Summary: {received} responses | {buffering_count} buffering | {errors} errors")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="emotion-cloud gRPC test client")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=50051, help="Server port (default: 50051)")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to send (default: 30)")
    parser.add_argument("--width", type=int, default=224, help="Frame width in pixels (default: 224)")
    parser.add_argument("--height", type=int, default=224, help="Frame height in pixels (default: 224)")
    parser.add_argument("--fps", type=float, default=10.0, help="Frames per second (default: 10)")
    parser.add_argument("--audio", action="store_true", help="Include synthetic audio chunks")
    parser.add_argument("--session-id", default=None, help="Session ID (default: random UUID)")
    parser.add_argument("--health-only", action="store_true", help="Only run health check, then exit")
    args = parser.parse_args()

    session_id = args.session_id or str(uuid.uuid4())[:8]
    target = f"{args.host}:{args.port}"

    print(f"Connecting to {target} ...")
    channel = grpc.insecure_channel(
        target,
        options=[
            ("grpc.max_receive_message_length", 10 * 1024 * 1024),
            ("grpc.max_send_message_length", 10 * 1024 * 1024),
        ],
    )
    stub = emotion_pb2_grpc.EmotionDetectionStub(channel)

    healthy = run_health_check(stub)
    if args.health_only:
        sys.exit(0 if healthy else 1)

    if not healthy:
        print("\nServer not healthy — stream may fail. Continuing anyway...")

    run_stream(
        stub=stub,
        session_id=session_id,
        num_frames=args.frames,
        width=args.width,
        height=args.height,
        fps=args.fps,
        include_audio=args.audio,
    )

    channel.close()


if __name__ == "__main__":
    main()
