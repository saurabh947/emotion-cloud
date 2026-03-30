"""
gRPC server — bidirectional streaming emotion inference.

Each call to StreamEmotion() represents one robot session (one Reachy Mini
connection).  The server:

  1. Forwards each incoming EmotionRequest frame to TorchServe over loopback HTTP.
  2. Yields EmotionResponse messages back to the robot as results arrive.
  3. On stream close (or error), sends a cleanup request to TorchServe so the
     session's EmotionDetector instance and rolling buffer are released.

Concurrency model:
  - grpc.aio (asyncio-native gRPC) so one thread can multiplex many streams.
  - TorchServe calls use httpx.AsyncClient with a connection pool.
  - TorchServe itself handles GPU work in its own process with its own threads.

Generated stubs (emotion_pb2, emotion_pb2_grpc) are produced at container build
time by the grpc_tools.protoc invocation in the Dockerfile.  At development time,
run `make proto` to regenerate them locally.
"""

from __future__ import annotations

import logging
from concurrent import futures

import grpc
import httpx

# Generated from proto/emotion.proto at build time.
# `make proto` regenerates these locally.
import emotion_pb2          # type: ignore[import]
import emotion_pb2_grpc     # type: ignore[import]

from api.schemas import InferenceRequest, InferenceResponse
from config import Settings

logger = logging.getLogger(__name__)


class EmotionServicer(emotion_pb2_grpc.EmotionDetectionServicer):
    """
    Implements the EmotionDetection gRPC service defined in emotion.proto.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._ts_base = (
            f"http://{settings.torchserve_host}"
            f":{settings.torchserve_inference_port}"
        )
        self._predictions_url = f"{self._ts_base}/predictions/{settings.model_name}"
        # Shared async HTTP client — reused across all streaming sessions.
        # Connection pool keeps loopback sockets open between frames.
        self._http = httpx.AsyncClient(
            base_url=self._ts_base,
            timeout=httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=50),
        )

    # ── StreamEmotion ─────────────────────────────────────────────────────────

    async def StreamEmotion(
        self,
        request_iterator,
        context: grpc.aio.ServicerContext,
    ):
        """
        Bidirectional streaming RPC.

        Runs for the lifetime of the robot's gRPC stream.  Each frame arrives
        as an EmotionRequest; each inference result is yielded as an
        EmotionResponse.
        """
        session_id: str | None = None

        try:
            async for req in request_iterator:
                session_id = req.session_id

                # Build TorchServe payload.
                ts_req = InferenceRequest(
                    session_id=req.session_id,
                    video_frame=req.video_frame.hex(),
                    frame_width=req.frame_width,
                    frame_height=req.frame_height,
                    audio_chunk=req.audio_chunk.hex() if req.audio_chunk else "",
                    audio_sample_rate=req.audio_sample_rate or 16000,
                    timestamp_ms=req.timestamp_ms,
                )

                try:
                    http_resp = await self._http.post(
                        f"/predictions/{self._settings.model_name}",
                        json=ts_req.model_dump(),
                    )
                except (httpx.ConnectError, httpx.TimeoutException) as exc:
                    logger.error("TorchServe unreachable for session %s: %s", session_id, exc)
                    yield emotion_pb2.EmotionResponse(
                        session_id=session_id,
                        error=f"Inference backend unavailable: {exc}",
                    )
                    continue

                if http_resp.status_code != 200:
                    logger.error(
                        "TorchServe HTTP %d for session %s: %s",
                        http_resp.status_code,
                        session_id,
                        http_resp.text[:200],
                    )
                    yield emotion_pb2.EmotionResponse(
                        session_id=session_id,
                        error=f"Backend error {http_resp.status_code}",
                    )
                    continue

                ts_resp = InferenceResponse.model_validate(http_resp.json())

                # Hard inference error from the handler.
                if ts_resp.error:
                    yield emotion_pb2.EmotionResponse(
                        session_id=session_id,
                        error=ts_resp.error,
                        timestamp_ms=ts_resp.timestamp_ms,
                    )
                    continue

                # Rolling buffer still warming up — suppress until ready.
                if ts_resp.buffering:
                    yield emotion_pb2.EmotionResponse(
                        session_id=session_id,
                        buffering=True,
                        timestamp_ms=ts_resp.timestamp_ms,
                    )
                    continue

                # Happy path — full result.
                yield emotion_pb2.EmotionResponse(
                    session_id=ts_resp.session_id,
                    dominant_emotion=ts_resp.dominant_emotion,
                    confidence_scores=ts_resp.confidence_scores,
                    stress=ts_resp.stress,
                    engagement=ts_resp.engagement,
                    arousal=ts_resp.arousal,
                    overall_confidence=ts_resp.overall_confidence,
                    embedding=ts_resp.embedding,
                    timestamp_ms=ts_resp.timestamp_ms,
                    buffering=False,
                    error="",
                )

        except Exception as exc:
            logger.error(
                "Unhandled error in StreamEmotion (session=%s): %s",
                session_id,
                exc,
                exc_info=True,
            )
            yield emotion_pb2.EmotionResponse(
                session_id=session_id or "",
                error=f"Internal server error: {exc}",
            )

        finally:
            # Always clean up the session's rolling buffer in TorchServe,
            # regardless of whether the stream closed cleanly or errored.
            if session_id:
                await self._cleanup_session(session_id)

    # ── HealthCheck ───────────────────────────────────────────────────────────

    async def HealthCheck(
        self,
        request,
        context: grpc.aio.ServicerContext,
    ) -> emotion_pb2.HealthResponse:
        try:
            resp = await self._http.get("/ping", timeout=3.0)
            model_status = "ready" if resp.status_code == 200 else "unavailable"
        except Exception as exc:
            logger.warning("Health check failed: %s", exc)
            model_status = "unreachable"

        return emotion_pb2.HealthResponse(
            healthy=(model_status == "ready"),
            model_status=model_status,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _cleanup_session(self, session_id: str) -> None:
        """Tell TorchServe to release the session's EmotionDetector."""
        try:
            await self._http.post(
                f"/predictions/{self._settings.model_name}",
                json={"session_id": session_id, "action": "cleanup"},
                timeout=5.0,
            )
            logger.info("Session cleaned up: %s", session_id)
        except Exception as exc:
            # Non-fatal — session reaper will evict it on idle timeout anyway.
            logger.warning("Session cleanup request failed for %s: %s", session_id, exc)

    async def aclose(self) -> None:
        """Gracefully close the shared HTTP client."""
        await self._http.aclose()


# ── Server bootstrap ──────────────────────────────────────────────────────────

async def serve(settings: Settings) -> None:
    """
    Start the gRPC server and block until shutdown.

    Called from main.py after TorchServe has confirmed readiness.
    """
    servicer = EmotionServicer(settings)

    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=settings.grpc_max_workers),
        options=[
            # Allow large video frames (max 10 MB per message).
            ("grpc.max_receive_message_length", 10 * 1024 * 1024),
            ("grpc.max_send_message_length", 10 * 1024 * 1024),
            # Keep-alive: detect dead robot connections within 30 s.
            ("grpc.keepalive_time_ms", 30_000),
            ("grpc.keepalive_timeout_ms", 10_000),
            ("grpc.keepalive_permit_without_calls", True),
        ],
    )

    emotion_pb2_grpc.add_EmotionDetectionServicer_to_server(servicer, server)

    listen_addr = f"[::]:{settings.grpc_port}"
    server.add_insecure_port(listen_addr)

    logger.info("gRPC server listening on %s", listen_addr)
    await server.start()

    try:
        await server.wait_for_termination()
    finally:
        await servicer.aclose()
        await server.stop(grace=5)
        logger.info("gRPC server stopped")
