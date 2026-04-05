"""
emotion-cloud entrypoint.

Boot sequence:
  1. Download model weights from GCS (skipped if already on disk).
  2. Launch TorchServe as a background process.
  3. Wait for TorchServe to report healthy on /ping.
  4. Start the gRPC server and serve until interrupted.

If any step fails the process exits non-zero so Docker restarts the container.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

from config import Settings
from models.loader import download_weights, start_torchserve, wait_for_torchserve
from api.server import serve

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    settings = Settings()

    logger.info(
        "emotion-cloud starting | grpc_port=%d | device=%s | int8=%s",
        settings.grpc_port,
        settings.device,
        settings.use_int8,
    )

    # Step 1: weights
    logger.info("Step 1/3 — downloading model weights")
    try:
        download_weights(settings)
    except Exception as exc:
        logger.critical("Weight download failed: %s", exc, exc_info=True)
        sys.exit(1)

    # Step 2: TorchServe
    logger.info("Step 2/3 — starting TorchServe")
    try:
        ts_proc = start_torchserve(settings)
    except Exception as exc:
        logger.critical("Failed to start TorchServe: %s", exc, exc_info=True)
        sys.exit(1)

    # Step 3: wait for TorchServe readiness
    logger.info(
        "Step 3/3 — waiting for TorchServe (timeout=%ds)",
        settings.torchserve_startup_timeout,
    )
    try:
        await wait_for_torchserve(settings)
    except RuntimeError as exc:
        logger.critical(str(exc))
        ts_proc.terminate()
        sys.exit(1)

    # Register graceful shutdown handlers.
    loop = asyncio.get_running_loop()

    def _shutdown(sig_name: str) -> None:
        logger.info("Received %s — shutting down", sig_name)
        ts_proc.terminate()
        # Cancel all running tasks; serve() will exit on its own.
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown, sig.name)

    # Hand off to the gRPC server — blocks until shutdown.
    logger.info("All systems go. Starting gRPC server.")
    await serve(settings)


if __name__ == "__main__":
    asyncio.run(main())
