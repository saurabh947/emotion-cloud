"""
Model loader: two responsibilities.

1. download_weights()   — pull model weights from GCS to local disk on pod
                          startup (before TorchServe reads them).

2. start_torchserve()   — launch the TorchServe process and wait until it
                          reports healthy on /ping.

Called once in sequence by main.py before the gRPC server starts accepting
connections.  Both functions raise on failure so the pod crashes fast and
Kubernetes restarts it — preferable to serving with a broken model.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import time
from pathlib import Path

import httpx

from config import Settings

logger = logging.getLogger(__name__)


# ── Weight download ───────────────────────────────────────────────────────────

def download_weights(settings: Settings) -> None:
    """
    Sync model weights from GCS to local disk using gsutil.

    Uses gsutil -m cp -r for parallel multi-threaded download.
    Skips if the local weights directory already contains files (e.g. warm
    restart after a pod reschedule onto a node with a warm local SSD).
    """
    dest = Path(settings.weights_local_path)
    dest.mkdir(parents=True, exist_ok=True)

    # Skip download if weights already present (e.g. node local cache).
    existing = list(dest.iterdir())
    if existing:
        logger.info(
            "Weights already present at %s (%d files) — skipping download",
            dest,
            len(existing),
        )
        return

    if not shutil.which("gsutil"):
        raise EnvironmentError(
            "gsutil not found. The Docker image must include google-cloud-cli."
        )

    logger.info("Downloading weights: %s → %s", settings.weights_gcs_uri, dest)
    t0 = time.monotonic()

    result = subprocess.run(
        ["gsutil", "-m", "cp", "-r", settings.weights_gcs_uri, str(dest)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"gsutil download failed (exit {result.returncode}):\n{result.stderr}"
        )

    elapsed = time.monotonic() - t0
    logger.info("Weights downloaded in %.1f s → %s", elapsed, dest)


# ── TorchServe startup ────────────────────────────────────────────────────────

def start_torchserve(settings: Settings) -> subprocess.Popen:
    """
    Launch TorchServe as a background process.

    Returns the Popen handle so main.py can monitor it.  Does NOT wait for
    readiness here — that is done by wait_for_torchserve() below so the
    caller can do other work (or just await) while TorchServe loads the model.
    """
    model_store = Path(settings.model_store_path)
    model_store.mkdir(parents=True, exist_ok=True)

    mar_name = f"{settings.model_name}.mar"

    cmd = [
        "torchserve",
        "--start",
        "--ncs",  # no config snapshots (cleaner for containers)
        "--model-store", str(model_store),
        "--models", f"{settings.model_name}={mar_name}",
        "--ts-config", settings.torchserve_config_path,
    ]

    logger.info("Starting TorchServe: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Pipe TorchServe stdout → our logger in a daemon thread.
    import threading

    def _pipe_logs():
        for line in proc.stdout:  # type: ignore[union-attr]
            logger.info("[torchserve] %s", line.rstrip())

    threading.Thread(target=_pipe_logs, daemon=True).start()
    return proc


async def wait_for_torchserve(settings: Settings) -> None:
    """
    Poll TorchServe's /ping endpoint until it returns 200 or timeout.

    Raises RuntimeError if TorchServe doesn't become healthy within
    settings.torchserve_startup_timeout seconds.
    """
    url = (
        f"http://{settings.torchserve_host}"
        f":{settings.torchserve_inference_port}/ping"
    )
    deadline = time.monotonic() + settings.torchserve_startup_timeout
    attempt = 0

    async with httpx.AsyncClient(timeout=3.0) as client:
        while time.monotonic() < deadline:
            attempt += 1
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    logger.info(
                        "TorchServe ready after %d poll(s) at %s",
                        attempt,
                        url,
                    )
                    return
            except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
                pass  # still starting up

            logger.debug(
                "Waiting for TorchServe... attempt %d/%d",
                attempt,
                int(settings.torchserve_startup_timeout / settings.torchserve_startup_poll_interval),
            )
            await asyncio.sleep(settings.torchserve_startup_poll_interval)

    raise RuntimeError(
        f"TorchServe did not become healthy within "
        f"{settings.torchserve_startup_timeout} s. Check logs above."
    )
