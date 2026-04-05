"""
Model loader: two responsibilities.

1. download_weights()   — pull model weights from GCS to local disk on
                          container startup (before TorchServe reads them).

2. start_torchserve()   — launch the TorchServe process and wait until it
                          reports healthy on /ping.

Called once in sequence by main.py before the gRPC server starts accepting
connections.  Both functions raise on failure so the container exits and Docker
restarts it — preferable to serving with a broken model.
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
    Download model weights and .mar archive from GCS to local disk.

    Weights: gsutil rsync (mirrors contents of GCS prefix into local dir).
    .mar:    gsutil cp (single small file, ~5 KB).

    Uses rsync instead of cp -r to avoid the nested-directory bug where
    ``gsutil cp -r gs://…/v1/ /local/dir`` creates ``/local/dir/v1/``
    when the destination already exists.  rsync mirrors the *contents*
    of the source prefix directly into the destination directory.

    Skips weights if the local directory already contains files (warm restart).
    Always overwrites the .mar so a redeployed archive is picked up on restart.
    """
    if not shutil.which("gsutil"):
        raise EnvironmentError(
            "gsutil not found. The Docker image must include google-cloud-cli."
        )

    # ── weights ────────────────────────────────────────────────────────────────
    dest = Path(settings.weights_local_path)
    dest.mkdir(parents=True, exist_ok=True)

    # Only check for actual files (not subdirectories) to detect a valid cache.
    existing_files = [f for f in dest.iterdir() if f.is_file()]
    if existing_files:
        logger.info(
            "Weights already present at %s (%d files) — skipping download",
            dest,
            len(existing_files),
        )
    else:
        logger.info("Downloading weights: %s → %s", settings.weights_gcs_uri, dest)
        t0 = time.monotonic()
        result = subprocess.run(
            ["gsutil", "-m", "rsync", "-r", settings.weights_gcs_uri, str(dest)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"gsutil download failed (exit {result.returncode}):\n{result.stderr}"
            )
        elapsed = time.monotonic() - t0

        # Verify the expected checkpoint actually landed.
        checkpoint = dest / settings.weights_file_name
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Download completed but checkpoint not found at {checkpoint}. "
                f"Check WEIGHTS_GCS_URI and WEIGHTS_LOCAL_PATH."
            )
        logger.info("Weights downloaded in %.1f s → %s", elapsed, dest)

    # ── .mar archive ───────────────────────────────────────────────────────────
    mar_dest = Path(settings.model_store_path) / f"{settings.model_name}.mar"
    mar_dest.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading .mar: %s → %s", settings.mar_gcs_uri, mar_dest)
    result = subprocess.run(
        ["gsutil", "cp", settings.mar_gcs_uri, str(mar_dest)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gsutil .mar download failed (exit {result.returncode}):\n{result.stderr}"
        )
    logger.info(".mar downloaded → %s", mar_dest)


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
