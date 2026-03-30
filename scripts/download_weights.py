#!/usr/bin/env python3
"""
Standalone weight download script.

Used by:
  - The Kubernetes init container (before TorchServe starts).
  - `make download-weights` during local development.

Reads WEIGHTS_GCS_URI and WEIGHTS_LOCAL_PATH from environment (or .env).
Skips the download if files already exist on disk.

Usage:
  python scripts/download_weights.py
  python scripts/download_weights.py --force   # re-download even if present
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Settings
from models.loader import download_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download emotion-cloud model weights from GCS")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download weights even if they already exist on disk",
    )
    args = parser.parse_args()

    settings = Settings()

    if args.force:
        dest = Path(settings.weights_local_path)
        if dest.exists():
            import shutil
            logger.info("--force: removing existing weights at %s", dest)
            shutil.rmtree(dest)

    logger.info(
        "Downloading weights | src=%s | dst=%s",
        settings.weights_gcs_uri,
        settings.weights_local_path,
    )

    try:
        download_weights(settings)
        logger.info("Done.")
    except Exception as exc:
        logger.critical("Download failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
