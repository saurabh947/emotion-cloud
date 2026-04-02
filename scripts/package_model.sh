#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# package_model.sh
#
# Creates the TorchServe Model Archive (.mar) for emotion-detector.
#
# The .mar bundles:
#   - models/handler.py          (custom stateful handler)
#   - requirements.txt           (dependencies installed inside TorchServe worker)
#
# Weights are NOT baked into the .mar — they are downloaded from GCS at runtime
# and passed to the handler via model_dir (set in config.properties).
#
# Usage:
#   bash scripts/package_model.sh
#   bash scripts/package_model.sh --version 2   # bump version
#
# Output: /opt/ml/model-store/emotion-detector.mar  (or MODEL_STORE if set)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_STORE="${MODEL_STORE:-${REPO_ROOT}/model-store}"
MODEL_NAME="${MODEL_NAME:-emotion-detector}"
MODEL_VERSION="${1:-1}"

echo "==> Packaging ${MODEL_NAME} v${MODEL_VERSION}"
echo "    repo root  : ${REPO_ROOT}"
echo "    model store: ${MODEL_STORE}"

mkdir -p "${MODEL_STORE}"

torch-model-archiver \
    --model-name "${MODEL_NAME}" \
    --version "${MODEL_VERSION}" \
    --handler "${REPO_ROOT}/models/handler.py" \
    --extra-files "${REPO_ROOT}/requirements.txt" \
    --export-path "${MODEL_STORE}" \
    --force

echo "==> Created ${MODEL_STORE}/${MODEL_NAME}.mar"
