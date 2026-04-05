#!/bin/bash
# VM startup script — runs automatically on every VM boot.
#
# On first boot:  /etc/emotion-cloud.env does not exist yet.
#                 `make vm-create` copies it after SSH is ready.
#                 This script exits early; vm-create triggers the container.
#
# On reboot:      env file is present → pulls latest image and starts container.
#
# Logs written to: /var/log/emotion-cloud-startup.log

set -euo pipefail
exec >> /var/log/emotion-cloud-startup.log 2>&1

TS() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
echo "[$(TS)] Startup script running"

ENV_FILE="/etc/emotion-cloud.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "[$(TS)] $ENV_FILE not found — VM not yet provisioned. Exiting."
  exit 0
fi

# Ensure Docker is installed (DLVM images do not pre-install it).
if ! command -v docker &>/dev/null; then
  echo "[$(TS)] Docker not found — installing..."
  curl -fsSL https://get.docker.com | sh
  systemctl enable docker
  systemctl start docker
  echo "[$(TS)] Docker installed"
fi

# Resolve project ID from GCE metadata server (no gcloud config needed).
PROJECT_ID=$(curl -sf \
  "http://metadata.google.internal/computeMetadata/v1/project/project-id" \
  -H "Metadata-Flavor: Google")

IMAGE="gcr.io/${PROJECT_ID}/emotion-cloud:latest"

# Configure Docker to authenticate to Container Registry.
gcloud auth configure-docker gcr.io --quiet

echo "[$(TS)] Pulling $IMAGE ..."
docker pull "$IMAGE"

# Remove existing container if present (idempotent).
docker stop emotion-cloud 2>/dev/null || true
docker rm   emotion-cloud 2>/dev/null || true

echo "[$(TS)] Starting emotion-cloud container..."
docker run -d \
  --name emotion-cloud \
  --restart=unless-stopped \
  --gpus all \
  -p 50051:50051 \
  --env-file "$ENV_FILE" \
  "$IMAGE"

echo "[$(TS)] emotion-cloud container started successfully"
