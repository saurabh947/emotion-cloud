# ─────────────────────────────────────────────────────────────────────────────
# emotion-cloud — production image
#
# Base: pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
#   Python 3.11, PyTorch 2.3, CUDA 12.1 — TorchServe installed via pip
#
# Build:
#   docker build -f deploy/docker/Dockerfile -t emotion-cloud:latest .
#
# Run (local, with GPU):
#   docker run --gpus all --env-file .env -p 50051:50051 emotion-cloud:latest
# ─────────────────────────────────────────────────────────────────────────────

FROM --platform=linux/amd64 pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# ── System dependencies ────────────────────────────────────────────────────────
# ffmpeg + libsndfile1: required by emotion2vec audio tower
# git: needed to pip-install SDK from GitHub
# google-cloud-cli: gsutil for GCS weight download
# curl: used in google-cloud-cli apt setup
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        curl \
        git \
        apt-transport-https \
        ca-certificates \
        gnupg \
        openjdk-17-jdk-headless \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] \
        https://packages.cloud.google.com/apt cloud-sdk main" \
        | tee /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update && apt-get install -y --no-install-recommends \
        google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

# ── App directory ──────────────────────────────────────────────────────────────
WORKDIR /opt/emotion-cloud

# ── Python dependencies ────────────────────────────────────────────────────────
# Copy requirements first so Docker layer cache survives code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
# Force-reinstall the SDK from GitHub to pick up the latest commit.
# Run in a separate layer so stable deps above stay cached.
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    "emotion-detection-action @ git+https://github.com/saurabh947/emotion-detection-action.git"

# ── Application source ─────────────────────────────────────────────────────────
COPY . .

# ── Generate gRPC stubs from proto ─────────────────────────────────────────────
# Outputs: emotion_pb2.py + emotion_pb2_grpc.py in the project root.
RUN python3 -m grpc_tools.protoc \
        -I proto \
        --python_out=. \
        --grpc_python_out=. \
        proto/emotion.proto

# ── TorchServe directory layout ────────────────────────────────────────────────
RUN mkdir -p /opt/ml/model-store /opt/ml/config

# Copy TorchServe config.
COPY deploy/docker/config.properties /opt/ml/config/config.properties


# ── Ports ──────────────────────────────────────────────────────────────────────
# 50051 → gRPC (exposed to the robot / external clients)
# 8080  → TorchServe inference (loopback only — not published externally)
# 8081  → TorchServe management (loopback only)
EXPOSE 50051

# ── Entrypoint ─────────────────────────────────────────────────────────────────
CMD ["python3", "main.py"]
