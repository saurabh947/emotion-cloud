# emotion-cloud

Cloud inference service for the [emotion-detection-action](https://github.com/saurabh947/emotion-detection-action) SDK. Deploys the Two-Tower Multimodal Transformer (AffectNet ViT-B/16 + emotion2vec) as a gRPC streaming API on Google Cloud.

Designed to work with [reachy-emotion](https://github.com/saurabh947/reachy-emotion) — a conversational app running on [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) that uses Gemini for conversation and delegates to this service for real-time human emotion detection.

---

## How it fits together

```
Reachy Mini
└── reachy-emotion app
      ├── Gemini API          ← conversation
      └── [tool call]
            └── gRPC stream ──→ emotion-cloud (this repo, GKE)
                                  └── TorchServe + NVIDIA T4
                                        └── Two-Tower Transformer
                                              └── NeuralEmotionResult ──→ robot
```

When Gemini determines that emotion detection is needed mid-conversation, reachy-emotion opens a gRPC bidirectional stream to emotion-cloud, sends video frames and audio chunks in real time, and receives `NeuralEmotionResult` messages back continuously.

---

## Model

The underlying model is a **Two-Tower Multimodal Transformer**:

| Tower | Backbone | Pre-trained on |
|-------|----------|----------------|
| Video | AffectNet ViT-B/16 (`trpakov/vit-face-expression`) | 450K emotion-labeled faces |
| Audio | emotion2vec_base (`iic/emotion2vec_base`) | IEMOCAP, MSP-Podcast, RAVDESS, CREMA-D |

The towers are fused via bidirectional cross-attention. Output is a `NeuralEmotionResult` with:
- `dominant_emotion` — one of `angry · disgusted · fearful · happy · neutral · sad · surprised · unclear`
- `emotion_scores` — per-class softmax probabilities
- `metrics` — `stress`, `engagement`, `arousal` in [0, 1]
- `latent_embedding` — 512-dim GRU-smoothed vector for VLA integration
- `confidence` — max softmax score

---

## Architecture

- **Protocol**: gRPC bidirectional streaming (Protobuf). One stream = one robot session.
- **Inference server**: [TorchServe](https://pytorch.org/serve/) with a custom stateful handler.
- **Session state**: each connected stream gets its own `EmotionDetector` instance with an isolated 16-frame rolling buffer. No cross-session bleed.
- **Hardware**: NVIDIA T4 GPU on GKE Autopilot (`us-central1`).
- **Model weights**: stored in Google Cloud Storage, downloaded by an init container before the app starts.
- **Backbone weights**: downloaded from HuggingFace / FunASR on first session creation, cached on-pod.

---

## Project layout

```
proto/                  gRPC service definition (emotion.proto)
api/
  server.py             Async gRPC server (bidirectional streaming)
  schemas.py            Pydantic schemas for gRPC ↔ TorchServe boundary
models/
  handler.py            TorchServe stateful handler (per-session rolling buffer)
  loader.py             GCS weight download + TorchServe process management
config/
  settings.py           Pydantic env-driven settings
deploy/
  docker/               Dockerfile, docker-compose, TorchServe config
  k8s/                  GKE manifests (Deployment, Service, HPA, ServiceAccount)
scripts/
  download_weights.py   Pull weights from GCS (also used by init container)
  package_model.sh      Create emotion-detector.mar for TorchServe
main.py                 Entrypoint: weights → TorchServe → gRPC server
Makefile                All operations: proto, build, push, deploy, logs
```

---

## gRPC API

Defined in [`proto/emotion.proto`](proto/emotion.proto).

```protobuf
service EmotionDetection {
  rpc StreamEmotion(stream EmotionRequest) returns (stream EmotionResponse);
  rpc HealthCheck(HealthRequest) returns (HealthResponse);
}
```

**Session lifecycle:**
1. Client opens `StreamEmotion` with a `session_id`.
2. Server creates an `EmotionDetector` for that session.
3. Client streams frames at 10–30 fps. Server replies after buffer warms up (16 frames).
4. Client closes the stream → server tears down the session.

**Frame format:** RGB uint8, raw bytes, `frame_height × frame_width × 3`. The server flips to BGR internally before passing to the SDK.

**Audio format:** Optional. Raw float32 PCM, mono, 16 kHz.

---

## Quickstart

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with NVIDIA Container Toolkit (for GPU)
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) authenticated to your project
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- Model weights in GCS (see [Weights setup](#weights-setup))

### Local development (Docker)

```bash
cp .env.example .env          # fill in WEIGHTS_GCS_URI and credentials
make proto                    # generate gRPC stubs
make docker-run               # build + run with docker compose
```

The gRPC server is available at `localhost:50051`.

### Weights setup

```bash
# Create the GCS bucket
gsutil mb -l us-central1 -p YOUR_PROJECT gs://emotion-cloud-models

# Upload your fine-tuned checkpoint
gsutil cp /path/to/phase2_last.pt gs://emotion-cloud-models/weights/v1/phase2_last.pt
```

### Deploy to GKE

```bash
# One-time: create cluster and set up IAM
make create-cluster
# Follow Workload Identity setup in deploy/k8s/serviceaccount.yaml comments

# Package the model
make package-model

# Build, push, deploy
make deploy

# Verify
make k8s-status
make port-forward
```

---

## Configuration

All settings are read from environment variables. Copy `.env.example` to `.env` to get started.

| Variable | Default | Description |
|----------|---------|-------------|
| `GRPC_PORT` | `50051` | gRPC server port |
| `MODEL_NAME` | `emotion-detector` | TorchServe model name |
| `DEVICE` | `cuda` | Compute device (`cuda` / `cpu`) |
| `USE_INT8` | `true` | Enable INT8 dynamic quantization |
| `WEIGHTS_GCS_URI` | — | GCS path to weights directory |
| `WEIGHTS_FILE_NAME` | `phase2_last.pt` | Checkpoint filename inside weights dir |
| `SESSION_TIMEOUT_SECONDS` | `300` | Idle session reap timeout |

---

## Local project config

Project-specific values (GCP project ID, etc.) go in `.makerc.local` — this file is gitignored and never committed.

```bash
echo "PROJECT_ID := your-project-id" > .makerc.local
```

---

## License

[MIT](LICENSE)
