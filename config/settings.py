from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All configuration is read from environment variables (or .env).
    Deployed via /etc/emotion-cloud.env on the GCE VM (Docker --env-file).
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── gRPC server ────────────────────────────────────────────────────────────
    grpc_port: int = 50051
    # Thread-pool workers for the gRPC server executor.
    # Each active streaming session occupies one worker for the duration.
    grpc_max_workers: int = 20

    # ── TorchServe ─────────────────────────────────────────────────────────────
    torchserve_host: str = "localhost"
    torchserve_inference_port: int = 8080
    torchserve_management_port: int = 8081
    # Name under which the model is registered in TorchServe.
    model_name: str = "emotion-detector"
    # Path TorchServe reads .mar files from.
    model_store_path: str = "/opt/ml/model-store"
    # TorchServe config file.
    torchserve_config_path: str = "/opt/ml/config/config.properties"

    # ── Model ──────────────────────────────────────────────────────────────────
    # Compute device passed to the emotion-detection-action Config (two_tower_device).
    device: str = "cuda"
    # Enable INT8 dynamic quantization via detector.quantize("dynamic").
    # Halves VRAM usage, ~15% latency reduction.
    use_int8: bool = True
    # Filename of the fine-tuned checkpoint inside weights_local_path.
    # Full path = weights_local_path / weights_file_name.
    weights_file_name: str = "phase2_last.pt"

    # ── Session management ─────────────────────────────────────────────────────
    # Seconds of inactivity before a session is reaped from the handler registry.
    session_timeout_seconds: int = 300

    # ── GCS / model weights ────────────────────────────────────────────────────
    # gs://bucket/path/to/weights/ — trailing slash required.
    weights_gcs_uri: str = "gs://emotion-cloud-models/weights/v1/"
    # Local directory where weights land after download.
    weights_local_path: str = "/opt/ml/model-store/weights"
    # TorchServe model archive in GCS — downloaded to model_store_path at startup.
    mar_gcs_uri: str = "gs://emotion-cloud-models/model-store/emotion-detector.mar"

    # ── Startup ────────────────────────────────────────────────────────────────
    # How long (seconds) to wait for TorchServe to become healthy after launch.
    torchserve_startup_timeout: int = 120
    # How often (seconds) to poll /ping during startup.
    torchserve_startup_poll_interval: float = 2.0
