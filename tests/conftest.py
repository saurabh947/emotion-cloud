"""
Root test configuration.

Automatically generates gRPC stubs from proto/emotion.proto before test
collection so that `import emotion_pb2` works without requiring `make proto`.
Requires grpcio-tools (included in requirements-dev.txt).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def pytest_configure(config) -> None:
    """Generate gRPC stubs if they are not already present."""
    root = Path(__file__).parent.parent
    if (root / "emotion_pb2.py").exists():
        return
    subprocess.run(
        [
            sys.executable, "-m", "grpc_tools.protoc",
            f"-I{root / 'proto'}",
            f"--python_out={root}",
            f"--grpc_python_out={root}",
            str(root / "proto" / "emotion.proto"),
        ],
        check=True,
        cwd=root,
    )
