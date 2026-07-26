"""Shared Modal infrastructure: image, volume, secret.

All jobs run on Modal (cloud GPUs). The image carries the Python deps and mounts
this package so the shared code is available in the remote container.

Set your own HuggingFace token as a Modal secret named `hf-token` that exports
`HF_TOKEN`, e.g.:

    modal secret create hf-token HF_TOKEN=hf_xxx
"""
from __future__ import annotations

import modal

HF_SECRET = "hf-token"        # rename to your Modal secret if different
VOLUME = "dondo-nanos"        # persistent volume for models/results

# Base deps only. `add_local_python_source` must be the LAST image step, so it is
# applied by IMAGE / with_onnx() after any pip installs.
_BASE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch", "transformers==4.56.0", "accelerate",
        "jiwer", "librosa", "soundfile", "numpy",
        "pandas", "pyarrow", "huggingface_hub",
    )
    .env({"HF_HOME": "/data/hf_cache"})
)

IMAGE = _BASE.add_local_python_source("dondo_nanos")

VOL = modal.Volume.from_name(VOLUME, create_if_missing=True)
SECRETS = [modal.Secret.from_name(HF_SECRET)]


def with_onnx() -> modal.Image:
    """Base image plus ONNX export/runtime deps (for quantize_onnx / export_frontend)."""
    return _BASE.pip_install("onnx", "onnxruntime", "onnxscript").add_local_python_source("dondo_nanos")
