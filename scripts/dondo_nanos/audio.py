"""Audio decoding and text normalization, shared across all jobs."""
from __future__ import annotations
import io
import re

import numpy as np

from .config import SR


def norm(s: str | None) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Matches the normalization used by the WAXAL metric so WER/CER are comparable.
    """
    s = (s or "").lower()
    s = re.sub(r"[^\w\s']", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def decode_bytes(b: bytes) -> np.ndarray:
    """Decode encoded audio bytes (mp3/wav) to a mono float32 array at SR."""
    import soundfile as sf

    y, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)
    if sr != SR:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y.astype(np.float32)
