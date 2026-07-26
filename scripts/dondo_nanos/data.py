"""WAXAL data loading, straight from the HuggingFace parquet files.

We read the parquet with pandas over the `hf://` filesystem rather than through
`datasets`, which avoids the torchcodec audio-decode dependency and a stray
index-column cast error in some WAXAL configs.
"""
from __future__ import annotations

from .audio import decode_bytes, norm
from .config import SR, WAXAL_REPO

_TCOLS = ("transcription", "text", "sentence")


def _glob(folder: str, split: str):
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    pat = f"datasets/{WAXAL_REPO}/data/ASR/{folder}/{folder}-{split}-*.parquet"
    return fs, sorted(fs.glob(pat))


def load_split(folder: str, split: str, n: int,
               min_s: float = 0.1, max_s: float = 20.0):
    """Load up to `n` (audio, text) pairs from a WAXAL split.

    Filters out empty transcripts and clips outside [min_s, max_s] seconds.
    Returns a list of (np.float32 array, normalized text).
    """
    import pandas as pd

    fs, files = _glob(folder, split)
    rows = []
    for fp in files:
        df = pd.read_parquet("hf://" + fp)
        tcol = next((c for c in _TCOLS if c in df.columns), None)
        for _, r in df.iterrows():
            t = norm(r[tcol] if tcol else "")
            if not t:
                continue
            a = r["audio"]
            b = a["bytes"] if a.get("bytes") is not None else open(a["path"], "rb").read()
            y = decode_bytes(b)
            if not (min_s * SR <= len(y) <= max_s * SR):
                continue
            rows.append((y, t))
            if len(rows) >= n:
                return rows
    return rows


def load_val(folder: str, n: int, **kw):
    """Convenience wrapper for the validation split."""
    return load_split(folder, "validation", n, **kw)
