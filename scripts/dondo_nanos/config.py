"""Shared constants for the DONDO-nanos pipeline."""

# Parent model we compress (Khaya AI DONDO multilingual, Apache-2.0).
PARENT = "KhayaAI/w2v-bert-ada_ewe_fat_fra_gaa_nzi_twi_en"

# WAXAL languages we target, mapping the dataset folder -> display name.
# The folder name is what appears under data/ASR/<folder>/ in the HF dataset.
LANGS = {"aka": "Akan/Twi", "ewe": "Ewe"}

# HuggingFace dataset the audio + transcripts come from (CC-BY-SA-4.0 / CC-BY-4.0).
WAXAL_REPO = "google/WaxalNLP"

# Target audio sample rate.
SR = 16000


def blended(wer: float, cer: float) -> float:
    """The WAXAL metric: 0.5*WER + 0.5*CER (lower is better)."""
    return 0.5 * (wer + cer)
