"""
Audio preprocessing shared by every feature extractor.

Same steps for both corpora, so differences in the features come from the
speech, not from the recording format:
  1. read file, convert to mono (average channels; 5 RAVDESS files are stereo)
  2. resample to 16 kHz (RAVDESS is 48 kHz, CREMA-D 16 kHz; WavLM needs 16 kHz)
  3. trim leading/trailing silence: anything 30 dB below the loudest point counts as silence
     (RAVDESS clips carry ~1 s more silence)

Amplitude is NOT normalised here: loudness is an emotion cue. Corpus-level
gain differences are handled later by feature standardisation (Phase 6).
"""
from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import librosa

from src.data import PROJECT_ROOT

TARGET_SR = 16_000
TRIM_TOP_DB = 30  # frames quieter than (peak - 30 dB) at the edges are cut


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_audio(path: str | Path, target_sr: int = TARGET_SR,
               trim: bool = True, top_db: float = TRIM_TOP_DB) -> tuple[np.ndarray, int]:
    """Return (waveform float32 in [-1, 1], sample_rate)."""
    y, sr = sf.read(str(resolve(path)), dtype="float32", always_2d=True)
    y = y.mean(axis=1)                                   # mono
    if sr != target_sr:                                  # exact rational resampling
        g = gcd(sr, target_sr)
        y = resample_poly(y, target_sr // g, sr // g).astype(np.float32)
    if trim:
        trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        if len(trimmed) >= int(0.25 * target_sr):        # never trim a clip to almost nothing
            y = trimmed
    return y, target_sr
