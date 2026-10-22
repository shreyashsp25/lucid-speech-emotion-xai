"""
Simulated recording conditions ("channels") for microphone-robust training and testing.

Why. The training clips were recorded in studios with good microphones at a fixed distance.
A laptop or phone microphone in a normal room changes every feature: the frequency range is
narrower, the room adds echo, the level is lower or higher, and there is background noise.
The model also leans on loudness (Phase 7), so these changes move its answers (Phase 8, section 7
of report/09_simulation.md).

Training uses RANDOM channels (one per clip, seeded by the clip's file name). Testing uses three
FIXED conditions; "far_room" lies partly outside the training ranges on purpose (longer echo, lower
level, brown noise), to check that the model did not just learn the training recipe.

All functions take and return float32 mono waveforms at 16 kHz, after the usual loading
(resample + silence trim, src.audio.load_audio).
"""
from __future__ import annotations

import zlib

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

SR = 16_000


def _band(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    hi = min(hi, SR / 2 - 200)
    sos = butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    return sosfilt(sos, y)


def _room(y: np.ndarray, decay_s: float, wet: float, rng) -> np.ndarray:
    """Synthetic room: exponentially decaying noise impulse response (decay_s ~ RT60 / 6.9)."""
    n = int(max(decay_s * 7, 0.05) * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-t / decay_s)
    ir[0] = 0.0
    ir = ir / (np.sqrt(np.sum(ir ** 2)) + 1e-12)
    rev = fftconvolve(y, ir)[: len(y)]
    rev *= np.sqrt(np.sum(y ** 2) / (np.sum(rev ** 2) + 1e-12))
    return (1 - wet) * y + wet * rev


def _noise(n: int, colour: str, rng) -> np.ndarray:
    w = rng.standard_normal(n)
    if colour == "white":
        return w
    spec = np.fft.rfft(w)
    f = np.maximum(np.fft.rfftfreq(n, 1 / SR), 1.0)
    spec /= np.sqrt(f) if colour == "pink" else f          # pink 1/f power, brown 1/f^2 power
    return np.fft.irfft(spec, n)


def _add_noise(y: np.ndarray, snr_db: float, colour: str, rng) -> np.ndarray:
    nz = _noise(len(y), colour, rng)
    p_sig = np.mean(y ** 2) + 1e-12
    nz *= np.sqrt(p_sig / (10 ** (snr_db / 10)) / (np.mean(nz ** 2) + 1e-12))
    return y + nz


def _finish(y: np.ndarray, peak_ref: float, gain_db: float) -> np.ndarray:
    """Restore the original peak level, apply the gain change, clip like a real converter."""
    y = y / (np.max(np.abs(y)) + 1e-9) * peak_ref * 10 ** (gain_db / 20)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def apply(y: np.ndarray, band=(150.0, 6000.0), decay_s: float | None = 0.05, wet: float = 0.5,
          gain_db: float = 0.0, snr_db: float | None = 30.0, colour: str = "white", seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    peak = float(np.max(np.abs(y)) + 1e-9)
    z = _band(y.astype(np.float64), *band) if band else y.astype(np.float64)
    if decay_s:
        z = _room(z, decay_s, wet, rng)
    if snr_db is not None:
        z = _add_noise(z, snr_db, colour, rng)
    return _finish(z, peak, gain_db)


# --- random channel for training ------------------------------------------------------------
def random_params(key: str) -> dict:
    """Reproducible random channel for one clip (seeded by its file name)."""
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    return {
        "band": (float(rng.uniform(60, 400)), float(rng.uniform(3000, 8000))),
        "decay_s": float(rng.uniform(0.02, 0.12)) if rng.random() < 0.7 else None,   # RT60 about 0.15-0.8 s
        "wet": float(rng.uniform(0.2, 0.7)),
        "gain_db": float(rng.uniform(-20, 6)),
        "snr_db": float(rng.uniform(10, 40)) if rng.random() < 0.85 else None,
        "colour": str(rng.choice(["white", "pink"])),
        "seed": int(rng.integers(1 << 31)),
    }


def random_channel(y: np.ndarray, key: str) -> np.ndarray:
    return apply(y, **random_params(key))


# --- fixed test conditions ------------------------------------------------------------------
TEST_CONDITIONS = {
    # close to the simulated laptop microphone of report/09_simulation.md section 7
    "laptop": dict(band=(150, 6000), decay_s=0.3 / 6.9, wet=0.5, gain_db=-10, snr_db=30, colour="white"),
    # narrow-band phone line, pink noise, slightly quiet
    "phone": dict(band=(300, 3400), decay_s=None, gain_db=-6, snr_db=20, colour="pink"),
    # far from the microphone in an echoey room: longer echo, lower level and brown noise than in training
    "far_room": dict(band=(100, 7000), decay_s=1.0 / 6.9, wet=0.8, gain_db=-24, snr_db=15, colour="brown"),
}


def test_condition(y: np.ndarray, name: str, key: str) -> np.ndarray:
    return apply(y, seed=zlib.crc32((name + key).encode()), **TEST_CONDITIONS[name])
