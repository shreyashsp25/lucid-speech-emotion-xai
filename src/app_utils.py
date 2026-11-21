"""
Helpers for the Phase 9 app (app/streamlit_app.py). No Streamlit code here, so
everything can be tested from a notebook or a script.

  * readable names for the 88 eGeMAPS features
  * signal views: waveform envelope and spectrogram
  * where a clip's feature values sit in the training distribution (percentiles)
  * controlled perturbations (gain, pitch shift, noise) for the robustness simulation
  * evidence split of the fusion model (eGeMAPS part vs WavLM part)
  * loaders for the saved results of Phases 6-9
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from src.audio import TARGET_SR
from src.data import PROJECT_ROOT, META_DIR
from src.explain import feature_group
from src.models import CLASSES

RESULTS = PROJECT_ROOT / "results"
MODEL_DIR = PROJECT_ROOT / "models"
HELDOUT_PATH = MODEL_DIR / "heldout_fold0.joblib"
NAME = {"ravdess": "RAVDESS", "cremad": "CREMA-D"}

# One colour per emotion, used everywhere in the app
EMO_COLOUR = {"neutral": "#8A8F98", "happy": "#E9A100", "sad": "#2A78D6",
              "angry": "#D64545", "fearful": "#7B5BD6", "disgust": "#1BAF7A"}
GROUP_COLOUR = {"Energy": "#EB6834", "Frequency": "#2A78D6", "Spectral": "#1BAF7A", "Temporal": "#7B5BD6"}


# ---------------------------------------------------------------------------
# Readable eGeMAPS names
# ---------------------------------------------------------------------------
_BASE = [
    (r"^F0semitoneFrom27\.5Hz_sma3nz", "Pitch"),
    (r"^loudness_sma3", "Loudness"),
    (r"^spectralFlux_sma3", "Spectral flux"),
    (r"^spectralFluxV_sma3nz", "Spectral flux (voiced)"),
    (r"^spectralFluxUV_sma3nz", "Spectral flux (unvoiced)"),
    (r"^mfcc(\d)_sma3", r"MFCC \1"),
    (r"^mfcc(\d)V_sma3nz", r"MFCC \1 (voiced)"),
    (r"^jitterLocal_sma3nz", "Jitter (pitch irregularity)"),
    (r"^shimmerLocaldB_sma3nz", "Shimmer (loudness irregularity)"),
    (r"^HNRdBACF_sma3nz", "Harmonics-to-noise ratio"),
    (r"^logRelF0-H1-H2_sma3nz", "H1–H2 (breathiness)"),
    (r"^logRelF0-H1-A3_sma3nz", "H1–A3 (spectral tilt)"),
    (r"^F(\d)frequency_sma3nz", r"Formant F\1 frequency"),
    (r"^F(\d)bandwidth_sma3nz", r"Formant F\1 bandwidth"),
    (r"^F(\d)amplitudeLogRelF0_sma3nz", r"Formant F\1 amplitude"),
    (r"^alphaRatioV_sma3nz", "Alpha ratio (voiced)"),
    (r"^alphaRatioUV_sma3nz", "Alpha ratio (unvoiced)"),
    (r"^hammarbergIndexV_sma3nz", "Hammarberg index (voiced)"),
    (r"^hammarbergIndexUV_sma3nz", "Hammarberg index (unvoiced)"),
    (r"^slopeV0-500_sma3nz", "Spectral slope 0–500 Hz (voiced)"),
    (r"^slopeV500-1500_sma3nz", "Spectral slope 0.5–1.5 kHz (voiced)"),
    (r"^slopeUV0-500_sma3nz", "Spectral slope 0–500 Hz (unvoiced)"),
    (r"^slopeUV500-1500_sma3nz", "Spectral slope 0.5–1.5 kHz (unvoiced)"),
]
_STAT = {"amean": "mean", "stddevNorm": "variability", "percentile20.0": "20th percentile",
         "percentile50.0": "median", "percentile80.0": "80th percentile", "pctlrange0-2": "range",
         "meanRisingSlope": "rising slope", "stddevRisingSlope": "rising-slope variability",
         "meanFallingSlope": "falling slope", "stddevFallingSlope": "falling-slope variability"}
_WHOLE = {"loudnessPeaksPerSec": "Loudness peaks per second (tempo)",
          "VoicedSegmentsPerSec": "Voiced segments per second",
          "MeanVoicedSegmentLengthSec": "Voiced segment length (mean)",
          "StddevVoicedSegmentLengthSec": "Voiced segment length (variability)",
          "MeanUnvoicedSegmentLength": "Pause / unvoiced length (mean)",
          "StddevUnvoicedSegmentLength": "Pause / unvoiced length (variability)",
          "equivalentSoundLevel_dBp": "Overall sound level"}


def readable(name: str) -> str:
    if name in _WHOLE:
        return _WHOLE[name]
    base, _, stat = name.rpartition("_")
    for pat, rep in _BASE:
        if re.match(pat, base):
            return f"{re.sub(pat, rep, base).split('_')[0]} ({_STAT.get(stat, stat)})"
    return name


# ---------------------------------------------------------------------------
# Signal views
# ---------------------------------------------------------------------------
def signal_views(y: np.ndarray, sr: int = TARGET_SR) -> dict:
    import librosa
    hop, n_fft = 160, 512                                    # 10 ms hop, 32 ms window
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    spec_db = librosa.amplitude_to_db(S, ref=np.max, top_db=80)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    t = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)
    step = max(1, len(y) // 4000)                            # light waveform for plotting
    return {"t_wave": np.arange(0, len(y), step) / sr, "wave": y[::step],
            "t": t, "freqs": librosa.fft_frequencies(sr=sr, n_fft=n_fft), "spec_db": spec_db,
            "rms_db": 20 * np.log10(rms[: len(t)] + 1e-6)}


# ---------------------------------------------------------------------------
# Training distribution of eGeMAPS features (for percentiles)
# ---------------------------------------------------------------------------
def training_egemaps() -> pd.DataFrame:
    """All labelled clips of both corpora (what the final model was trained on)."""
    from src.features import load_egemaps
    parts = []
    for c in ["ravdess", "cremad"]:
        meta = pd.read_csv(META_DIR / f"{c}.csv")
        meta = meta[meta["label"].notna() & (meta["fold"] >= 0)]
        parts.append(load_egemaps(c).loc[meta["file"]])
    return pd.concat(parts)


def percentiles(values: np.ndarray, train: pd.DataFrame) -> np.ndarray:
    """Percentile (0-100) of each feature value within the training clips."""
    T = train.values
    return np.array([(T[:, j] < v).mean() * 100 for j, v in enumerate(values)])


def describe_level(p: float) -> str:
    if p < 10: return "very low"
    if p < 30: return "low"
    if p <= 70: return "typical"
    if p <= 90: return "high"
    return "very high"


def explanation_sentence(expl: pd.DataFrame, label: str, n: int = 3, versus: str | None = None,
                         relative_to: str | None = None) -> str:
    """
    One plain-language sentence from the top features pushing toward `label`.

    versus:       a second emotion; `expl` then holds contrastive contributions (label minus versus)
    relative_to:  e.g. "your normal voice"; `value` then holds differences from that voice
    """
    def lower_first(t):                      # "Loudness (mean)" -> "loudness (mean)", keep "MFCC 4"
        return t if t[:2].isupper() else t[0].lower() + t[1:]

    def describe(r):
        if relative_to is not None:
            return f"{'higher' if r.value > 0 else 'lower'} {lower_first(r.readable)}"
        return f"{describe_level(r.percentile)} {lower_first(r.readable)}"

    toward = expl[expl["contribution"] > 0].head(n)
    against = expl[expl["contribution"] < 0].head(2)
    where = f" Compared with {relative_to}," if relative_to else ""
    if versus is None:
        if toward.empty:
            return f"No single voice feature strongly supports '{label}'."
        head = f"{where} the main cues toward **{label}** are: " if relative_to else f"Main voice cues toward **{label}**: "
        return head.strip() + " " + "; ".join(describe(r) for r in toward.itertuples()) + "."
    text = f"**Why {label} and not {versus}?**" + (f"{where} what" if relative_to else " What")
    text += (f" tips it toward {label}: " + "; ".join(describe(r) for r in toward.itertuples()) + "."
             if not toward.empty else f" favours {label}: no single feature stands out.")
    if not against.empty:
        text += f" Pulling toward {versus}: " + "; ".join(describe(r) for r in against.itertuples()) + "."
    return text


# ---------------------------------------------------------------------------
# Explanations from the saved bundle
# ---------------------------------------------------------------------------
def local_shap(egemaps: np.ndarray, bundle: dict, k: int) -> pd.DataFrame:
    """Exact linear SHAP of the eGeMAPS explainer for class k (all 88 features)."""
    exp = bundle["explainer"]
    z = exp[:-1].transform(egemaps[None, :])[0]
    phi = (z - bundle["explainer_train_mean_z"]) * exp.named_steps["clf"].coef_[k]
    names = bundle["egemaps_names"]
    df = pd.DataFrame({"feature": names, "readable": [readable(n) for n in names],
                       "group": [feature_group(n) for n in names], "value": egemaps,
                       "z": z, "contribution": phi})
    return df.reindex(df["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def fusion_evidence_split(egemaps: np.ndarray, wavlm_layer: np.ndarray, bundle: dict, k: int) -> dict:
    """
    Split the fusion model's evidence for class k into its eGeMAPS part and its WavLM part
    (sum of |weight x standardised value|; reference = training mean, which is 0 after scaling).
    """
    pipe = bundle["predictor"]
    z = pipe[:-1].transform(np.hstack([egemaps, wavlm_layer])[None, :])[0]
    contrib = z * pipe.named_steps["clf"].coef_[k]
    n_eg = len(egemaps)
    a, b = np.abs(contrib[:n_eg]).sum(), np.abs(contrib[n_eg:]).sum()
    return {"eGeMAPS (88 named features)": a / (a + b), "WavLM (768 learned dimensions)": b / (a + b)}


# ---------------------------------------------------------------------------
# Robustness simulation
# ---------------------------------------------------------------------------
def perturb(y: np.ndarray, kind: str, amount: float, sr: int = TARGET_SR, seed: int = 0) -> np.ndarray:
    """Return a modified copy of the waveform. kind in {"gain", "pitch", "noise"}."""
    if kind == "gain":                                  # amount in dB
        return np.clip(y * 10 ** (amount / 20), -1, 1).astype(np.float32)
    if kind == "pitch":                                 # amount in semitones
        import librosa
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=amount).astype(np.float32)
    if kind == "noise":                                 # amount = SNR in dB
        rng = np.random.default_rng(seed)
        p_sig = np.mean(y ** 2)
        noise = rng.standard_normal(len(y)) * np.sqrt(p_sig / 10 ** (amount / 10))
        return np.clip(y + noise, -1, 1).astype(np.float32)
    raise ValueError(kind)


PERTURBATIONS = [("original", None, 0, "Original"),
                 ("gain", "gain", -12, "12 dB quieter"), ("gain", "gain", 6, "6 dB louder"),
                 ("pitch", "pitch", -2, "Pitch −2 semitones"), ("pitch", "pitch", 2, "Pitch +2 semitones"),
                 ("noise", "noise", 20, "Noise, SNR 20 dB"), ("noise", "noise", 10, "Noise, SNR 10 dB")]


# ---------------------------------------------------------------------------
# Dataset examples (speakers held out from the demo model)
# ---------------------------------------------------------------------------
def heldout_examples(fold: int = 0) -> pd.DataFrame:
    rows = []
    for c in ["ravdess", "cremad"]:
        m = pd.read_csv(META_DIR / f"{c}.csv")
        m = m[m["label"].notna() & (m["fold"] == fold)].copy()
        m["corpus_name"] = NAME[c]
        cols = ["path", "file", "corpus_name", "speaker_id", "gender", "label"]
        if "voice_vote" in m:
            cols += ["voice_agrees"]
        rows.append(m[[k for k in cols if k in m]])
    out = pd.concat(rows, ignore_index=True)
    out["exists"] = [(PROJECT_ROOT / p).exists() for p in out["path"]]
    return out[out["exists"]].drop(columns="exists").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Saved results
# ---------------------------------------------------------------------------
def load_json(name: str) -> dict:
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else {}


def load_csv(rel: str) -> pd.DataFrame | None:
    p = RESULTS / rel
    return pd.read_csv(p) if p.exists() else None


def labels_for(corpus: str) -> np.ndarray:
    """Labels in the same order as the Phase 6 out-of-fold predictions."""
    m = pd.read_csv(META_DIR / f"{corpus}.csv")
    m = m[m["label"].notna() & (m["fold"] >= 0)].reset_index(drop=True)
    return m["label"].map({c: i for i, c in enumerate(CLASSES)}).values


def parse_ci(s: str) -> tuple[float, ...]:
    """'30.1 [28.9, 31.2]' -> (30.1, 28.9, 31.2);  '[-1.7, +0.9]' -> (-1.7, 0.9)"""
    return tuple(float(x) for x in re.findall(r"[-+]?\d+\.?\d*", s))
