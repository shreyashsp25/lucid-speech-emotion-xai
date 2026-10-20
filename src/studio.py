"""
Interactive "studio" tools for the app (Phase 9 extension). No Streamlit code here.

  * emotion compass   : places the six class scores on a pleasantness x energy map
  * voice profile     : a clip's key voice features as training percentiles, next to
                        the typical profile of an emotion in the training data
  * what-if studio    : shift named voice features (in training standard deviations)
                        and recompute the interpretable eGeMAPS model exactly
  * two-dataset lens  : three eGeMAPS models trained on RAVDESS only, CREMA-D only and
                        both (fold-0 speakers always held out), so a user can see the
                        cross-corpus effect of Phase 8 on a single clip
  * emotion timeline  : the eGeMAPS model applied to 1-second sliding windows
  * waveform_svg      : a compact SVG trace of the clip for the page header
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.audio import TARGET_SR
from src.data import META_DIR
from src.explain import feature_group
from src.features import load_egemaps
from src.models import CLASSES, CLIP_Z, preprocess_steps

# ---------------------------------------------------------------------------
# Emotion compass
# ---------------------------------------------------------------------------
# x = pleasantness (valence), y = energy (arousal). Positions are a readable layout
# loosely following psychology's valence-arousal arrangement; they are not measured.
COMPASS = {"happy": (0.80, 0.45), "angry": (-0.45, 0.85), "fearful": (-0.85, 0.40),
           "disgust": (-0.80, -0.30), "sad": (-0.40, -0.85), "neutral": (0.55, -0.55)}


def compass_point(proba: dict | pd.Series) -> tuple[float, float]:
    """Score-weighted position: near an emotion's anchor = decisive, near the centre = mixed."""
    p = np.array([float(proba[c]) for c in CLASSES])
    p = p / p.sum()
    xy = np.array([COMPASS[c] for c in CLASSES])
    x, y = p @ xy
    return float(x), float(y)


# ---------------------------------------------------------------------------
# Named features used by the profile and the what-if studio
# ---------------------------------------------------------------------------
KNOBS = [
    ("loudness_sma3_amean", "Loudness"),
    ("equivalentSoundLevel_dBp", "Overall sound level"),
    ("loudness_sma3_meanRisingSlope", "Loudness attack"),
    ("F0semitoneFrom27.5Hz_sma3nz_amean", "Pitch"),
    ("F0semitoneFrom27.5Hz_sma3nz_stddevNorm", "Pitch movement"),
    ("HNRdBACF_sma3nz_amean", "Voice clarity (HNR)"),
    ("alphaRatioV_sma3nz_amean", "Brightness (alpha ratio)"),
    ("VoicedSegmentsPerSec", "Speech rate"),
]
KNOB_HELP = {
    "Loudness": "average perceived loudness",
    "Overall sound level": "level of the whole clip in dB",
    "Loudness attack": "how sharply loudness rises at word onsets",
    "Pitch": "average pitch in semitones",
    "Pitch movement": "how much the pitch varies",
    "Voice clarity (HNR)": "clear tone (high) versus breathy or noisy (low)",
    "Brightness (alpha ratio)": "share of high-frequency energy; higher in tense voices",
    "Speech rate": "voiced segments per second",
}


# ---------------------------------------------------------------------------
# Training table with labels (for profiles and the lens)
# ---------------------------------------------------------------------------
def labelled_egemaps() -> pd.DataFrame:
    """All labelled clips of both corpora with corpus, fold and label columns."""
    parts = []
    for c in ["ravdess", "cremad"]:
        meta = pd.read_csv(META_DIR / f"{c}.csv")
        meta = meta[meta["label"].notna() & (meta["fold"] >= 0)]
        eg = load_egemaps(c).loc[meta["file"]].reset_index(drop=True)
        eg["corpus"] = c
        eg["fold"] = meta["fold"].values
        eg["label"] = meta["label"].values
        eg["speaker"] = meta["speaker_id"].values
        parts.append(eg)
    return pd.concat(parts, ignore_index=True)


def voice_profile(egemaps: np.ndarray, names: list[str], table: pd.DataFrame, emotion: str) -> pd.DataFrame:
    """Percentile of the clip and of the emotion's training median, for each knob feature."""
    rows = []
    for f, label in KNOBS:
        j = names.index(f)
        col = table[f].values
        typical = float(np.median(table.loc[table["label"] == emotion, f]))
        rows.append({"feature": label, "clip": float((col < egemaps[j]).mean() * 100),
                     "typical": float((col < typical).mean() * 100)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# What-if studio (exact, because the explainer is linear)
# ---------------------------------------------------------------------------
def _z(egemaps: np.ndarray, model: Pipeline) -> np.ndarray:
    return model[:-1].transform(egemaps[None, :])[0]


def what_if(egemaps: np.ndarray, model: Pipeline, names: list[str], shifts: dict[str, float]) -> np.ndarray:
    """Class probabilities after moving the given features by `shift` training SDs."""
    z = _z(egemaps, model).copy()
    for f, d in shifts.items():
        z[names.index(f)] += d
    z = np.clip(z, -CLIP_Z, CLIP_Z)
    return model.named_steps["clf"].predict_proba(z[None, :])[0]


def smallest_flips(egemaps: np.ndarray, model: Pipeline, names: list[str], target: str,
                   max_sd: float = 3.0, step: float = 0.05) -> pd.DataFrame:
    """For each knob alone: the smallest shift (in SD) that makes `target` the top emotion."""
    k = CLASSES.index(target)
    clf = model.named_steps["clf"]
    z0 = _z(egemaps, model)
    deltas = np.arange(step, max_sd + 1e-9, step)
    rows = []
    for f, label in KNOBS:
        j = names.index(f)
        found = None
        for d in deltas:                                   # grow |shift| until it flips
            for s in (d, -d):
                z = z0.copy(); z[j] = np.clip(z[j] + s, -CLIP_Z, CLIP_Z)
                p = clf.predict_proba(z[None, :])[0]
                if int(np.argmax(p)) == k:
                    found = (s, p[k]); break
            if found:
                break
        rows.append({"feature": label, "shift_sd": found[0] if found else np.nan,
                     "score": found[1] if found else np.nan})
    return pd.DataFrame(rows).sort_values("shift_sd", key=np.abs, na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Two-dataset lens
# ---------------------------------------------------------------------------
LENSES = {"RAVDESS only": ["ravdess"], "CREMA-D only": ["cremad"], "Both datasets": ["ravdess", "cremad"]}


def train_lenses(table: pd.DataFrame, names: list[str], heldout_fold: int = 0, C: float = 1.0) -> dict:
    """eGeMAPS LogReg (same recipe as the explainer) per training source; fold-0 speakers never used."""
    out = {}
    for lens, corpora in LENSES.items():
        d = table[table["corpus"].isin(corpora) & (table["fold"] != heldout_fold)]
        X = d[names].values.astype(np.float32)
        y = d["label"].map({c: i for i, c in enumerate(CLASSES)}).values
        pipe = Pipeline(preprocess_steps() + [("clf", LogisticRegression(C=C, max_iter=3000,
                                                                         class_weight="balanced"))])
        pipe.fit(X, y)
        out[lens] = {"model": pipe, "mean_z": pipe[:-1].transform(X).mean(axis=0),
                     "n_clips": int(len(d)), "n_speakers": int(d["speaker"].nunique())}
    return out


def lens_view(egemaps: np.ndarray, lens: dict, names: list[str], n_top: int = 3) -> dict:
    model = lens["model"]
    p = model.predict_proba(egemaps[None, :])[0]
    k = int(np.argmax(p))
    phi = (_z(egemaps, model) - lens["mean_z"]) * model.named_steps["clf"].coef_[k]
    order = np.argsort(-phi)[:n_top]
    from src.app_utils import readable
    return {"proba": dict(zip(CLASSES, p)), "label": CLASSES[k],
            "drivers": [(readable(names[j]), feature_group(names[j]), float(phi[j])) for j in order]}


# ---------------------------------------------------------------------------
# Emotion timeline
# ---------------------------------------------------------------------------
def timeline(y: np.ndarray, smile, model: Pipeline, win: float = 1.0, hop: float = 0.25,
             sr: int = TARGET_SR) -> pd.DataFrame:
    """eGeMAPS model on sliding windows. Rougher than the whole-clip result (trained on full clips)."""
    w, h = int(win * sr), int(hop * sr)
    if len(y) <= w:
        starts = [0]
    else:
        starts = list(range(0, len(y) - w + 1, h))
        if starts[-1] + w < len(y):
            starts.append(len(y) - w)
    rows = []
    for s in starts:
        seg = y[s: s + w]
        x = smile.process_signal(seg, sr).iloc[0].values.astype(np.float32)
        p = model.predict_proba(np.nan_to_num(x)[None, :])[0]
        rows.append({"t": (s + len(seg) / 2) / sr, **dict(zip(CLASSES, p))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Header graphics
# ---------------------------------------------------------------------------
def waveform_svg(y: np.ndarray | None, colour: str, width: int = 640, height: int = 120,
                 bars: int = 120) -> str:
    """A mirrored bar trace of the clip's envelope (a flat line when there is no clip)."""
    mid = height / 2
    if y is None or len(y) == 0:
        return (f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" '
                f'aria-label="No recording yet"><line x1="0" y1="{mid}" x2="{width}" y2="{mid}" '
                f'stroke="{colour}" stroke-width="2" stroke-dasharray="4 6" opacity=".6"/></svg>')
    chunks = np.array_split(np.abs(y), bars)
    env = np.array([c.max() if len(c) else 0 for c in chunks])
    env = env / (env.max() + 1e-9)
    bw = width / bars
    rects = []
    for i, v in enumerate(env):
        hgt = max(2.0, v * (height - 8))
        rects.append(f'<rect x="{i * bw + bw * 0.2:.1f}" y="{mid - hgt / 2:.1f}" width="{bw * 0.6:.1f}" '
                     f'height="{hgt:.1f}" rx="{bw * 0.3:.1f}"/>')
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" '
            f'aria-label="Waveform of the analysed clip"><g fill="{colour}">{"".join(rects)}</g></svg>')


def ring_svg(value: float, colour: str, track: str, size: int = 92, stroke: int = 9) -> str:
    """Circular gauge for a 0-1 score."""
    r = (size - stroke) / 2
    c = 2 * np.pi * r
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" role="img" '
            f'aria-label="Model score {value:.0%}">'
            f'<circle cx="{size / 2}" cy="{size / 2}" r="{r}" fill="none" stroke="{track}" stroke-width="{stroke}"/>'
            f'<circle cx="{size / 2}" cy="{size / 2}" r="{r}" fill="none" stroke="{colour}" stroke-width="{stroke}" '
            f'stroke-linecap="round" stroke-dasharray="{c * value:.1f} {c:.1f}" '
            f'transform="rotate(-90 {size / 2} {size / 2})"/>'
            f'<text x="50%" y="54%" text-anchor="middle" dominant-baseline="middle" fill="currentColor" '
            f'font-size="{size * 0.24:.0f}" font-weight="700">{value:.0%}</text></svg>')
