"""
"Calibrate to my voice": judge a recording relative to the same speaker's normal voice.

Why. A new microphone, room or voice shifts every feature (Phase 8 domain shift). The model
also leans on loudness (Phase 7), so a quiet laptop microphone pushes normal speech towards
*sad*, and any loud speech towards *angry*. Per-corpus normalisation (Phase 8) removed part of
that shift for a whole dataset; this module does the same for ONE speaker, using a few
recordings of their normal, neutral voice as the reference point.

How.
  baseline       = mean features of k neutral recordings of the speaker (k = 3 by default)
  relative clip  = clip features - baseline          ("how is this different from MY normal voice?")
  relative model = same recipe as the final model (LogReg, fusion of eGeMAPS + WavLM layer),
                   trained on relative features of every training speaker, whose baseline is
                   the mean of k randomly chosen neutral clips (so the model learns to cope with
                   a baseline estimated from only a few recordings)
  calibrated     = geometric mean of the original model's and the relative model's probabilities
                   (multiply, then renormalise: a class wins only if BOTH models find it plausible)

Tested on held-out speakers, across datasets and on a simulated laptop microphone in
report/09_simulation.md (section 7).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import PROJECT_ROOT
from src.models import CLASSES, load_corpus, preprocess_steps

MODEL_DIR = PROJECT_ROOT / "models"
NEUTRAL = CLASSES.index("neutral")
K_BASELINE = 3


def path_for(heldout_fold: int | None) -> Path:
    return MODEL_DIR / ("calibration_all.joblib" if heldout_fold is None else f"calibration_heldout_fold{heldout_fold}.joblib")


def speaker_relative(X: np.ndarray, y: np.ndarray, spk: np.ndarray, k: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Subtract, for every speaker, the mean of k of their neutral clips. Returns (X_rel, usable mask)."""
    Xr, ok = X.astype(np.float32).copy(), np.ones(len(X), bool)
    for s in np.unique(spk):
        idx = np.where(spk == s)[0]
        neu = idx[y[idx] == NEUTRAL]
        if len(neu) < k:
            ok[idx] = False
            continue
        base = rng.choice(neu, k, replace=False)
        Xr[idx] = X[idx] - X[base].mean(axis=0)
    return Xr, ok


def _logreg(C: float) -> Pipeline:
    return Pipeline(preprocess_steps() + [("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))])


def relative_training_data(layer: int, heldout_fold: int | None = None, k: int = K_BASELINE,
                           seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Relative fusion features (eGeMAPS first, then the WavLM layer) and labels of every training speaker."""
    rng = np.random.default_rng(seed)
    Xf, ys = [], []
    names = None
    for c in ["ravdess", "cremad"]:
        d = load_corpus(c)
        keep = np.ones(len(d["y"]), bool) if heldout_fold is None else np.asarray(d["folds"]) != heldout_fold
        y, spk = np.asarray(d["y"])[keep], np.asarray(d["groups"])[keep]
        fusion = np.hstack([d["egemaps"][keep], d["wavlm"][keep, layer].astype(np.float32)])
        rel, ok = speaker_relative(fusion, y, spk, k, rng)
        Xf.append(rel[ok]); ys.append(y[ok])
        names = d["egemaps_names"]
    return np.vstack(Xf), np.concatenate(ys), names


def train_relative(layer: int, heldout_fold: int | None = None, k: int = K_BASELINE, seed: int = 0,
                   C_fusion: float = 0.01, C_egemaps: float = 1.0) -> dict:
    """Relative fusion model (predicts) and relative eGeMAPS model (explains) on both corpora."""
    Xf, y, names = relative_training_data(layer, heldout_fold, k, seed)
    Xe = Xf[:, : len(names)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = _logreg(C_fusion).fit(Xf, y)
        expl = _logreg(C_egemaps).fit(Xe, y)
    return {"predictor": pred, "explainer": expl, "layer": layer, "k": k, "egemaps_names": names,
            "explainer_train_mean_z": expl[:-1].transform(Xe).mean(axis=0),
            "heldout_fold": heldout_fold, "n_clips": int(len(y))}


def load_or_train(layer: int, heldout_fold: int | None) -> dict:
    p = path_for(heldout_fold)
    if p.exists():
        m = joblib.load(p)
        if m.get("layer") == layer:
            return m
    m = train_relative(layer, heldout_fold)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, p)
    return m


# ---------------------------------------------------------------------------
# Using a calibration
# ---------------------------------------------------------------------------
def baseline_from(features: list[dict]) -> dict:
    """Mean eGeMAPS (88) and WavLM (13 x 768) over the speaker's neutral recordings."""
    return {"egemaps": np.mean([f["egemaps"] for f in features], axis=0),
            "wavlm": np.mean([f["wavlm"] for f in features], axis=0), "n": len(features)}


def calibrated_predict(egemaps: np.ndarray, wavlm: np.ndarray, base: dict, rel_model: dict,
                       abs_proba: np.ndarray) -> dict:
    """Relative-model probabilities, and the calibrated result (geometric mean with the original model)."""
    L = rel_model["layer"]
    rel_eg = egemaps - base["egemaps"]
    rel_f = np.hstack([rel_eg, wavlm[L] - base["wavlm"][L]])[None, :]
    p_rel = rel_model["predictor"].predict_proba(rel_f)[0]
    g = np.sqrt(np.clip(np.asarray(abs_proba, float), 1e-9, 1) * np.clip(p_rel, 1e-9, 1))
    p_cal = g / g.sum()
    return {"relative_proba": dict(zip(CLASSES, p_rel)), "proba": dict(zip(CLASSES, p_cal)),
            "label": CLASSES[int(np.argmax(p_cal))], "relative_egemaps": rel_eg}


def relative_shap(rel_eg: np.ndarray, rel_model: dict, k: int) -> np.ndarray:
    """Exact linear SHAP of the relative eGeMAPS model for class k (per feature)."""
    exp = rel_model["explainer"]
    z = exp[:-1].transform(rel_eg[None, :])[0]
    return (z - rel_model["explainer_train_mean_z"]) * exp.named_steps["clf"].coef_[k]
