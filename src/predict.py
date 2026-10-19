"""
Final models and prediction on a NEW audio file.

Two models are trained on ALL labelled clips of BOTH corpora (8,498 clips, 115 speakers),
because Phase 8 showed that training on more diverse data transfers better:

  * predictor  : LogReg on fusion features (eGeMAPS + WavLM layer), the best model in
                 Phases 6 and 8; it gives the predicted emotion and probabilities.
  * explainer  : LogReg on eGeMAPS only; it gives a human-readable SHAP explanation
                 (named voice features). Its own prediction is shown alongside.

Usage
    from src.predict import load_final, predict_file
    bundle = load_final()
    result = predict_file("my_voice.wav", bundle)
"""
from __future__ import annotations

import json
import platform
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.audio import load_audio, TARGET_SR
from src.data import PROJECT_ROOT
from src.explain import feature_group
from src.models import CLASSES, CLIP_Z, fit_tuned, select_layer

MODEL_DIR = PROJECT_ROOT / "models"
BUNDLE_PATH = MODEL_DIR / "final_models.joblib"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def pooled_training_data(corpora: dict) -> dict:
    """Stack the labelled clips of both corpora (speaker ids are already corpus-prefixed)."""
    keys = ["y", "groups", "egemaps", "wavlm"]
    out = {k: np.concatenate([np.asarray(corpora[c][k]) for c in corpora]) for k in keys}
    out["egemaps_names"] = corpora[next(iter(corpora))]["egemaps_names"]
    out["corpus"] = np.concatenate([[c] * len(corpora[c]["y"]) for c in corpora])
    return out


def train_final(corpora: dict, seed: int = 0) -> dict:
    d = pooled_training_data(corpora)
    layer, layer_profile = select_layer(d["wavlm"], d["y"], d["groups"])          # inner speaker CV
    X_fusion = np.hstack([d["egemaps"], d["wavlm"][:, layer]])
    predictor, p_params = fit_tuned("logreg", X_fusion, d["y"], d["groups"], seed)
    explainer, e_params = fit_tuned("logreg", d["egemaps"], d["y"], d["groups"], seed)
    z_train = explainer[:-1].transform(d["egemaps"])
    return {
        "predictor": predictor, "explainer": explainer, "layer": int(layer),
        "classes": CLASSES, "egemaps_names": d["egemaps_names"],
        "explainer_train_mean_z": z_train.mean(axis=0),          # SHAP reference point
        "egemaps_train_quantiles": np.percentile(d["egemaps"], [1, 99], axis=0),
        "info": {"trained_on": "RAVDESS + CREMA-D (all labelled clips)",
                 "n_clips": int(len(d["y"])), "n_speakers": int(len(np.unique(d["groups"]))),
                 "predictor_params": p_params, "explainer_params": e_params,
                 "layer_profile_inner_uar": [round(float(v), 4) for v in layer_profile],
                 "wavlm_model": "microsoft/wavlm-base-plus", "sample_rate": TARGET_SR,
                 "sklearn": sklearn.__version__, "python": platform.python_version(),
                 "date": str(date.today())},
    }


def save_final(bundle: dict, path: Path = BUNDLE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    (path.parent / "final_models_info.json").write_text(json.dumps(bundle["info"], indent=2, default=str))
    return path


def load_final(path: Path = BUNDLE_PATH) -> dict:
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Feature extraction for one new file (same pipeline as training, Phase 5)
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def _extractors(device: str = "cpu"):
    if "smile" not in _CACHE:
        from src.features import egemaps_extractor, load_wavlm
        _CACHE["smile"] = egemaps_extractor()
        _CACHE["fe"], _CACHE["wavlm"] = load_wavlm(device=device)
        _CACHE["device"] = device
    return _CACHE["smile"], _CACHE["fe"], _CACHE["wavlm"], _CACHE["device"]


def extract_one(path_or_wave, sr: int | None = None, device: str = "cpu") -> dict:
    """Return the waveform and both feature sets for one recording."""
    from src.features import wavlm_embed
    if isinstance(path_or_wave, (str, Path)):
        y, sr = load_audio(path_or_wave)
    else:                                           # raw waveform (e.g. from a microphone)
        import librosa
        y = np.asarray(path_or_wave, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        y, _ = librosa.effects.trim(y, top_db=30)
        sr = TARGET_SR
    smile, fe, wavlm, dev = _extractors(device)
    eg = smile.process_signal(y, sr).iloc[0].values.astype(np.float32)
    wl = wavlm_embed(y, fe, wavlm, dev)             # (13, 768)
    return {"wave": y, "sr": sr, "egemaps": eg, "wavlm": wl}


# ---------------------------------------------------------------------------
# Prediction + explanation
# ---------------------------------------------------------------------------
def predict_features(egemaps: np.ndarray, wavlm: np.ndarray, bundle: dict, top: int = 8) -> dict:
    x_fusion = np.hstack([egemaps, wavlm[bundle["layer"]]])[None, :]
    proba = bundle["predictor"].predict_proba(x_fusion)[0]
    k = int(np.argmax(proba))

    exp = bundle["explainer"]
    z = exp[:-1].transform(egemaps[None, :])[0]
    contrib = (z - bundle["explainer_train_mean_z"])[:, None] * exp.named_steps["clf"].coef_.T  # (F, K)
    names = bundle["egemaps_names"]
    order = np.argsort(-np.abs(contrib[:, k]))[:top]
    explanation = pd.DataFrame({
        "feature": [names[j] for j in order],
        "group": [feature_group(names[j]) for j in order],
        "value": egemaps[order],
        "contribution": contrib[order, k],
        "direction": ["toward" if contrib[j, k] > 0 else "away from" for j in order],
    })
    exp_proba = exp.predict_proba(egemaps[None, :])[0]

    # Honest warnings for the user
    warnings = []
    lo, hi = bundle["egemaps_train_quantiles"]
    outside = int(((egemaps < lo) | (egemaps > hi)).sum())
    if outside > 10:
        warnings.append(f"{outside} of 88 voice features are outside the range seen in training "
                        "(different microphone/recording conditions?) — treat the result with caution.")
    if np.abs(z).max() >= CLIP_Z:
        warnings.append("Some feature values were extreme and were clipped.")
    if proba.max() < 0.4:
        warnings.append("Low confidence: no emotion clearly dominates.")
    return {"label": bundle["classes"][k], "proba": dict(zip(bundle["classes"], proba.round(4))),
            "explainer_label": bundle["classes"][int(np.argmax(exp_proba))],
            "explainer_proba": dict(zip(bundle["classes"], exp_proba.round(4))),
            "explanation": explanation, "warnings": warnings}


def predict_file(path_or_wave, bundle: dict | None = None, sr: int | None = None,
                 device: str = "cpu") -> dict:
    bundle = bundle or load_final()
    feats = extract_one(path_or_wave, sr=sr, device=device)
    out = predict_features(feats["egemaps"], feats["wavlm"], bundle)
    out["duration_s"] = round(len(feats["wave"]) / feats["sr"], 2)
    if out["duration_s"] < 1.0:
        out["warnings"].append("Very short clip (< 1 s) after trimming silence.")
    out["features"] = feats
    return out
