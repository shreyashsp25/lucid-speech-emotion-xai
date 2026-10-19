"""
Extra evaluation (Phase 8, E6): probabilities, classification reports, ROC-AUC,
calibration, overfitting (train vs validation) and learning curves.

Everything re-uses the saved speaker folds and the per-fold hyperparameters and
WavLM layers chosen in Phase 6, so these numbers describe the same models.
"""
from __future__ import annotations

import ast
import json
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from src.data import PROJECT_ROOT
from src.models import make_model, features

P6 = PROJECT_ROOT / "results" / "phase6"


def phase6_settings(corpus: str, feature_set: str, model: str) -> tuple[dict, dict]:
    """Per-fold hyperparameters and WavLM layers chosen in Phase 6."""
    df = pd.read_csv(P6 / f"e1_{corpus}_{feature_set}_{model}.csv")
    params = {int(r.fold): ast.literal_eval(r.params) for r in df.itertuples()}
    layers = {int(k): v for k, v in json.loads((P6 / f"layers_{corpus}.json").read_text())["layers"].items()}
    return params, layers


def refit_folds(data: dict, corpus: str, feature_set: str, model: str, seed: int = 0) -> dict:
    """
    Refit every outer fold with its Phase 6 settings.
    Returns out-of-fold probabilities (if the model has predict_proba) and the
    train vs validation UAR of every fold (overfitting check).
    """
    params, layers = phase6_settings(corpus, feature_set, model)
    y, folds = data["y"], data["folds"]
    proba = np.full((len(y), 6), np.nan)
    rows = []
    for k in sorted(np.unique(folds)):
        tr, te = folds != k, folds == k
        X = features(data, feature_set, None if feature_set == "egemaps" else layers[k])
        pipe, _ = make_model(model, seed)
        pipe.set_params(**params[k]).fit(X[tr], y[tr])
        if hasattr(pipe, "predict_proba"):
            proba[te] = pipe.predict_proba(X[te])
        rows.append({"fold": int(k),
                     "train_uar": balanced_accuracy_score(y[tr], pipe.predict(X[tr])),
                     "val_uar": balanced_accuracy_score(y[te], pipe.predict(X[te]))})
    return {"proba": proba, "folds": pd.DataFrame(rows)}


def expected_calibration_error(proba: np.ndarray, y: np.ndarray, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    """ECE on the top-class confidence; also returns the reliability table."""
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc, cf = correct[m].mean(), conf[m].mean()
        ece += m.mean() * abs(acc - cf)
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "confidence": cf, "accuracy": acc})
    return float(ece), pd.DataFrame(rows)


def learning_curve_speakers(data: dict, feature_set: str, layer: int | None, C: float,
                            sizes: list[int], seed: int = 0) -> pd.DataFrame:
    """
    UAR on each held-out speaker fold when training on only n of the training
    speakers (random subsets), for n in sizes. Shows whether more speakers help.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from src.models import preprocess_steps
    rng = np.random.default_rng(seed)
    X, y, g, folds = features(data, feature_set, layer), data["y"], data["groups"], data["folds"]
    rows = []
    for k in sorted(np.unique(folds)):
        tr_spk = np.unique(g[folds != k])
        te = folds == k
        for n in sizes:
            if n > len(tr_spk):
                continue
            chosen = rng.choice(tr_spk, size=n, replace=False)
            m = np.isin(g, chosen)
            pipe = Pipeline(preprocess_steps() + [("clf", LogisticRegression(
                C=C, max_iter=3000, class_weight="balanced"))]).fit(X[m], y[m])
            rows.append({"fold": int(k), "n_speakers": n,
                         "train_uar": balanced_accuracy_score(y[m], pipe.predict(X[m])),
                         "val_uar": balanced_accuracy_score(y[te], pipe.predict(X[te]))})
    return pd.DataFrame(rows)
