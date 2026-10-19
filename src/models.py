"""
Models and leak-free evaluation (Phase 6).

Leakage rules enforced here:
  * scaling lives inside a Pipeline, so it is fitted on training folds only
  * hyperparameters are tuned with an INNER speaker-grouped CV on training data
  * the WavLM layer is chosen with an INNER speaker-grouped CV on training data
  * the outer test fold is used once, for scoring
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data import META_DIR, HARMONISED
from src.features import load_egemaps, load_wavlm_features

CLASSES = HARMONISED  # fixed label order everywhere
DEFAULT_LAYER = 6     # a-priori middle layer (P4, P7), used only where stated


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_corpus(corpus: str) -> dict:
    """Labelled clips of one corpus with all feature sets aligned to the metadata."""
    meta = pd.read_csv(META_DIR / f"{corpus}.csv")
    meta = meta[meta["label"].notna() & (meta["fold"] >= 0)].reset_index(drop=True)
    eg = load_egemaps(corpus).loc[meta["file"]]
    wl_all, files = load_wavlm_features(corpus)
    pos = pd.Series(range(len(files)), index=files)
    wl = wl_all[pos.loc[meta["file"]].values].astype(np.float32)   # (n, 13, 768)
    return {
        "meta": meta,
        "y": meta["label"].map({c: i for i, c in enumerate(CLASSES)}).values,
        "groups": meta["speaker_id"].values,
        "folds": meta["fold"].values,
        "egemaps": eg.values.astype(np.float32),
        "egemaps_names": list(eg.columns),
        "wavlm": wl,
    }


def features(data: dict, feature_set: str, layer: int | None = None) -> np.ndarray:
    if feature_set == "egemaps":
        return data["egemaps"]
    if feature_set == "wavlm":
        return data["wavlm"][:, layer]
    if feature_set == "fusion":
        return np.hstack([data["egemaps"], data["wavlm"][:, layer]])
    raise ValueError(feature_set)


# ---------------------------------------------------------------------------
# Models (no KNN, by project rule)
# ---------------------------------------------------------------------------
def make_model(name: str, seed: int = 0) -> tuple[Pipeline, dict]:
    """Return (pipeline, small hyperparameter grid)."""
    if name == "logreg":
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        grid = {"clf__C": [0.01, 0.1, 1.0]}
    elif name == "svm":
        clf = SVC(kernel="rbf", gamma="scale", class_weight="balanced")
        grid = {"clf__C": [1.0, 10.0]}
    elif name == "lgbm":
        from lightgbm import LGBMClassifier
        # n_jobs=1: the grid search already runs fits in parallel; letting each
        # LightGBM fit also grab every core makes them fight and run ~10x slower.
        clf = LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=15,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                             class_weight="balanced", random_state=seed, verbose=-1, n_jobs=1)
        grid = {"clf__min_child_samples": [10, 30]}
    elif name == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(256,), early_stopping=True, max_iter=300,
                            random_state=seed)
        grid = {"clf__alpha": [1e-3, 1e-1]}
    else:
        raise ValueError(name)
    return Pipeline([("scale", StandardScaler()), ("clf", clf)]), grid


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def scores(y_true, y_pred) -> dict:
    return {"uar": balanced_accuracy_score(y_true, y_pred),          # = macro recall
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "accuracy": accuracy_score(y_true, y_pred)}


# ---------------------------------------------------------------------------
# Inner-CV helpers (training data only)
# ---------------------------------------------------------------------------
def select_layer(wavlm: np.ndarray, y, groups, n_inner: int = 3) -> tuple[int, list[float]]:
    """Pick the WavLM layer with the best inner-CV UAR (logistic regression, C=0.1)."""
    pipe = Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))])
    cv = GroupKFold(n_splits=n_inner)
    uars = []
    for layer in range(wavlm.shape[1]):
        pred = cross_val_predict(pipe, wavlm[:, layer], y, groups=groups, cv=cv, n_jobs=-1)
        uars.append(balanced_accuracy_score(y, pred))
    return int(np.argmax(uars)), uars


def fit_tuned(model_name: str, X, y, groups, seed: int = 0, n_inner: int = 3):
    pipe, grid = make_model(model_name, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs = GridSearchCV(pipe, grid, scoring="balanced_accuracy",
                          cv=GroupKFold(n_splits=n_inner), n_jobs=-1, refit=True)
        gs.fit(X, y, groups=groups)
    return gs.best_estimator_, gs.best_params_


# ---------------------------------------------------------------------------
# E1: within-corpus, speaker-independent outer CV
# ---------------------------------------------------------------------------
def run_within_corpus(data: dict, feature_set: str, model_name: str, seed: int = 0,
                      layers_per_fold: dict | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Outer loop = the saved speaker folds. Returns (per-fold metrics, out-of-fold predictions).
    layers_per_fold: {fold: layer} chosen on training data (needed for wavlm/fusion).
    """
    y, groups, folds = data["y"], data["groups"], data["folds"]
    oof = np.full(len(y), -1)
    rows = []
    for k in sorted(np.unique(folds)):
        tr, te = folds != k, folds == k
        layer = None if feature_set == "egemaps" else layers_per_fold[k]
        X = features(data, feature_set, layer)
        model, params = fit_tuned(model_name, X[tr], y[tr], groups[tr], seed)
        oof[te] = model.predict(X[te])
        rows.append({"fold": int(k), "layer": layer, "params": str(params), **scores(y[te], oof[te])})
    return pd.DataFrame(rows), oof


# ---------------------------------------------------------------------------
# E0: leakage check — random split vs speaker split, same model
# ---------------------------------------------------------------------------
def leakage_check(data: dict, feature_set: str, layer: int | None = None,
                  seed: int = 0) -> dict:
    X, y = features(data, feature_set, layer), data["y"]
    pipe = Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))])
    random_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred_random = cross_val_predict(pipe, X, y, cv=random_cv, n_jobs=-1)
    pred_speaker = np.empty_like(y)
    for k in np.unique(data["folds"]):
        tr, te = data["folds"] != k, data["folds"] == k
        pred_speaker[te] = clone(pipe).fit(X[tr], y[tr]).predict(X[te])
    return {"random_split_uar": balanced_accuracy_score(y, pred_random),
            "speaker_split_uar": balanced_accuracy_score(y, pred_speaker)}
