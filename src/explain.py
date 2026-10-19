"""
Explainability (Phase 7): SHAP for eGeMAPS models, computed out-of-fold.

For each outer speaker fold, the model is refitted on the training folds with the
hyperparameters chosen in Phase 6, then the test-fold clips are explained.
So every clip is explained by a model that never heard that speaker.

Logistic regression: exact linear SHAP (interventional / independent features)
    phi[i, j, k] = w[k, j] * (z[i, j] - mean_train_z[j]),  z = clipped standardised value
which is what shap.LinearExplainer computes; we verify this in the notebook.
LightGBM: exact TreeSHAP via shap.TreeExplainer.
"""
from __future__ import annotations

import ast
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import PROJECT_ROOT
from src.models import make_model, CLASSES

PHASE6_DIR = PROJECT_ROOT / "results" / "phase6"
GROUPS = ["Energy", "Frequency", "Spectral", "Temporal"]   # P5 / GeMAPS categories


def feature_group(name: str) -> str:
    """Map an eGeMAPS feature name to its parameter group (GeMAPS paper, P3; P5 Table I)."""
    n = name.lower()
    temporal = ("loudnesspeakspersec", "voicedsegmentspersec", "voicedsegmentlength",
                "unvoicedsegmentlength")
    if any(t in n for t in temporal):
        return "Temporal"
    if any(t in n for t in ("f0semitone", "jitter", "frequency", "bandwidth")):
        return "Frequency"
    if any(t in n for t in ("loudness", "shimmer", "hnr", "equivalentsoundlevel")):
        return "Energy"
    return "Spectral"   # alpha ratio, Hammarberg, slopes, formant amplitudes, H1-H2, H1-A3, MFCC, flux


def phase6_params(corpus: str, model: str, feature_set: str = "egemaps") -> dict[int, dict]:
    """Hyperparameters chosen per fold by the inner CV in Phase 6."""
    df = pd.read_csv(PHASE6_DIR / f"e1_{corpus}_{feature_set}_{model}.csv")
    return {int(r.fold): ast.literal_eval(r.params) for r in df.itertuples()}


def _fit(model: str, params: dict, X, y, seed: int = 0) -> Pipeline:
    pipe, _ = make_model(model, seed)
    pipe.set_params(**params)
    return pipe.fit(X, y)


def linear_shap(pipe: Pipeline, X_train: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Exact SHAP for a (scale, clip) + logistic-regression pipeline. Returns (n, F, K)."""
    prep, clf = pipe[:-1], pipe.named_steps["clf"]
    Z, Ztr = prep.transform(X), prep.transform(X_train)
    return (Z - Ztr.mean(axis=0))[:, :, None] * clf.coef_.T[None, :, :]


def tree_shap(pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    import shap
    Z = pipe[:-1].transform(X)
    sv = shap.TreeExplainer(pipe.named_steps["clf"]).shap_values(Z)
    if isinstance(sv, list):                     # older shap: list of (n, F) per class
        sv = np.stack(sv, axis=-1)
    return np.asarray(sv)                        # (n, F, K)


def oof_explanations(data: dict, corpus: str, model: str = "logreg", seed: int = 0,
                     with_permutation: bool = True) -> dict:
    """
    Out-of-fold SHAP for every clip, plus per-fold global importances.
    Returns dict with:
      shap      (n, F, K)  out-of-fold SHAP values (log-odds units for logreg)
      pred      (n,)       out-of-fold predictions
      fold_imp  (folds, F) mean |SHAP| per fold (global importance, used in Phase 8)
      fold_perm (folds, F) permutation importance (drop in UAR) per fold
    """
    X, y, folds = data["egemaps"], data["y"], data["folds"]
    params = phase6_params(corpus, model)
    n, F, K = X.shape[0], X.shape[1], len(CLASSES)
    shap_all, pred = np.zeros((n, F, K), dtype=np.float32), np.full(n, -1)
    fold_ids = sorted(np.unique(folds))
    fold_imp, fold_perm = np.zeros((len(fold_ids), F)), np.zeros((len(fold_ids), F))
    for i, k in enumerate(fold_ids):
        tr, te = folds != k, folds == k
        pipe = _fit(model, params[k], X[tr], y[tr], seed)
        sv = linear_shap(pipe, X[tr], X[te]) if model == "logreg" else tree_shap(pipe, X[te])
        shap_all[te], pred[te] = sv, pipe.predict(X[te])
        fold_imp[i] = np.abs(sv).mean(axis=(0, 2))
        if with_permutation:
            pi = permutation_importance(pipe, X[te], y[te], scoring="balanced_accuracy",
                                        n_repeats=10, random_state=seed, n_jobs=-1)
            fold_perm[i] = pi.importances_mean
    return {"shap": shap_all, "pred": pred, "fold_imp": fold_imp, "fold_perm": fold_perm}


def global_importance(shap_values: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Mean |SHAP| over clips and classes, plus per-class columns."""
    df = pd.DataFrame({"feature": names,
                       "group": [feature_group(n) for n in names],
                       "importance": np.abs(shap_values).mean(axis=(0, 2))})
    for k, c in enumerate(CLASSES):
        df[c] = np.abs(shap_values[:, :, k]).mean(axis=0)
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def direction(shap_values: np.ndarray, X: np.ndarray, class_idx: int) -> np.ndarray:
    """Spearman-style sign: does a HIGHER feature value push TOWARD this class? (+1 / -1)"""
    from scipy.stats import spearmanr
    out = []
    for j in range(X.shape[1]):
        r = spearmanr(X[:, j], shap_values[:, j, class_idx]).statistic
        out.append(0.0 if np.isnan(r) else r)
    return np.array(out)


def group_shares(imp: pd.DataFrame) -> pd.Series:
    """Share of total importance per feature group (sums to 1)."""
    s = imp.groupby("group")["importance"].sum()
    return (s / s.sum()).reindex(GROUPS)


def explain_clip(i: int, shap_values: np.ndarray, X: np.ndarray, names: list[str],
                 pred: np.ndarray, y: np.ndarray, top: int = 5) -> str:
    """Plain-language explanation of one clip's prediction."""
    k = int(pred[i])
    contrib = shap_values[i, :, k]
    order = np.argsort(-np.abs(contrib))[:top]
    lines = [f"Predicted **{CLASSES[k]}** (true: {CLASSES[int(y[i])]})."
             f" Top features for the '{CLASSES[k]}' score:"]
    for j in order:
        arrow = "pushed TOWARD" if contrib[j] > 0 else "pushed AWAY from"
        lines.append(f"  - {names[j]} = {X[i, j]:.3g}  {arrow} {CLASSES[k]} ({contrib[j]:+.2f})")
    return "\n".join(lines)
