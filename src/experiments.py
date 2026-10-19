"""
Phase 8 experiments.

E2  cross-corpus transfer (H1)         train on one corpus, test on the other
E3  explanation stability (H2)          SHAP rankings across resamples, speakers, corpora
E4  stable-core feature selection (H3)  do consistently-important features transfer better?
E5  error analysis helpers              per-class, per-gender, listener agreement

Leakage rules: every selection (WavLM layer, hyperparameters, feature subset)
uses SOURCE-corpus data only. The target corpus is used for scoring, plus,
in the per-corpus-normalisation variant only, its *unlabelled* feature
statistics (mean/std), stated explicitly wherever it is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.pipeline import Pipeline

from src.models import (features, fit_tuned, select_layer, preprocess_steps, CLASSES)
from src.explain import linear_shap

FIXED_C = {"egemaps": 0.1, "wavlm": 0.01, "fusion": 0.01}   # Phase 6 typical choices


# ---------------------------------------------------------------------------
# Normalisation variants
# ---------------------------------------------------------------------------
def corpus_standardise(X: np.ndarray) -> np.ndarray:
    """Standardise one corpus with ITS OWN mean/std (uses no labels)."""
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)


# ---------------------------------------------------------------------------
# Speaker-level bootstrap (clips of one speaker are not independent)
# ---------------------------------------------------------------------------
def speaker_bootstrap(y, preds: dict, groups, n_boot: int = 1000, seed: int = 0) -> dict:
    """
    preds: {name: predictions}. Resamples TARGET speakers with replacement and
    recomputes UAR for every prediction set on the same resample (paired).
    Returns {name: array of n_boot UARs}.
    """
    rng = np.random.default_rng(seed)
    spk = np.unique(groups)
    idx_by_spk = {s: np.where(groups == s)[0] for s in spk}
    out = {k: np.empty(n_boot) for k in preds}
    for b in range(n_boot):
        draw = rng.choice(spk, size=len(spk), replace=True)
        idx = np.concatenate([idx_by_spk[s] for s in draw])
        for k, p in preds.items():
            out[k][b] = balanced_accuracy_score(y[idx], p[idx])
    return out


def ci(a: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    lo, hi = np.percentile(a, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# E2: cross-corpus transfer
# ---------------------------------------------------------------------------
def cross_corpus(src: dict, tgt: dict, feature_set: str, model_name: str,
                 norm: str = "source", seed: int = 0, layer: int | None = None) -> dict:
    """
    Train on all labelled source clips, predict all target clips.
    norm = "source": scaler fitted on source only (strict).
           "corpus": each corpus first standardised with its own statistics
                     (unsupervised adaptation; uses unlabelled target features).
    """
    if feature_set != "egemaps" and layer is None:   # choose WavLM layer on SOURCE only
        layer, _ = select_layer(src["wavlm"], src["y"], src["groups"])
    if feature_set == "egemaps":
        layer = None
    Xs, Xt = features(src, feature_set, layer), features(tgt, feature_set, layer)
    if norm == "corpus":
        Xs, Xt = corpus_standardise(Xs), corpus_standardise(Xt)
    model, params = fit_tuned(model_name, Xs, src["y"], src["groups"], seed)
    pred = model.predict(Xt)
    return {"pred": pred, "layer": layer, "params": params,
            "uar": balanced_accuracy_score(tgt["y"], pred),
            "recall": recall_score(tgt["y"], pred, labels=range(len(CLASSES)), average=None)}


# ---------------------------------------------------------------------------
# E3: explanation stability
# ---------------------------------------------------------------------------
def _fit_logreg(X, y, C):
    pipe = Pipeline(preprocess_steps() +
                    [("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))])
    return pipe.fit(X, y)


def importance_logreg(X, y, C) -> np.ndarray:
    """Global importance = mean |SHAP| over the model's own training clips and classes."""
    pipe = _fit_logreg(X, y, C)
    return np.abs(linear_shap(pipe, X, X)).mean(axis=(0, 2))


def importance_lgbm(X, y, seed=0) -> np.ndarray:
    from src.models import make_model
    from src.explain import tree_shap
    pipe, _ = make_model("lgbm", seed)
    pipe.set_params(clf__min_child_samples=10)
    pipe.fit(X, y)
    return np.abs(tree_shap(pipe, X)).mean(axis=(0, 2))


def speaker_subsets(groups, n_speakers: int, n_pairs: int, seed: int = 0):
    """n_pairs of DISJOINT speaker sets of size n_speakers each."""
    rng = np.random.default_rng(seed)
    spk = np.unique(groups)
    for _ in range(n_pairs):
        perm = rng.permutation(spk)
        yield set(perm[:n_speakers]), set(perm[n_speakers:2 * n_speakers])


def similarity(a: np.ndarray, b: np.ndarray, k: int) -> dict:
    ta, tb = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
    return {"spearman": spearmanr(a, b).statistic, "kendall": kendalltau(a, b).statistic,
            "topk_jaccard": len(ta & tb) / len(ta | tb)}


def null_similarity(n_features: int, k: int, n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Similarity between RANDOM rankings: the 'no agreement' baseline (P10)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame([similarity(rng.random(n_features), rng.random(n_features), k)
                         for _ in range(n)])


# ---------------------------------------------------------------------------
# E4: stable-core feature selection
# ---------------------------------------------------------------------------
def subset_importances(X, y, groups, C, n_models: int = 20, frac: float = 0.5,
                       seed: int = 0) -> np.ndarray:
    """Importance vectors of n_models LogReg models, each trained on a random half of the SOURCE speakers."""
    rng = np.random.default_rng(seed)
    spk = np.unique(groups)
    imps = []
    for _ in range(n_models):
        chosen = rng.choice(spk, size=max(2, int(frac * len(spk))), replace=False)
        m = np.isin(groups, chosen)
        imps.append(importance_logreg(X[m], y[m], C))
    return np.array(imps)


def stability_scores(imps: np.ndarray, k: int) -> pd.DataFrame:
    """For each feature: share of models where it is in the top-k, and its mean importance."""
    topk = np.argsort(-imps, axis=1)[:, :k]
    freq = np.array([(topk == j).any(axis=1).mean() for j in range(imps.shape[1])])
    return pd.DataFrame({"freq_topk": freq, "mean_imp": imps.mean(0)})


def select_stable(scores: pd.DataFrame, k: int) -> np.ndarray:
    """Top-k by how CONSISTENTLY a feature ranks highly (ties: mean importance)."""
    return scores.sort_values(["freq_topk", "mean_imp"], ascending=False).index[:k].values


def select_top_importance(X, y, C, k: int) -> np.ndarray:
    """Top-k by importance of ONE model trained on all source data (the usual approach)."""
    return np.argsort(-importance_logreg(X, y, C))[:k]


def transfer_with_subset(Xs, ys, Xt, cols, C) -> np.ndarray:
    return _fit_logreg(Xs[:, cols], ys, C).predict(Xt[:, cols])
