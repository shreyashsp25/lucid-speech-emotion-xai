"""
Final-review analyses (November 2026), asked for by the 23CSE301 evaluation components:

  * train / validation / test scores for every model           (component 4, PDF note 5)
  * class separability: pairwise AUC, silhouette, LDA map       (component 4)
  * dimensionality: PCA variance, UAR vs number of components   (component 5)
  * correlation analysis of the 88 eGeMAPS features             (component 5)

Run from the project root:

    python -m src.review_analysis tvt_all     # train / validation / test for all 24 models (slow: hours on 2 cores)
    python -m src.review_analysis tvt_all cremad 1 wavlm   # one corpus / n_jobs / feature set, to run parts in parallel
    python -m src.review_analysis tvt_split   # 3-1-1 fold split for the two chosen models
    python -m src.review_analysis separability
    python -m src.review_analysis dimensionality
    python -m src.review_analysis all

Outputs go to results/review/. Every number in the final report and the review deck
that is not from Phases 6-9 comes from this file.

Definitions (all speaker-independent):
  * test UAR       outer fold of the nested CV (the same numbers as Phase 6)
  * validation UAR inner 3-fold speaker-grouped CV score of the settings chosen for that
                   outer fold, computed on the training folds only
  * train UAR      the refitted model scored on its own training folds
"""
from __future__ import annotations

import ast
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, silhouette_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models import CLASSES, features, load_corpus, make_model, preprocess_steps, scores

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "review"
CORPORA = ["ravdess", "cremad"]
FINAL_LAYER = 7        # layer of the final predictor (Phase 9)
SEED = 0


def _lr(C: float, extra: list | None = None) -> Pipeline:
    return Pipeline(preprocess_steps() + (extra or []) +
                    [("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))])


def _save(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(obj)
    path.write_text(json.dumps(old, indent=1))


# ---------------------------------------------------------------------------
# A) train / validation / test for all 24 nested-CV models
# ---------------------------------------------------------------------------
def tvt_all(corpora: list[str] | None = None, n_jobs: int = -1, feats: list[str] | None = None) -> pd.DataFrame:
    """Re-uses the layer and hyperparameters that the nested CV chose in Phase 6.
    Progress is saved per corpus (results/review/_tvt_<corpus>.csv), so the two corpora can run
    in parallel and an interrupted run resumes where it stopped."""
    chosen = pd.read_csv(ROOT / "results" / "phase6" / "e1_all_folds.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    for corpus in corpora or CORPORA:
        part = OUT / (f"_tvt_{corpus}.csv" if not feats else f"_tvt_{corpus}_{'_'.join(feats)}.csv")
        rows = pd.read_csv(part).to_dict("records") if part.exists() else []
        done = {(x["features"], x["model"], x["fold"]) for f in OUT.glob(f"_tvt_{corpus}*.csv")
                for x in pd.read_csv(f).to_dict("records")}
        d = load_corpus(corpus)
        y, groups, folds = d["y"], d["groups"], d["folds"]
        for (feat, model), g in chosen[chosen.corpus == corpus].groupby(["features", "model"]):
            if feats and feat not in feats:
                continue
            for _, r in g.sort_values("fold").iterrows():
                k = int(r.fold)
                if (feat, model, k) in done or any(x["features"] == feat and x["model"] == model and x["fold"] == k for x in rows):
                    continue
                layer = None if pd.isna(r.layer) else int(r.layer)
                X = features(d, feat, layer)
                tr, te = folds != k, folds == k
                pipe, _ = make_model(model, SEED)
                pipe.set_params(**ast.literal_eval(r.params))
                inner = cross_val_predict(clone(pipe), X[tr], y[tr], groups=groups[tr],
                                          cv=GroupKFold(n_splits=3), n_jobs=n_jobs)
                pipe.fit(X[tr], y[tr])
                s_tr = scores(y[tr], pipe.predict(X[tr]))
                s_va = scores(y[tr], inner)
                s_te = scores(y[te], pipe.predict(X[te]))
                rows.append({"corpus": corpus, "features": feat, "model": model, "fold": k,
                             "layer": layer, "params": r.params,
                             **{f"train_{m}": v for m, v in s_tr.items()},
                             **{f"val_{m}": v for m, v in s_va.items()},
                             **{f"test_{m}": v for m, v in s_te.items()}})
                pd.DataFrame(rows).to_csv(part, index=False)          # save as we go
                print(corpus, feat, model, k, round(s_tr["uar"], 3), round(s_va["uar"], 3),
                      round(s_te["uar"], 3), flush=True)
    df = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("_tvt_*.csv"))], ignore_index=True)
    df = df.drop_duplicates(["corpus", "features", "model", "fold"]).sort_values(["corpus", "features", "model", "fold"])
    df.to_csv(OUT / "train_val_test_all_models.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# B) explicit 3-1-1 split for the two chosen models
# ---------------------------------------------------------------------------
def tvt_split() -> dict:
    """Fold k = test, fold k+1 = validation, the other three = train, for k = 0..4."""
    res = {}
    for corpus in CORPORA:
        d = load_corpus(corpus)
        y, f = d["y"], d["folds"]
        X = {"fusion": features(d, "fusion", FINAL_LAYER), "egemaps": d["egemaps"]}
        res[corpus] = {}
        for name, C in [("fusion", 0.01), ("egemaps", 1.0)]:
            per = []
            for k in range(5):
                te, va = f == k, f == (k + 1) % 5
                tr = ~(te | va)
                m = _lr(C).fit(X[name][tr], y[tr])
                per.append({s: scores(y[i], m.predict(X[name][i]))
                            for s, i in [("train", tr), ("val", va), ("test", te)]})
            res[corpus][name] = {s: {m: [float(np.mean([p[s][m] for p in per]) * 100),
                                         float(np.std([p[s][m] for p in per]) * 100)]
                                     for m in ["uar", "macro_f1", "accuracy"]}
                                 for s in ["train", "val", "test"]}
    _save("analysis.json", {"tvt_split": res})
    return res


# ---------------------------------------------------------------------------
# C) class separability
# ---------------------------------------------------------------------------
def separability() -> dict:
    res = {}
    for corpus in CORPORA:
        d = load_corpus(corpus)
        y, f = d["y"], d["folds"]
        X = features(d, "fusion", FINAL_LAYER)
        pair = np.full((6, 6), np.nan)
        for i in range(6):
            for j in range(i + 1, 6):
                sel = (y == i) | (y == j)
                yy, ff, Xs = (y[sel] == j).astype(int), f[sel], X[sel]
                p = np.zeros(len(yy))
                for k in range(5):
                    m = _lr(0.01).fit(Xs[ff != k], yy[ff != k])
                    p[ff == k] = m.predict_proba(Xs[ff == k])[:, 1]
                pair[i, j] = pair[j, i] = roc_auc_score(yy, p)
        sil = {n: float(silhouette_score(StandardScaler().fit_transform(features(d, n, FINAL_LAYER)), y,
                                         metric="cosine", sample_size=min(3000, len(y)), random_state=0))
               for n in ["egemaps", "wavlm", "fusion"]}
        # LDA map: fitted on folds 1-4 only, fold 0 projected (speakers it never saw)
        sc = StandardScaler().fit(X[f != 0])
        Z = lambda A: np.clip(sc.transform(A), -5, 5)
        lda = LinearDiscriminantAnalysis(n_components=2, solver="eigen", shrinkage="auto").fit(Z(X[f != 0]), y[f != 0])
        np.savez(OUT / f"lda_{corpus}.npz", Z=lda.transform(Z(X)), y=y, fold=f,
                 clip=d["meta"]["file"].values)
        res[corpus] = {"pair_auc": [[None if np.isnan(v) else float(v) for v in r] for r in pair],
                       "classes": CLASSES, "silhouette_cosine": sil}
    _save("analysis.json", {"separability": res})
    return res


# ---------------------------------------------------------------------------
# D) dimensionality and correlation
# ---------------------------------------------------------------------------
def dimensionality() -> dict:
    res = {}
    for corpus in CORPORA:
        d = load_corpus(corpus)
        y, f = d["y"], d["folds"]
        r = {}
        for n in ["egemaps", "wavlm"]:
            X = features(d, n, FINAL_LAYER)
            ev = PCA().fit(np.clip(StandardScaler().fit_transform(X), -5, 5)).explained_variance_ratio_.cumsum()
            r[f"pca_{n}"] = {"cum": ev[:120].tolist(), "k90": int(np.searchsorted(ev, .90) + 1),
                             "k95": int(np.searchsorted(ev, .95) + 1), "k99": int(np.searchsorted(ev, .99) + 1)}
        E = d["egemaps"]
        Cm = np.corrcoef(StandardScaler().fit_transform(E), rowvar=False)
        a = np.abs(Cm[np.triu_indices(88, 1)])
        names = d["egemaps_names"]
        iu = np.triu_indices(88, 1)
        top = np.argsort(-np.nan_to_num(a))[:10]
        r["corr"] = {"pairs": int(len(a)), "gt08": int((a > .8).sum()), "gt09": int((a > .9).sum()),
                     "mean_abs": float(np.nanmean(a)),
                     "top_pairs": [[names[iu[0][t]], names[iu[1][t]], float(Cm[iu[0][t], iu[1][t]])] for t in top]}
        np.save(OUT / f"corr_{corpus}.npy", Cm)
        # UAR vs number of PCA components, PCA fitted inside the training folds
        for n, ks, C in [("wavlm", [4, 8, 16, 32, 64, 128, 256, 768], 0.01), ("egemaps", [2, 5, 10, 20, 40, 88], 1.0)]:
            X = features(d, n, FINAL_LAYER)
            u = {}
            for kk in ks:
                p = np.zeros(len(y), int)
                for k in range(5):
                    extra = [("pca", PCA(n_components=kk, random_state=0))] if kk < X.shape[1] else None
                    p[f == k] = _lr(C, extra).fit(X[f != k], y[f != k]).predict(X[f == k])
                u[str(kk)] = float(balanced_accuracy_score(y, p) * 100)
            r[f"pca_uar_{n}"] = u
        res[corpus] = r
    _save("analysis.json", {"dimensionality": res})
    return res


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "all"
    if task == "tvt_all" and len(sys.argv) > 2:          # e.g. tvt_all cremad 1  (one corpus, n_jobs)
        tvt_all([sys.argv[2]], int(sys.argv[3]) if len(sys.argv) > 3 else -1,
                sys.argv[4].split(",") if len(sys.argv) > 4 else None)
        sys.exit()
    jobs = {"tvt_all": tvt_all, "tvt_split": tvt_split, "separability": separability,
            "dimensionality": dimensionality}
    for name, fn in jobs.items():
        if task in (name, "all"):
            print("==", name, flush=True)
            fn()
