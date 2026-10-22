"""
Experiments behind the voice features of the app (report/09_simulation.md, sections 7, 8 and 9).

    python -m src.voice_experiments calibration    # section 7: calibrate to the speaker's normal voice
    python -m src.voice_experiments corrections    # section 8: learning from the user's corrections
    python -m src.voice_experiments robust         # section 9: microphone-robust training

Every test speaker is new to the models (fold 0 held out, or the other data set). Results are
written to results/phase9/voice_<experiment>.csv. Needs the stored features (notebook 02) and,
for "robust", the simulated-channel features (python -m src.robust_features ...).
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.pipeline import Pipeline

from src import calibration as K
from src.data import PROJECT_ROOT
from src.models import CLASSES, load_corpus, preprocess_steps
from src.personal import Personaliser, WEIGHT

LAYER, C_FUSION, K_BASE = 7, 0.01, 3
OUT = PROJECT_ROOT / "results" / "phase9"
ROBUST = PROJECT_ROOT / "data" / "processed" / "robust"
NEU, HAP, ANG, SAD = (CLASSES.index(c) for c in ["neutral", "happy", "angry", "sad"])
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def fusion(d: dict, idx=None) -> np.ndarray:
    idx = np.arange(len(d["y"])) if idx is None else idx
    return np.hstack([d["egemaps"][idx], d["wavlm"][idx, LAYER].astype(np.float32)])


def logreg(X, y, C=C_FUSION) -> Pipeline:
    return Pipeline(preprocess_steps() + [("clf", LogisticRegression(C=C, max_iter=3000,
                                                                     class_weight="balanced"))]).fit(X, y)


def geo(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    g = np.sqrt(np.clip(a, 1e-9, 1) * np.clip(b, 1e-9, 1))
    return g / g.sum(axis=1, keepdims=True)


def relative_test(X, y, spk, rng):
    """Test clips minus the mean of K_BASE neutral clips of the same speaker; those clips are removed."""
    Xr, keep = X.copy(), np.ones(len(X), bool)
    for s in np.unique(spk):
        idx = np.where(spk == s)[0]
        neu = idx[y[idx] == NEU]
        if len(neu) < K_BASE + 1:
            keep[idx] = False
            continue
        b = rng.choice(neu, K_BASE, replace=False)
        Xr[idx] = X[idx] - X[b].mean(axis=0)
        keep[b] = False
    return Xr, keep


def relative_train(sets, seed=0):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for X, y, spk in sets:
        Xr, ok = K.speaker_relative(X, y, spk, K_BASE, rng)
        Xs.append(Xr[ok]); ys.append(y[ok])
    return np.vstack(Xs), np.concatenate(ys)


def scores(y, p, **extra) -> dict:
    rec = recall_score(y, p, labels=range(len(CLASSES)), average=None)
    out = {"n": len(y), "uar": balanced_accuracy_score(y, p) * 100,
           "share_sad": float((p == SAD).mean()), "happy_as_angry": float((p[y == HAP] == ANG).mean()),
           "angry_as_happy": float((p[y == ANG] == HAP).mean())}
    out.update({f"recall_{c}": round(float(r), 3) for c, r in zip(CLASSES, rec)})
    out.update(extra)
    return out


def speaker_bootstrap(y, pa, pb, spk, n=1000, seed=0):
    """95% interval of UAR(b) - UAR(a), resampling speakers."""
    rng = np.random.default_rng(seed)
    S = np.unique(spk)
    by = {s: np.where(spk == s)[0] for s in S}
    diffs = []
    for _ in range(n):
        idx = np.concatenate([by[s] for s in rng.choice(S, len(S), replace=True)])
        diffs.append(balanced_accuracy_score(y[idx], pb[idx]) - balanced_accuracy_score(y[idx], pa[idx]))
    return np.percentile(diffs, [2.5, 97.5]) * 100


def corpora():
    return {c: load_corpus(c) for c in ["ravdess", "cremad"]}


def splits(C):
    """(name, training (X, y, speaker) per corpus, test corpus, test index)."""
    out = []
    tr = [(c, np.where(C[c]["folds"] != 0)[0]) for c in C]
    for c in C:
        out.append((f"held-out fold 0, {c}", tr, c, np.where(C[c]["folds"] == 0)[0]))
    for s, t in [("ravdess", "cremad"), ("cremad", "ravdess")]:
        out.append((f"train {s} -> test {t}", [(s, np.arange(len(C[s]["y"])))], t, np.arange(len(C[t]["y"]))))
    return out


# ---------------------------------------------------------------------------
# section 7: calibration
# ---------------------------------------------------------------------------
def calibration(reps: int = 10):
    C, rows = corpora(), []
    for name, train, tc, ti in splits(C):
        Xa = np.vstack([fusion(C[c], i) for c, i in train]); ya = np.concatenate([C[c]["y"][i] for c, i in train])
        A = logreg(Xa, ya)
        X, y, spk = fusion(C[tc], ti), C[tc]["y"][ti], C[tc]["groups"][ti]
        for r in range(reps):
            R = logreg(*relative_train([(fusion(C[c], i), C[c]["y"][i], C[c]["groups"][i]) for c, i in train], seed=r))
            Xr, keep = relative_test(X, y, spk, np.random.default_rng(100 + r))
            pa, pr = A.predict_proba(X[keep]), R.predict_proba(Xr[keep])
            for m, p in [("original", pa), ("relative only", pr), ("arithmetic mean", (pa + pr) / 2),
                         ("geometric mean (used)", geo(pa, pr))]:
                rows.append(scores(y[keep], p.argmax(1), test=name, method=m, rep=r))
        print(name, "done", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# section 8: corrections
# ---------------------------------------------------------------------------
def corrections(ms=(1, 3, 5), seeds=(0, 1), weight: float = WEIGHT):
    C, rows = corpora(), []
    for name, train, tc, ti in splits(C):
        if "-> test cremad" in name:      # 30 CREMA-D speakers are enough for the cross-dataset check
            keep_spk = np.random.default_rng(0).choice(np.unique(C[tc]["groups"]), 30, replace=False)
            ti = ti[np.isin(C[tc]["groups"][ti], keep_spk)]
        Xa = np.vstack([fusion(C[c], i) for c, i in train]); ya = np.concatenate([C[c]["y"][i] for c, i in train])
        A = logreg(Xa, ya)
        Xr_tr, yr_tr = relative_train([(fusion(C[c], i), C[c]["y"][i], C[c]["groups"][i]) for c, i in train])
        R = logreg(Xr_tr, yr_tr)
        P = Personaliser(R, Xr_tr, yr_tr)
        X, y, spk = fusion(C[tc], ti), C[tc]["y"][ti], C[tc]["groups"][ti]
        for seed in seeds:
            rng = np.random.default_rng(100 + seed)
            Xr, keep = relative_test(X, y, spk, rng)
            for s in np.unique(spk[keep]):
                order = rng.permutation(np.where((spk == s) & keep)[0])
                pa = A.predict_proba(X[order])
                base = geo(pa, R.predict_proba(Xr[order])).argmax(1)
                used, model, snap = [], R, {}
                for j, i in enumerate(order):                     # the user corrects each mistake
                    if len(used) >= max(ms):
                        break
                    if geo(pa[[j]], model.predict_proba(Xr[[i]])).argmax(1)[0] != y[i]:
                        used.append(i)
                        model = P.fit(Xr[used], y[used], weight)
                        if len(used) in ms:
                            snap[len(used)] = (model, j)
                for m, (model, j) in snap.items():
                    rest = order[j + 1:]
                    if len(rest) < 5:
                        continue
                    p = geo(pa[j + 1:], model.predict_proba(Xr[rest])).argmax(1)
                    rows.append({"test": name, "seed": seed, "speaker": s, "corrections": m,
                                 "y": y[rest].tolist(), "before": base[j + 1:].tolist(), "after": p.tolist()})
        print(name, "done", flush=True)
    # pool clips per test and number of corrections
    out = []
    for (t, m), g in pd.DataFrame(rows).groupby(["test", "corrections"]):
        yy, b, a = (np.concatenate(g[c].map(np.array).values) for c in ["y", "before", "after"])
        out.append(scores(yy, b, test=t, corrections=m, when="before"))
        out.append(scores(yy, a, test=t, corrections=m, when="after"))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# section 9: microphone-robust training
# ---------------------------------------------------------------------------
def load_condition(cond: str, corpus: str, files) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(ROBUST / f"{cond}_{corpus}.npz", allow_pickle=True)
    pos = pd.Series(range(len(z["files"])), index=z["files"])
    i = pos.loc[list(files)].values
    return z["egemaps"][i].astype(np.float32), z["wavlm"][i].astype(np.float32)


def robust(reps: int = 3):
    C, rows = corpora(), []
    aug = {c: load_condition("train_aug", c, C[c]["meta"]["file"]) for c in C}
    aug_fusion = {c: np.hstack([aug[c][0], aug[c][1][:, LAYER]]) for c in C}
    for name, train, tc, ti in splits(C):
        Xc = np.vstack([fusion(C[c], i) for c, i in train]); yc = np.concatenate([C[c]["y"][i] for c, i in train])
        Xg = np.vstack([aug_fusion[c][i] for c, i in train])
        models = {"studio only (current)": logreg(Xc, yc), "studio + simulated mics": logreg(np.vstack([Xc, Xg]),
                                                                                              np.r_[yc, yc])}
        Rs = [logreg(*relative_train([(fusion(C[c], i), C[c]["y"][i], C[c]["groups"][i]) for c, i in train], seed=r))
              for r in range(reps)]
        conds = ["studio"] + (["laptop", "phone", "far_room"] if name.startswith("held-out") else [])
        for cond in conds:
            if cond == "studio":
                X = fusion(C[tc], ti)
            else:
                e, w = load_condition(cond, tc, C[tc]["meta"]["file"].values[ti])
                X = np.hstack([e, w[:, LAYER]])
            y, spk = C[tc]["y"][ti], C[tc]["groups"][ti]
            preds = {m: M.predict(X) for m, M in models.items()}
            lo, hi = speaker_bootstrap(y, preds["studio only (current)"], preds["studio + simulated mics"], spk)
            for m, p in preds.items():
                rows.append(scores(y, p, test=name, condition=cond, model=m, calibrated=False,
                                   gain_ci_low=lo, gain_ci_high=hi))
            for r, R in enumerate(Rs):                            # with "calibrate to my voice"
                Xr, keep = relative_test(X, y, spk, np.random.default_rng(100 + r))
                pr = R.predict_proba(Xr[keep])
                for m, M in models.items():
                    rows.append(scores(y[keep], geo(M.predict_proba(X[keep]), pr).argmax(1), test=name,
                                       condition=cond, model=m, calibrated=True, rep=r))
            print(name, cond, "done", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    what = sys.argv[1]
    df = {"calibration": calibration, "corrections": corrections, "robust": robust}[what]()
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"voice_{what}.csv", index=False)
    key = ["test"] + [c for c in ["condition", "model", "calibrated", "method", "corrections", "when"] if c in df]
    print(df.groupby(key, sort=False)[["uar", "share_sad", "happy_as_angry", "angry_as_happy"]].mean().round(3)
          .to_string())
