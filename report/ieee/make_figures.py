"""
Colour figures for the IEEE report (report/ieee/figures/*.pdf).

Run from the project root after `python -m src.review_analysis all`:

    python report/ieee/make_figures.py

Needs: data/raw (Clip A only), data/processed, models/heldout_fold0.joblib,
results/phase6, results/phase8 and results/review.
"""
from __future__ import annotations

import json
import sys
from math import gcd
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models import CLASSES, load_corpus  # noqa: E402

FIG = ROOT / "report" / "ieee" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
REV = ROOT / "results" / "review"
for f in font_manager.findSystemFonts():
    if "texgyretermes" in f.lower():
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.0,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
COL = 3.45          # IEEE column width, inches
K, G, L = "#1B1A3A", "#8A88A6", "#D9D7E3"   # ink, grey, light grey
TEAL, CORAL, BLUE, VIOLET, AMBER, GREEN = "#14958A", "#E0533D", "#3D6FD6", "#6B5BD6", "#D4941A", "#2E9A68"
EMO = {"neutral": "#9A98B5", "happy": AMBER, "sad": BLUE, "angry": CORAL, "fearful": VIOLET, "disgust": GREEN}
MARK = {"neutral": "o", "happy": "^", "sad": "s", "angry": "X", "fearful": "D", "disgust": "P"}
CLIP_A = "03-01-05-02-01-02-09.wav"


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def fig_preprocessing():
    import librosa
    import soundfile as sf
    from scipy.signal import resample_poly
    y, sr = sf.read(str(ROOT / "data/raw/ravdess/Actor_09" / CLIP_A), dtype="float32", always_2d=True)
    y = y.mean(1)
    g = gcd(sr, 16000)
    y = resample_poly(y, 16000 // g, sr // g)
    _, (a, b) = librosa.effects.trim(y, top_db=30)
    t = np.arange(len(y)) / 16000
    fig, ax = plt.subplots(figsize=(COL, 1.15))
    ax.axvspan(0, a / 16000, color=CORAL, alpha=0.12, lw=0)
    ax.axvspan(b / 16000, t[-1], color=CORAL, alpha=0.12, lw=0)
    ax.plot(t, y, color=G, lw=0.35)
    ax.plot(t[a:b], y[a:b], color=TEAL, lw=0.35)
    ax.set_xlim(0, t[-1]); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("time (s)")
    ax.text(0.05, 0.8, "silence", transform=ax.transAxes, color=CORAL)
    ax.text(0.86, 0.8, "silence", transform=ax.transAxes, color=CORAL)
    ax.text(a / 16000 + 0.05, -0.85 * np.abs(y).max(), f"kept: {(b - a) / 16000:.2f} s", fontsize=7)
    save(fig, "fig_preprocessing")


def fig_layers():
    fig, ax = plt.subplots(figsize=(COL, 1.55))
    for corpus, m, ls, col in [("ravdess", "o", "-", CORAL), ("cremad", "s", "-", TEAL)]:
        js = json.loads((ROOT / f"results/phase6/layers_{corpus}.json").read_text())
        v = js["profile"]
        v = np.asarray(v, float)
        v = v * 100 if v.max() <= 1 else v
        ax.plot(range(13), v, color=col, marker=m, ms=3, ls=ls, mfc="white",
                label="RAVDESS" if corpus == "ravdess" else "CREMA-D")
    ax.set_xlabel("WavLM layer"); ax.set_ylabel("inner-CV UAR (%)"); ax.set_xticks(range(13))
    ax.legend(frameon=False, loc="lower right"); ax.grid(axis="y", color=L, lw=0.4)
    save(fig, "fig_layers")


def fig_dimensionality():
    A = json.loads((REV / "analysis.json").read_text())["dimensionality"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 1.7))
    for corpus, ls, col in [("ravdess", "-", CORAL), ("cremad", "--", TEAL)]:
        name = "RAVDESS" if corpus == "ravdess" else "CREMA-D"
        for feat, m in [("egemaps", "o"), ("wavlm", "s")]:
            cum = np.asarray(A[corpus][f"pca_{feat}"]["cum"]) * 100
            ks = [k for k in [1, 2, 5, 10, 20, 40, 60, 80, 100, 120] if k <= len(cum)]
            a1.plot(ks, [cum[k - 1] for k in ks], color=col if feat == "wavlm" else G, ls=ls, marker=m, ms=2.5, mfc="white",
                    label=f"{name} {'eGeMAPS' if feat == 'egemaps' else 'WavLM'}")
        u = A[corpus]["pca_uar_wavlm"]
        a2.plot(list(map(int, u)), list(u.values()), color=col, ls=ls, marker="s", ms=2.5, mfc="white", label=f"{name} WavLM")
        u = A[corpus]["pca_uar_egemaps"]
        a2.plot(list(map(int, u)), list(u.values()), color=G, ls=ls, marker="o", ms=2.5, mfc="white", label=f"{name} eGeMAPS")
    a1.set_xscale("log"); a1.set_xlabel("principal components"); a1.set_ylabel("variance explained (%)")
    a1.axhline(95, color=G, lw=0.5, ls=":"); a1.legend(frameon=False, ncol=1, fontsize=6)
    a1.set_title("(a) cumulative variance")
    a2.set_xscale("log"); a2.set_xlabel("components kept (PCA inside training folds)"); a2.set_ylabel("test UAR (%)")
    a2.axhline(100 / 6, color=G, lw=0.5, ls=":"); a2.text(2.2, 18.5, "chance", color=G, fontsize=6)
    a2.legend(frameon=False, fontsize=6, loc="lower right"); a2.set_title("(b) UAR after PCA reduction")
    for a in (a1, a2):
        a.grid(color=L, lw=0.3)
    save(fig, "fig_dimensionality")


def fig_correlation():
    from matplotlib.colors import LinearSegmentedColormap
    C = np.nan_to_num(np.load(REV / "corr_ravdess.npy"))
    cmap = LinearSegmentedColormap.from_list("d", [BLUE, "white", CORAL])
    fig, ax = plt.subplots(figsize=(COL * 0.62, COL * 0.56))
    im = ax.imshow(C, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.4)
    ax.set_xlabel("88 eGeMAPS features (RAVDESS)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("Pearson r"); cb.outline.set_linewidth(0.4)
    save(fig, "fig_correlation")


def fig_lda():
    z = np.load(REV / "lda_ravdess.npz", allow_pickle=True)
    Z, y, f, clip = z["Z"], z["y"], z["fold"], z["clip"]
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    for i, c in enumerate(CLASSES):
        m = (y == i) & (f != 0)
        ax.scatter(Z[m, 0], Z[m, 1], s=5, marker="o", facecolor=EMO[c], alpha=0.25, edgecolor="none")
    for i, c in enumerate(CLASSES):
        m = (y == i) & (f == 0)
        ax.scatter(Z[m, 0], Z[m, 1], s=14, marker=MARK[c], facecolor=EMO[c], edgecolor="white", lw=0.4, label=c)
    j = int(np.where(clip == CLIP_A)[0][0])
    ax.scatter(Z[j, 0], Z[j, 1], s=110, facecolor="none", edgecolor=K, lw=1.2)
    ax.annotate("Clip A", (Z[j, 0], Z[j, 1]), xytext=(9, 6), textcoords="offset points", fontsize=7, weight="bold")
    ax.set_xlabel("LD 1"); ax.set_ylabel("LD 2")
    ax.legend(frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.05, columnspacing=0.5, fontsize=6.5)
    save(fig, "fig_lda")


def fig_confusion():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.0))
    for ax, corpus, name in zip(axes, ["ravdess", "cremad"], ["RAVDESS", "CREMA-D"]):
        d = load_corpus(corpus)
        p = np.load(ROOT / f"results/phase6/e1_{corpus}_fusion_logreg_oof.npy")
        cm = pd.crosstab(pd.Series(d["y"], name="t"), pd.Series(p, name="p")).reindex(index=range(6), columns=range(6), fill_value=0).values
        cm = cm / cm.sum(1, keepdims=True)
        from matplotlib.colors import LinearSegmentedColormap
        ax.imshow(cm, cmap=LinearSegmentedColormap.from_list("t", ["white", "#BFE6E1", TEAL, K]), vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{cm[i, j] * 100:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if cm[i, j] > 0.5 else "black")
        ax.set_xticks(range(6)); ax.set_xticklabels(CLASSES, rotation=35, ha="right")
        ax.set_yticks(range(6)); ax.set_yticklabels(CLASSES if ax is axes[0] else [])
        ax.set_xlabel("predicted"); ax.set_title(f"({'ab'[corpus == 'cremad']}) {name}")
        if ax is axes[0]:
            ax.set_ylabel("performed")
    save(fig, "fig_confusion")


def fig_shap_clip_a():
    m = joblib.load(ROOT / "models/heldout_fold0.joblib")
    d = load_corpus("ravdess")
    i = int(np.where(d["meta"]["file"].values == CLIP_A)[0][0])
    pipe = m["explainer"]
    Z = pipe[:-1].transform(d["egemaps"][i:i + 1])[0]
    w = pipe[-1].coef_[CLASSES.index("angry")]
    phi = (Z - m["explainer_train_mean_z"]) * w
    names = np.array(m["egemaps_names"])
    order = np.argsort(-np.abs(phi))[:8][::-1]
    short = {"alphaRatioUV_sma3nz_amean": "alpha ratio (unvoiced)", "loudness_sma3_meanRisingSlope": "loudness rising slope",
             "loudness_sma3_amean": "loudness, mean", "equivalentSoundLevel_dBp": "equivalent sound level",
             "mfcc2_sma3_amean": "MFCC 2, mean", "F1frequency_sma3nz_amean": "F1 frequency, mean",
             "alphaRatioV_sma3nz_amean": "alpha ratio (voiced)", "hammarbergIndexUV_sma3nz_amean": "Hammarberg index (unvoiced)"}
    labels = [short.get(n, n.replace("_sma3nz", "").replace("_sma3", "")) for n in names[order]]
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    ax.barh(range(len(order)), phi[order], color=[CORAL if v > 0 else BLUE for v in phi[order]],
            edgecolor="none", height=0.62)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(labels)
    ax.axvline(0, color=K, lw=0.6)
    for k, v in enumerate(phi[order]):
        ax.text(v + (0.08 if v > 0 else -0.08), k, f"{v:+.2f}", va="center", ha="left" if v > 0 else "right", fontsize=6.5)
    ax.set_xlabel("SHAP value for “angry” (log-odds)")
    lim = np.abs(phi[order]).max() * 1.35
    ax.set_xlim(-lim, lim)
    save(fig, "fig_shap_clip_a")
    total = float(phi.sum())
    base = float(pipe[-1].intercept_[CLASSES.index("angry")] + (m["explainer_train_mean_z"] * w).sum())
    print("Clip A angry: base", round(base, 2), "sum phi", round(total, 2), "score", round(base + total, 2))


def fig_stability():
    labels = ["resamples", "speaker groups", "corpora", "random"]
    vals = {"LogReg, eGeMAPS": [0.78, 0.63, 0.50, 0.0], "LightGBM, eGeMAPS": [0.76, 0.44, 0.20, 0.0],
            "LogReg, WavLM": [0.68, 0.34, 0.10, 0.0]}
    fig, ax = plt.subplots(figsize=(COL, 1.6))
    x = np.arange(4)
    for k, ((name, v), fc) in enumerate(zip(vals.items(), [VIOLET, CORAL, TEAL])):
        bars = ax.bar(x + (k - 1) * 0.26, v, 0.26, color=fc, edgecolor="none", label=name)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", fontsize=5.5)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("median Spearman")
    ax.set_ylim(0, 0.95); ax.legend(frameon=False, fontsize=6)
    save(fig, "fig_stability")


if __name__ == "__main__":
    for fn in [fig_preprocessing, fig_layers, fig_shap_clip_a, fig_stability, fig_confusion,
               fig_correlation, fig_dimensionality, fig_lda]:
        try:
            fn()
        except FileNotFoundError as e:
            print("skipped", fn.__name__, e)
