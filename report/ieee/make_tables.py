"""
LaTeX tables for the IEEE report, generated from the results files so that no number
is typed by hand. Run from the project root:

    python report/ieee/make_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "report" / "ieee" / "tables"
TAB.mkdir(parents=True, exist_ok=True)
RES = ROOT / "results"
CLASSES = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]
f1 = lambda v: f"{v:.1f}"


def load_tvt() -> tuple[pd.DataFrame, float]:
    """Train/validation from results/review; test = the official Phase 6 outer-fold scores.
    Returns the table and the largest |re-run test UAR - Phase 6 test UAR| in points."""
    df = pd.read_csv(RES / "review" / "train_val_test_all_models.csv")
    e1 = pd.read_csv(RES / "phase6" / "e1_all_folds.csv")[["corpus", "features", "model", "fold", "uar", "macro_f1", "accuracy"]]
    m = df.merge(e1, on=["corpus", "features", "model", "fold"], how="left")
    diff = float((m["test_uar"] - m["uar"]).abs().max() * 100)
    m["test_uar"], m["test_macro_f1"], m["test_accuracy"] = m["uar"], m["macro_f1"], m["accuracy"]
    return m.drop(columns=["uar", "macro_f1", "accuracy"]), diff


def tab_tvt():
    df, diff = load_tvt()
    print("max re-run deviation of test UAR (points):", round(diff, 2))
    g = df.groupby(["corpus", "features", "model"])
    rows = []
    best = {}
    for c in ["ravdess", "cremad"]:
        sub = df[df.corpus == c].groupby(["features", "model"])["test_uar"].mean()
        best[c] = sub.idxmax() if len(sub) else None
    names = {"logreg": "LogReg", "svm": "SVM (RBF)", "lgbm": "LightGBM", "mlp": "MLP"}
    feats = {"egemaps": "eGeMAPS", "wavlm": "WavLM", "fusion": "Fusion"}
    lines = [r"\begin{table*}[!t]", r"\centering",
             r"\caption{Train, validation and test scores of all 24 models (mean $\pm$ SD over 5 speaker-independent outer folds, \%). "
             r"Train: refitted model on its own training folds; validation: inner 3-fold speaker-grouped CV of the chosen settings; "
             r"test: unseen outer fold (the Phase 6 results; refitting with the same settings reproduced them within "
             + f"{diff:.1f}" + r" points). Bold: best test UAR per corpus.}",
             r"\label{tab:tvt}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}ll ccc cc @{\hspace{14pt}} ccc cc@{}}", r"\toprule",
             r"& & \multicolumn{5}{c}{\textbf{RAVDESS}} & \multicolumn{5}{c}{\textbf{CREMA-D}} \\",
             r"\cmidrule(lr){3-7}\cmidrule(l){8-12}",
             r"\textbf{Features} & \textbf{Model} & Train UAR & Val.\ UAR & Test UAR & Test F1 & Test Acc. & Train UAR & Val.\ UAR & Test UAR & Test F1 & Test Acc. \\",
             r"\midrule"]
    for fi, feat in enumerate(["egemaps", "wavlm", "fusion"]):
        for mi, model in enumerate(["logreg", "svm", "lgbm", "mlp"]):
            cells = [feats[feat] if mi == 0 else "", names[model]]
            for c in ["ravdess", "cremad"]:
                r = df[(df.corpus == c) & (df.features == feat) & (df.model == model)]
                if len(r) < 5:
                    cells += ["--"] * 5
                    continue
                m = lambda col: r[col].mean() * 100
                s = lambda col: r[col].std(ddof=1) * 100
                test = f"{m('test_uar'):.1f} $\\pm$ {s('test_uar'):.1f}"
                if best[c] == (feat, model):
                    test = r"\textbf{" + test + "}"
                cells += [f"{m('train_uar'):.1f}", f"{m('val_uar'):.1f} $\\pm$ {s('val_uar'):.1f}", test,
                          f"{m('test_macro_f1'):.1f}", f"{m('test_accuracy'):.1f}"]
            lines.append(" & ".join(cells) + r" \\")
        if fi < 2:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TAB / "tab_tvt.tex").write_text("\n".join(lines) + "\n")
    # consolidated CSV for the submission
    g = df.groupby(["corpus", "features", "model"])
    out = g[[c for c in df.columns if c.startswith(("train_", "val_", "test_"))]].agg(["mean", "std"]) * 100
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out.round(2).reset_index().to_csv(RES / "review" / "train_val_test_summary_percent.csv", index=False)
    js = {}
    for (c, feat, model), r in df.groupby(["corpus", "features", "model"]):
        js.setdefault(c, {}).setdefault(feat, {})[model] = {
            "train": r.train_uar.mean() * 100, "val": r.val_uar.mean() * 100,
            "test": r.test_uar.mean() * 100, "test_sd": r.test_uar.std(ddof=1) * 100, "folds": int(len(r))}
    (RES / "review" / "train_val_test_summary.json").write_text(json.dumps(js, indent=1))
    print("tab_tvt", best)


def tab_pair():
    A = json.loads((RES / "review" / "analysis.json").read_text())["separability"]
    ab = ["neu", "hap", "sad", "ang", "fea", "dis"]
    lines = [r"\begin{table}[!t]", r"\centering",
             r"\caption{Pairwise separability: speaker-independent ROC-AUC of a binary LogReg on fused features. "
             r"Upper triangle RAVDESS, lower triangle CREMA-D}",
             r"\label{tab:pair}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule", " & " + " & ".join(ab) + r" \\", r"\midrule"]
    R, C = A["ravdess"]["pair_auc"], A["cremad"]["pair_auc"]
    for i in range(6):
        row = [ab[i]]
        for j in range(6):
            if i == j:
                row.append("--")
            else:
                v = R[i][j] if j > i else C[i][j]
                row.append(f"{v:.2f}" if v >= 0.95 else r"\textbf{" + f"{v:.2f}" + "}")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\multicolumn{7}{@{}l}{\scriptsize Bold: AUC below 0.95.}", r"\end{tabular}", r"\end{table}"]
    (TAB / "tab_pair.tex").write_text("\n".join(lines) + "\n")
    print("tab_pair")


def tab_prf():
    J = json.loads((RES / "phase8_e6_summary.json").read_text())["classification_reports"]
    lines = [r"\begin{table}[!t]", r"\centering",
             r"\caption{Precision / recall / F1 of the best model (LogReg, fusion), out-of-fold on unseen speakers}",
             r"\label{tab:prf}", r"\footnotesize", r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
             r"\textbf{Emotion} & \textbf{RAVDESS} & \textbf{CREMA-D} \\", r"\midrule"]
    from decimal import Decimal, ROUND_HALF_UP
    r2 = lambda v: str(Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    fmt = lambda d: f"{r2(d['precision'])} / {r2(d['recall'])} / {r2(d['f1-score'])}"
    for e in CLASSES:
        lines.append(f"{e} & {fmt(J['ravdess_fusion'][e])} & {fmt(J['cremad_fusion'][e])} \\\\")
    lines += [r"\midrule", f"macro & {fmt(J['ravdess_fusion']['macro avg'])} & {fmt(J['cremad_fusion']['macro avg'])} \\\\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "tab_prf.tex").write_text("\n".join(lines) + "\n")
    print("tab_prf")




# ---------------------------------------------------------------------------
# Consolidated results for the submission (results/consolidated_results.md + .csv)
# ---------------------------------------------------------------------------
def consolidated():
    md = ["# Consolidated results — Team Lucid", "",
          "All numbers are speaker-independent (no speaker in both training and test data). UAR = mean per-emotion recall; chance = 16.7%.",
          "Generated by `python report/ieee/make_tables.py` from the files named under each table.", ""]
    df, _ = load_tvt()
    agg = df.groupby(["corpus", "features", "model"]).agg(
        **{f"{p}_{m}": (f"{p}_{m}", "mean") for p in ["train", "val", "test"] for m in ["uar", "macro_f1", "accuracy"]},
        test_uar_sd=("test_uar", lambda x: x.std(ddof=1))).reset_index()
    for c in [x for x in agg.columns if x not in ("corpus", "features", "model")]:
        agg[c] = (agg[c] * 100).round(1)
    agg.to_csv(RES / "consolidated_metrics.csv", index=False)
    md += ["## 1. All 24 models: train / validation / test", "",
           "Train = refitted model on its own training folds; validation = inner 3-fold speaker-grouped CV score of the chosen settings; "
           "test = unseen outer fold. Mean over 5 outer folds (%). Source: `results/review/train_val_test_all_models.csv`.", "",
           "| Corpus | Features | Model | Train UAR | Val UAR | Test UAR (± SD) | Test macro-F1 | Test accuracy |", "|---|---|---|---|---|---|---|---|"]
    for _, r in agg.sort_values(["corpus", "features", "model"], ascending=[False, True, True]).iterrows():
        md.append(f"| {r.corpus} | {r.features} | {r.model} | {r.train_uar} | {r.val_uar} | {r.test_uar} ± {r.test_uar_sd} | {r.test_macro_f1} | {r.test_accuracy} |")
    A = json.loads((RES / "review" / "analysis.json").read_text())
    md += ["", "## 2. Chosen models on an explicit 3-1-1 fold split", "",
           "Fold k = test, fold k+1 = validation, other three = train (k = 0..4). Source: `results/review/analysis.json` (`tvt_split`).", "",
           "| Corpus | Model | Train UAR | Val UAR | Test UAR |", "|---|---|---|---|---|"]
    for c in ["ravdess", "cremad"]:
        for name, lab in [("fusion", "Predictor: LogReg, fusion, C=0.01"), ("egemaps", "Explainer: LogReg, eGeMAPS, C=1")]:
            t = A["tvt_split"][c][name]
            md.append(f"| {c} | {lab} | " + " | ".join(f"{t[s]['uar'][0]:.1f} ± {t[s]['uar'][1]:.1f}" for s in ["train", "val", "test"]) + " |")
    J = json.loads((RES / "phase8_e6_summary.json").read_text())
    md += ["", "## 3. Best model per emotion (LogReg, fusion, out-of-fold)", "", "Source: `results/phase8_e6_summary.json`.", "",
           "| Emotion | RAVDESS P / R / F1 | CREMA-D P / R / F1 |", "|---|---|---|"]
    from decimal import Decimal, ROUND_HALF_UP
    r2 = lambda v: str(Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    f = lambda d: f"{r2(d['precision'])} / {r2(d['recall'])} / {r2(d['f1-score'])}"
    for e in CLASSES + ["macro avg"]:
        md.append(f"| {e} | {f(J['classification_reports']['ravdess_fusion'][e])} | {f(J['classification_reports']['cremad_fusion'][e])} |")
    md += ["", f"ECE: {J['calibration_ece']}", ""]
    md += ["| Corpus | Features | Model | Macro ROC-AUC |", "|---|---|---|---|"] + \
          [f"| {r['corpus']} | {r['features']} | {r['model']} | {r['macro_auc']:.3f} |" for r in J["roc_auc"]]
    lk = pd.read_csv(RES / "phase6" / "e0_leakage.csv")
    md += ["", "## 4. Speaker leakage check (H0)", "", "Source: `results/phase6/e0_leakage.csv`.", "",
           "| Corpus | Features | Random split UAR | Speaker split UAR | Inflation (points) |", "|---|---|---|---|---|"] + \
          [f"| {r.corpus} | {r.features} | {r.random_split_uar*100:.1f} | {r.speaker_split_uar*100:.1f} | +{r.inflation_pts:.1f} |" for r in lk.itertuples()]
    e2 = pd.read_csv(RES / "phase8" / "e2_all.csv")
    md += ["", "## 5. Cross-corpus transfer, all models (H1)", "", "Train on all of one corpus, test on all of the other. Source: `results/phase8/e2_all.csv`.", "",
           "| Source → target | Features | Model | Scaling | Cross UAR | In-domain UAR | Drop (points) |", "|---|---|---|---|---|---|---|"] + \
          [f"| {r.source} → {r.target} | {r.features} | {r.model} | {r.norm} | {r.cross_uar*100:.1f} | {r.indomain_uar*100:.1f} | {r.drop_pts:.1f} |" for r in e2.itertuples()]
    h2 = pd.read_csv(RES / "phase8" / "e3_h2_test.csv")
    md += ["", "## 6. Explanation stability (H2) and stable-core selection (H3)", "", "Source: `results/phase8/e3_h2_test.csv`, `e4_h3_test.csv`.", "",
           "| Representation | Resamples | Speaker groups | Corpora | Random |", "|---|---|---|---|---|"] + \
          [f"| {r.representation} | {r.median_resample:.2f} | {r.median_speaker:.2f} | {r.median_corpus:.2f} | {r.median_random:.2f} |" for r in h2.itertuples()]
    h3 = pd.read_csv(RES / "phase8" / "e4_h3_test.csv")
    md += ["", "| Direction | Scaling | k | Stable core − top-k (points) | 95% CI | Significant |", "|---|---|---|---|---|---|"] + \
          [f"| {r.direction} | {r.norm} | {r.k} | {r.stable_minus_top_pts:+.2f} | {r.ci95} | {r.significant} |" for r in h3.itertuples()]
    S, D = A["separability"], A["dimensionality"]
    md += ["", "## 7. Class separability, dimensionality and correlation", "", "Source: `results/review/analysis.json`.", "",
           "| | RAVDESS | CREMA-D |", "|---|---|---|",
           f"| Silhouette (cosine) eGeMAPS / WavLM / fusion | {' / '.join(f'{v:.3f}' for v in S['ravdess']['silhouette_cosine'].values())} | {' / '.join(f'{v:.3f}' for v in S['cremad']['silhouette_cosine'].values())} |",
           f"| Lowest pairwise AUC | {min(v for r in S['ravdess']['pair_auc'] for v in r if v):.3f} | {min(v for r in S['cremad']['pair_auc'] for v in r if v):.3f} |",
           f"| eGeMAPS PCA components for 90 / 95 / 99% variance | {D['ravdess']['pca_egemaps']['k90']} / {D['ravdess']['pca_egemaps']['k95']} / {D['ravdess']['pca_egemaps']['k99']} | {D['cremad']['pca_egemaps']['k90']} / {D['cremad']['pca_egemaps']['k95']} / {D['cremad']['pca_egemaps']['k99']} |",
           f"| WavLM L7 PCA components for 90 / 95 / 99% | {D['ravdess']['pca_wavlm']['k90']} / {D['ravdess']['pca_wavlm']['k95']} / {D['ravdess']['pca_wavlm']['k99']} | {D['cremad']['pca_wavlm']['k90']} / {D['cremad']['pca_wavlm']['k95']} / {D['cremad']['pca_wavlm']['k99']} |",
           f"| eGeMAPS pairs with abs(r) > 0.8 / > 0.9 (of 3,828) | {D['ravdess']['corr']['gt08']} / {D['ravdess']['corr']['gt09']} | {D['cremad']['corr']['gt08']} / {D['cremad']['corr']['gt09']} |",
           f"| UAR with WavLM PCA k = 16 / 64 / 768 | {D['ravdess']['pca_uar_wavlm']['16']:.1f} / {D['ravdess']['pca_uar_wavlm']['64']:.1f} / {D['ravdess']['pca_uar_wavlm']['768']:.1f} | {D['cremad']['pca_uar_wavlm']['16']:.1f} / {D['cremad']['pca_uar_wavlm']['64']:.1f} / {D['cremad']['pca_uar_wavlm']['768']:.1f} |",
           f"| UAR with eGeMAPS PCA k = 10 / 40 / 88 | {D['ravdess']['pca_uar_egemaps']['10']:.1f} / {D['ravdess']['pca_uar_egemaps']['40']:.1f} / {D['ravdess']['pca_uar_egemaps']['88']:.1f} | {D['cremad']['pca_uar_egemaps']['10']:.1f} / {D['cremad']['pca_uar_egemaps']['40']:.1f} / {D['cremad']['pca_uar_egemaps']['88']:.1f} |"]
    fh = pd.read_csv(RES / "phase9" / "final_recipe_heldout.csv")
    md += ["", "## 8. Saved models", "",
           "| File | What it is |", "|---|---|",
           "| `models/final_models.joblib` | predictor (LogReg on fusion, WavLM layer 7, C=0.01) + explainer (LogReg on eGeMAPS, C=1), trained on all 8,498 clips of 115 speakers |",
           "| `models/heldout_fold0.joblib` | the same recipe trained without fold-0 speakers; used by the demo for dataset clips |",
           "| `models/final_models_robust.joblib` | microphone-robust variant used for uploads and recordings |",
           "| `models/calibration_*.joblib` | speaker-calibration models |", "",
           "Same recipe with fold-0 speakers held out (source: `results/phase9/final_recipe_heldout.csv`):", "",
           "| Test corpus | Model | Speakers | UAR | Macro-F1 |", "|---|---|---|---|---|"] + \
          [f"| {r['test corpus']} | {r.model} | {r.n_speakers} | {r.UAR*100:.1f} | {r.macro_F1*100:.1f} |" for _, r in fh.iterrows()]
    (RES / "consolidated_results.md").write_text("\n".join(md) + "\n")
    print("consolidated")


if __name__ == "__main__":
    for fn in (tab_pair, tab_prf, tab_tvt, consolidated):
        try:
            fn()
        except FileNotFoundError as e:
            print("skipped", fn.__name__, e)
