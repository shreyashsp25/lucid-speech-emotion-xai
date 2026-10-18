# Paper 10 — Explanation Multiplicity in SHAP

**Citation:** Hwang, H., Lee, S., Rosenblatt, L., Whang, S. E., & Stoyanovich, J. (2026). Explanation Multiplicity in SHAP: Characterization and Assessment. *arXiv preprint* arXiv:2601.12654.

**Free PDF:** https://arxiv.org/abs/2601.12654
**Type:** General XAI methodology (not SER-specific)

---

## What they did (plain language)
They showed that if you run SHAP twice on the same model and the same input, you can get different explanations — and if you retrain the same model with a different random seed, explanations change even more. They call this "explanation multiplicity" and give a way to measure it.

## Datasets / models
Multiple tabular datasets and model classes (specific names: see PDF). Not speech.

## Method
- **Dual-seed protocol**: separates variability caused by the *model* (retraining seed) from variability caused by the *explainer* (SHAP sampling seed).
- Metrics: magnitude-based (L2 distance), **rank-based** (changes in top features), set-based.
- Random null models (Dirichlet, Mallows) as reference scales.

## Key findings
- Explanation multiplicity is **widespread**, even for high-confidence predictions.
- **L2 distance can look small while the identity and order of top features change** → rank-based metrics reveal instability that magnitude metrics hide.
- Model-induced disagreement larger on smaller datasets; explainer-induced larger on larger datasets.
- Background compression (K-Means) reduces run-to-run variation but can shift explanations away from the reference.

## Evaluation protocol / results
Repeated runs with controlled seeds; numeric results in paper tables.

## Limitations
Not speech; general tabular setting — applicability to SER is untested.

## Relevance to Team Lucid
- SER papers (05, 09) report **one SHAP ranking from one run**. This paper shows single-run rankings can be unreliable.
- **Small datasets → more model-induced variation** — RAVDESS (1,440 clips, 24 speakers) is exactly that case.
- Gives us **measurement tools** (rank correlation, top-k overlap, dual seeds) to test whether SER explanations are stable across folds, seeds and corpora.

## Viva-ready points
- "Why trust your SHAP plot?" → We don't by default; we measure its stability across seeds/folds/corpora using rank-based metrics.
- "Magnitude vs rank metrics?" → Magnitudes can agree while the top-feature list changes; ranks are what users read.
