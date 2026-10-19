# Phase 6 — Modelling (E0 leakage check, E1 within-corpus benchmark)

Team Lucid · Results from `notebooks/03_modelling.ipynb` (19 Oct 2026). Raw numbers: `results/phase6_summary.json`, `results/phase6/`.

---

## 1. Set-up

| Item | Choice |
|---|---|
| Classes | 6 harmonised emotions (RAVDESS 1,056 clips / 24 speakers; CREMA-D 7,442 clips / 91 speakers) |
| Outer evaluation | The 5 saved **speaker-independent, gender-balanced folds** (Phase 3) |
| Inner tuning | 3-fold **speaker-grouped** CV on the training folds only |
| Models (no KNN) | Logistic Regression (C ∈ {0.01, 0.1, 1}), SVM-RBF (C ∈ {1, 10}), LightGBM (min_child_samples ∈ {10, 30}), MLP 256 units (α ∈ {0.001, 0.1}) |
| Class imbalance | `class_weight="balanced"` (LogReg, SVM, LightGBM); UAR as primary metric |
| Features | eGeMAPS (88) · WavLM base-plus, one layer (768) · fusion (856) |
| Scaling | StandardScaler inside the pipeline, fitted on training data only |
| Metric | **UAR** (mean per-emotion recall); also macro-F1, accuracy. Chance = 16.7% |

## 2. E0 — Speaker leakage inflates accuracy

Same logistic regression; *random split* lets a speaker's clips appear in both training and test.

| Corpus | Features | Random split UAR | Speaker split UAR | Inflation |
|---|---|---|---|---|
| RAVDESS | eGeMAPS | 62.2% | 54.1% | **+8.1 pts** |
| RAVDESS | WavLM L6 | 91.8% | 83.9% | **+7.9 pts** |
| CREMA-D | eGeMAPS | 51.9% | 50.6% | +1.3 pts |
| CREMA-D | WavLM L6 | 74.2% | 72.4% | +1.8 pts |

**Finding (supports H0):** random splits inflate results by about **8 points on RAVDESS** (24 speakers, 60 clips each) and only about **1–2 points on CREMA-D** (91 speakers). The smaller the speaker pool, the more a random split rewards recognising *voices* rather than emotions. This is our own evidence for the pitfall in Paper 6, and it shows that **a RAVDESS score above 90% can be a leakage artefact**.

## 3. Which WavLM layer carries emotion?

The layer was chosen per outer fold by inner CV on training data only (`results/figures/p6_wavlm_layer_profile.png`).

| | Chosen layer per fold | Inner-CV UAR profile |
|---|---|---|
| RAVDESS | 6, 6, 5, 5, 5 | rises from 56% (layer 0) to a **peak at layer 5 (79%)**, then falls to 70% at layer 12 |
| CREMA-D | 6, 6, 5, 7, 7 | rises from 58% to a **plateau at layers 5–7 (71%)**, then falls slightly |

**Finding:** emotion information peaks in the **middle layers (5–7)** of WavLM, consistent with P4 and P7. The top layers lose some of it, plausibly because they specialise in the speech-content objective used in pre-training. The selection was stable across folds.

## 4. E1 — Within-corpus results (mean UAR ± std over 5 speaker-independent folds)

| Corpus | Model | eGeMAPS | WavLM | Fusion |
|---|---|---|---|---|
| RAVDESS | LogReg | 54.6 ± 2.8 | 81.0 ± 5.9 | **82.1 ± 5.5** |
| | SVM | 52.2 ± 4.8 | 78.5 ± 7.5 | 78.4 ± 8.8 |
| | LightGBM | 52.1 ± 4.0 | 76.8 ± 7.9 | 76.5 ± 7.0 |
| | MLP | 51.4 ± 3.8 | 78.6 ± 8.1 | 81.1 ± 4.0 |
| CREMA-D | LogReg | 51.2 ± 3.1 | 73.8 ± 1.9 | **73.9 ± 2.2** |
| | SVM | 51.9 ± 3.6 | 73.4 ± 3.0 | 73.1 ± 3.2 |
| | LightGBM | **53.3 ± 3.9** | 70.3 ± 3.2 | 70.2 ± 4.7 |
| | MLP | 52.2 ± 2.7 | 72.7 ± 2.1 | 72.1 ± 3.4 |

Macro-F1 and accuracy track UAR within ±2 points (see `e1_summary.csv`).

### Findings
1. **WavLM beats eGeMAPS by a large margin within each corpus:** about **+26 pts on RAVDESS** and **+22 pts on CREMA-D** (LogReg). This holds in **every one of the 5 folds** on both corpora. With only 5 folds a paired test cannot reach p < 0.05 (the minimum Wilcoxon p is 0.0625), so we report it as a 5/5-fold consistent difference. This replicates P4 and P5 and is the first half of H1.
2. **Fusion adds almost nothing** within-corpus (+0.1 to +1.1 pts for LogReg). WavLM already seems to capture what eGeMAPS measures, consistent with P5's probing results. Whether eGeMAPS adds value **across** corpora is tested in Phase 8.
3. **Simple models are enough.** Logistic Regression is the best or tied-best for WavLM and fusion. The non-linear models (SVM, LightGBM, MLP) did not help on 768-dimensional embeddings, and LightGBM was the weakest there. With eGeMAPS, LightGBM is slightly ahead on CREMA-D (53.3 vs 51.2). That fits theory: eGeMAPS is a small set of hand-designed descriptors where feature interactions can matter, while self-supervised embeddings are close to linearly separable.
4. **Variation across folds is larger on RAVDESS** (±5–9 pts vs ±2–3 on CREMA-D), because each RAVDESS test fold holds only 4–5 speakers. Single-split RAVDESS results in the literature should be read with this in mind.

### Confusions (best model: LogReg + fusion, out-of-fold)
- RAVDESS: angry 89%, disgust 88%, neutral 85%; **fearful and neutral are confused with sad** (13% and 11%).
- CREMA-D: neutral 86%, angry 82%, happy 76%; **fearful → sad 17%**, **disgust → sad 12%**, **sad → fearful/disgust 11%/10%**.
- **Sad is the "sink" class** in both corpora: low-energy emotions get pulled towards it. On CREMA-D, this is also where human listeners disagreed most (16% agreement for sad, Phase 3).

### A caution about the human reference lines
On CREMA-D the model's UAR (≈74%) is far above the 41% human audio-only agreement. These numbers are **not directly comparable**:
1. Human raters labelled each clip once without training on these actors' style, while the model is trained on thousands of clips from the same corpus.
2. The human figure measures agreement with the *intended* label from a single listen.
3. The model may exploit acoustic regularities of this recording setup that won't carry over to new recordings, which is exactly what the cross-corpus test (Phase 8) will probe.

We therefore present human rates only as **context**, not as a ceiling the model "beat".

## 5. What this means for the research question
- **In-corpus**, the pretrained representation wins clearly, and a linear model is sufficient.
- That makes the cross-corpus test (Phase 8, E2) decisive. If WavLM's ~22–26 point advantage shrinks or reverses when training on RAVDESS and testing on CREMA-D (or the reverse), H1 is supported.
- **Logistic Regression is our main model** for the explainability phases: it is the best performer, and SHAP explanations for it are **exact** (LinearExplainer). LightGBM stays as the non-linear comparison, also with exact TreeSHAP.

## 6. Limitations
- One model seed per configuration. LogReg and SVM are deterministic; seed variation for LightGBM and MLP is added in Phase 8.
- Small hyperparameter grids, chosen to keep runtime under an hour on a laptop.
- Speaker normalisation (P4) was not used, by design: it requires test-speaker statistics.
