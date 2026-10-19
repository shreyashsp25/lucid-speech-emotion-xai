# Phase 8 — Experiments

Team Lucid · Results from `notebooks/05_experiments.ipynb` (run on our MacBook) and `notebooks/06_evaluation.ipynb`. Raw numbers: `results/phase8_summary.json`, `results/phase8_e6_summary.json`, `results/phase8/`.

All results use **speaker-independent** evaluation. Every choice (WavLM layer, hyperparameters, feature subset) is made on **source / training speakers only**. Confidence intervals (95%) come from a **speaker-level bootstrap** (1,000 resamples of target speakers), because clips from one speaker are not independent.

---

## 1. Overview

| Exp. | Question | Hypothesis | Verdict |
|---|---|---|---|
| E2 | How much is lost when a model trained on one corpus is tested on the other? | **H1:** WavLM loses relatively more than eGeMAPS | **Depends on direction** |
| E3 | Are SHAP explanations stable across resamples, speakers and corpora? | **H2:** resample > speaker > corpus > random | **Supported** (all 3 models) |
| E4 | Do features that are *consistently* important transfer better than the usual top-k? | **H3:** stable-core subset transfers better | **Inconclusive** |
| E5 | Where and for whom do the models fail? | (descriptive) | see §5 |
| E6 | Standard evaluation: precision/recall/F1, ROC-AUC, calibration, overfitting, learning curve | (descriptive) | see §6 |

Two ways of normalising features are compared throughout E2 and E4:
- **source:** the scaler is fitted on the source corpus only (strict; the target is completely unseen).
- **corpus:** each corpus is first standardised with **its own** mean and standard deviation. This uses the *unlabelled* target features, never target labels. It is a simple, widely used unsupervised adaptation.

## 2. E2 — Cross-corpus transfer (H1)

Logistic Regression, UAR in % with 95% CI. In-domain (Phase 6) UAR for reference: RAVDESS eGeMAPS 55.4 / WavLM 81.1 / fusion 82.2; CREMA-D 51.4 / 73.8 / 74.1.

| Direction | Norm | eGeMAPS | WavLM | Fusion | Δ relative drop (WavLM − eGeMAPS) | H1 |
|---|---|---|---|---|---|---|
| RAVDESS → CREMA-D | source | 30.1 [28.9, 31.2] | 38.7 [37.0, 40.6] | 42.1 [40.2, 44.0] | **+6.1 pts** [+3.1, +9.4] | ✔ |
| RAVDESS → CREMA-D | corpus | 35.6 [34.3, 36.9] | 45.0 [43.3, 46.7] | 48.6 [46.8, 50.6] | **+8.3 pts** [+5.1, +11.6] | ✔ |
| CREMA-D → RAVDESS | source | 30.0 [27.0, 33.2] | 58.8 [54.7, 62.9] | 66.3 [62.8, 70.1] | **−18.4 pts** [−27.9, −8.1] | ✘ |
| CREMA-D → RAVDESS | corpus | 43.2 [41.2, 45.3] | 67.8 [65.1, 70.5] | 68.3 [65.6, 70.9] | −5.7 pts [−12.4, +1.4] | ✘ |

Chance = 16.7%. Full table for all four models: `results/phase8/e2_all.csv`; figure `results/figures/p8_e2_cross_corpus.png`.

**Findings**
1. **Every model loses a lot across corpora** (typically 20–35 UAR points). In-domain numbers alone overstate real-world performance.
2. **H1 depends on the direction.** Trained on RAVDESS (24 actors, one studio), WavLM loses *relatively more* than eGeMAPS, as H1 predicted. Trained on CREMA-D (91 actors, more varied), WavLM loses *much less* and keeps 59–68% UAR on RAVDESS. A plausible reading: a 768-dimensional representation over-fits the narrow style of a small corpus, but learns transferable cues when the training corpus is diverse. Training-data diversity matters more than the representation type.
3. **WavLM still gives the higher absolute UAR in every direction** (38.7–67.8 vs 30.0–43.2), even where its *relative* drop is larger.
4. **Per-corpus normalisation helps every LogReg configuration** (+2 to +13 points), most for eGeMAPS CREMA-D → RAVDESS (+13.2). Part of the cross-corpus gap is a simple shift in feature levels (consistent with Phase 5 PCA and the Phase 7 formant-amplitude finding).
5. **RBF-SVM fails under the strict setting**: WavLM RAVDESS → CREMA-D drops to 20.6% (almost chance), but recovers to 43.2% with per-corpus normalisation. Distance-based kernels are very sensitive to shifted feature levels.

## 3. E3 — Explanation stability (H2)

We compare global SHAP importance vectors (Spearman correlation over features) between two models trained on:
- **resample:** two bootstrap resamples of the same training data,
- **speaker:** two disjoint sets of speakers from the same corpus,
- **corpus:** one corpus each,
- **random:** two random rankings (no-agreement baseline).

| Representation | Resample | Speaker | Corpus | Random | Order holds (Mann–Whitney, all p < 10⁻¹⁰) |
|---|---|---|---|---|---|
| LogReg · eGeMAPS | 0.78 | 0.63 | 0.50 | 0.00 | ✔ |
| LightGBM · eGeMAPS | 0.76 | 0.44 | 0.20 | 0.00 | ✔ |
| LogReg · WavLM layer 6 | 0.68 | 0.34 | 0.10 | 0.00 | ✔ |

Figure: `results/figures/p8_e3_stability.png`; pairs: `results/phase8/e3_pairs.csv`.

**Findings**
1. **H2 is supported for all three models:** explanations are most stable under resampling, less across speakers and least across corpora, but always far above random.
2. **Explanations of the interpretable model (LogReg · eGeMAPS) are the most stable.** Across corpora it keeps a Spearman of 0.50; LightGBM keeps 0.20 and WavLM dimensions only 0.10. **The most accurate model has the least transferable explanation.** This is a concrete accuracy-vs-explainability trade-off.
3. **Group-level explanations are stable even when feature-level ones are not.** Per-feature share of |SHAP| (mean ± SD over models):

| Group (per feature) | RAVDESS | CREMA-D |
|---|---|---|
| Energy | 0.269 ± 0.019 | 0.293 ± 0.025 |
| Frequency | 0.253 ± 0.011 | 0.210 ± 0.018 |
| Spectral | 0.217 ± 0.012 | 0.220 ± 0.013 |
| Temporal | 0.261 ± 0.019 | 0.277 ± 0.032 |

Energy features are, per feature, the most informative group in both corpora, confirming Phase 7 and P5. Users of SHAP should report **feature groups** rather than single top features, especially across datasets.

## 4. E4 — Stable-core feature selection (H3)

Idea: choose the k eGeMAPS features that are most **consistently** in the top-k across 20 models trained on random halves of the source speakers ("stable core"), and compare with the usual choice, the top-k of one model trained on all source data. Both subsets are then used to train LogReg on the source and tested on the target.

Difference in target UAR, stable core − usual top-k (points, 95% CI):

| Direction | Norm | k = 10 | k = 20 | k = 30 |
|---|---|---|---|---|
| RAVDESS → CREMA-D | source | −0.4 [−1.7, +0.9] | **−1.2** [−1.9, −0.5] | **−1.3** [−2.0, −0.6] |
| RAVDESS → CREMA-D | corpus | +1.0 [−0.2, +2.1] | **−1.3** [−2.2, −0.5] | −0.3 [−1.1, +0.6] |
| CREMA-D → RAVDESS | source | **−3.8** [−7.0, −0.7] | +1.4 [−1.1, +3.8] | −0.9 [−3.9, +2.0] |
| CREMA-D → RAVDESS | corpus | +1.5 [−1.8, +4.9] | +2.5 [−0.3, +5.5] | **+5.0** [+2.3, +7.8] |

Bold = CI excludes 0. Full results: `results/phase8/e4_results.csv`; figure `results/figures/p8_e4_stable_core.png`.

**Findings**
1. **H3 is not supported in general:** 1 significant gain, 4 significant losses, 7 differences not distinguishable from zero.
2. The two subsets overlap heavily (70–87% shared features), so they rarely differ much.
3. The only clear gain (+5.0) appears with the diverse source (CREMA-D) and per-corpus normalisation, the same conditions where transfer was best in E2. A stable-core selection may only help once the feature-level shift has been removed; this would need more corpora to confirm.
4. **Honest conclusion:** explanation stability is useful for *diagnosing* models (E3), but in our data it is not a reliable recipe for *selecting* features that transfer.

## 5. E5 — Error analysis, fairness and label quality

### 5.1 Which emotions break across corpora (best cross-corpus model: fusion LogReg, per-corpus norm; recall)

| Emotion | CREMA-D → RAVDESS: in-domain → cross | RAVDESS → CREMA-D: in-domain → cross |
|---|---|---|
| neutral | 0.87 → 0.92 | 0.86 → **0.09** |
| happy | 0.76 → **0.44** | 0.76 → 0.55 |
| sad | 0.75 → **0.36** | 0.67 → 0.74 |
| angry | 0.90 → 0.82 | 0.83 → 0.64 |
| fearful | 0.79 → 0.69 | 0.64 → 0.51 |
| disgust | 0.88 → 0.88 | 0.68 → 0.39 |

- **Neutral collapses from RAVDESS to CREMA-D (recall 0.09).** RAVDESS "neutral" is a calm, controlled delivery; CREMA-D's neutral sentences are louder and less controlled, so the RAVDESS model sends them to other classes. The label "neutral" does not mean the same thing in both corpora.
- **Sad and happy break from CREMA-D to RAVDESS.** Consistent with Phase 7: the CREMA-D model relies on loudness (quiet = sad), and RAVDESS levels differ.
- Angry and disgust are the most robust emotions. Confusion matrices: `results/figures/p8_e5_cross_confusion.png`.

### 5.2 Gender and age (LogReg UAR; cross = per-corpus norm)

| Target | Features | Female | Male | 20–29 | 30–44 | 45+ |
|---|---|---|---|---|---|---|
| CREMA-D in-domain | eGeMAPS | 0.528 | 0.502 | 0.530 | 0.506 | 0.503 |
| CREMA-D in-domain | WavLM | 0.759 | 0.719 | 0.749 | 0.736 | 0.726 |
| CREMA-D cross | WavLM | 0.448 | 0.452 | 0.455 | 0.425 | 0.475 |
| RAVDESS in-domain | WavLM | 0.844 | 0.778 | – | – | – |

- WavLM is **4–7 points better for female speakers in-domain** in both corpora; the gap disappears cross-corpus (where everything is worse).
- Age differences are small (≤ 3 points in-domain).
- Groups are balanced in our folds (Phase 6), so these gaps are not caused by unequal training data. With 24 / 91 actors, we report them as observations, not as proven bias.

### 5.3 Label quality (CREMA-D listener votes)

| Features (LogReg) | Clips | n | In-domain UAR | Cross UAR (RAVDESS → CREMA-D, per-corpus norm) |
|---|---|---|---|---|
| eGeMAPS | listeners agreed with the intended emotion | 3,099 | 0.622 | 0.447 |
| eGeMAPS | listeners disagreed | 4,343 | 0.445 | 0.307 |
| WavLM | listeners agreed | 3,099 | **0.843** | 0.570 |
| WavLM | listeners disagreed | 4,343 | 0.672 | 0.385 |

- On clips where **human listeners** recognised the intended emotion, the models are 17–18 points better. Much of the "error" is on clips that are ambiguous to humans too.
- Per emotion, human recognition rate and model recall correlate (Spearman 0.60): **what humans find hard, the model finds hard.**
- Our CREMA-D scores are therefore a **lower bound** limited by label noise, not only by the model.

## 6. E6 — Standard evaluation

All on the same speaker-independent folds and Phase 6 settings (out-of-fold predictions; `notebooks/06_evaluation.ipynb`).

### 6.1 Precision, recall and F1 per emotion (best model: LogReg + fusion)

| Emotion | RAVDESS P / R / F1 | CREMA-D P / R / F1 |
|---|---|---|
| neutral | 0.76 / 0.87 / 0.81 | 0.77 / 0.86 / 0.81 |
| happy | 0.84 / 0.76 / 0.80 | 0.78 / 0.76 / 0.77 |
| sad | 0.68 / 0.73 / 0.71 | 0.65 / 0.67 / 0.66 |
| angry | 0.89 / 0.89 / 0.89 | 0.84 / 0.83 / 0.83 |
| fearful | 0.82 / 0.79 / 0.80 | 0.69 / 0.64 / 0.67 |
| disgust | 0.90 / 0.88 / 0.89 | 0.71 / 0.68 / 0.70 |
| **macro avg** | **0.81 / 0.82 / 0.82** | **0.74 / 0.74 / 0.74** |
| accuracy | 0.815 | 0.738 |

- **Angry** is the easiest emotion in both corpora; **sad** is the hardest on RAVDESS, sad and fearful on CREMA-D.
- Neutral has higher recall than precision: the model uses "neutral" as a fallback for uncertain clips.
- The interpretable eGeMAPS model: macro F1 0.54 (RAVDESS), 0.51 (CREMA-D). Full reports: `results/phase8/e6_report_*.csv`.

### 6.2 ROC-AUC (one-vs-rest, macro)

| Model | RAVDESS | CREMA-D |
|---|---|---|
| LogReg · eGeMAPS | 0.853 | 0.833 |
| LightGBM · eGeMAPS | 0.853 | 0.844 |
| MLP · eGeMAPS | 0.853 | 0.833 |
| LogReg · WavLM | **0.969** | 0.943 |
| MLP · WavLM | 0.966 | 0.935 |
| LogReg · fusion | 0.968 | **0.944** |

AUC = probability that a random clip of an emotion gets a higher score for that emotion than a random clip of another emotion (0.5 = chance). All models rank far above chance; WavLM-based models are close to 0.95–0.97. Lowest per-class AUC: CREMA-D disgust (0.75) and fearful (0.78) for eGeMAPS. SVM is excluded (no probabilities). Figure: `results/figures/p8_e6_roc.png`.

### 6.3 Calibration (can the confidence be trusted?)

| Model | ECE RAVDESS | ECE CREMA-D |
|---|---|---|
| LogReg · fusion | 0.025 | 0.021 |
| LogReg · eGeMAPS | 0.079 | 0.040 |

Expected calibration error = average gap between stated confidence and actual accuracy. The fusion model is **well calibrated in-domain** (in the 70–90% confidence bins its actual accuracy is within about 4 points of its confidence; `results/figures/p8_e6_calibration.png`). The eGeMAPS model is slightly over-confident at high confidence on RAVDESS. **Caveat for the app:** calibration was measured on speakers from the same corpora; on a new microphone or speaker style (E2), confidence will be too high. The app must say so.

### 6.4 Overfitting check (mean train vs validation UAR over the 5 folds)

| Model | RAVDESS train → val (gap) | CREMA-D train → val (gap) |
|---|---|---|
| LogReg · eGeMAPS | 73.1 → 55.4 (17.6) | 55.9 → 51.5 (**4.5**) |
| LogReg · WavLM | 99.7 → 81.0 (18.6) | 85.4 → 73.8 (11.6) |
| LogReg · fusion | 99.7 → 82.0 (17.8) | 86.6 → 74.1 (12.5) |
| SVM · eGeMAPS | 86.1 → 52.4 (33.7) | 65.5 → 52.8 (12.7) |
| LightGBM · eGeMAPS | 100.0 → 53.9 (**46.1**) | 97.7 → 53.8 (**43.9**) |
| MLP · eGeMAPS | 76.0 → 51.4 (24.6) | 66.0 → 51.8 (14.2) |
| MLP · WavLM | 98.6 → 78.6 (20.0) | 95.7 → 72.6 (23.1) |

- **Every model fits its training speakers better than new speakers.** Gaps are larger on RAVDESS (only ~19 training actors per fold) than on CREMA-D (~73).
- **LightGBM memorises** the training data (≈100% train UAR) yet does not beat LogReg on new speakers; this is why LogReg was kept as the main model.
- The regularisation chosen by inner CV (small C for LogReg) keeps the linear models' gaps the smallest.
- Because validation speakers are never seen in training, the reported validation UAR is the honest number; the gap shows the remaining speaker-specific overfitting.

### 6.5 Learning curve (CREMA-D, LogReg; validation = held-out speakers)

| Training speakers | 9 | 18 | 36 | 54 | 72 |
|---|---|---|---|---|---|
| eGeMAPS val UAR | 45.1 | 47.6 | 49.4 | 51.0 | 50.9 |
| WavLM val UAR | 67.0 | 70.5 | 72.5 | 73.6 | 74.6 |
| WavLM train − val gap | 26.9 | 21.7 | 15.8 | 12.9 | 11.1 |

- **More speakers help**, and the train–validation gap closes steadily (`results/figures/p8_e6_learning_curve.png`).
- eGeMAPS **plateaus** around 51% after ~54 speakers: its limit is the representation, not the data.
- WavLM is **still improving** at 72 speakers: more speakers would likely help further. This agrees with E2 (the diverse CREMA-D source transfers best) and justifies training the final model on **both** corpora (Phase 9).

## 7. Threats to validity

- **Two corpora only**, both acted English speech, so conclusions about "cross-corpus" are about one pair. H1's direction-dependence in particular needs a third corpus.
- **Acted emotions** are exaggerated compared with real life.
- Bootstrap CIs resample *speakers of the target corpus*; they do not capture variation from choosing different source corpora.
- E4 tests one stability-based selection rule; other rules might behave differently.
- E3 importance for WavLM uses unnamed dimensions; its stability says nothing about human-readable meaning.

## 8. Answer to the research question

> *How well do emotion models and their explanations transfer from one corpus to another?*

1. **Accuracy transfers poorly**, and how poorly depends more on the *training corpus diversity* than on the representation: WavLM transfers worse than eGeMAPS from the small RAVDESS but much better from the diverse CREMA-D (H1 direction-dependent).
2. **Explanations transfer partly:** feature rankings agree much more than chance but much less across corpora than within (H2 supported). Interpretable eGeMAPS explanations are the most stable; **group-level** explanations (Energy > others per feature) are stable across corpora.
3. **Stable explanations are not a reliable feature-selection recipe** (H3 inconclusive).
4. **Simple per-corpus normalisation** recovers 3–13 UAR points, and the remaining errors concentrate on emotions whose acted style differs between corpora (neutral, sad, happy) and on clips that human listeners also find ambiguous.
