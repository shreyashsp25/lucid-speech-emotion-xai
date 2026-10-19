# Emotion Recognition Using Explainable Machine Learning

### Do the accuracy and the explanations of a speech emotion model survive a change of corpus?

**Team Lucid**

| Member | Roll number |
|---|---|
| Pranav Sajith | BL.SC.U4CSE24039 |
| Piriya Shreyash Shanmukh | BL.SC.U4CSE24037 |
| Praveen Raj | BL.SC.U4CSE24040 |

Repository: `lucid-speech-emotion-xai` (GitHub) · Final report · October 2026

---

## Abstract

Speech emotion recognition (SER) systems are usually evaluated, and explained, inside a single dataset. We ask whether a model's **accuracy** and its **SHAP explanations** remain valid when it is moved to a different corpus, and whether explanation stability can guide feature selection. Using RAVDESS (24 actors) and CREMA-D (91 actors) harmonised to six emotions, we compare interpretable eGeMAPS features (88 named acoustic descriptors) with frozen WavLM embeddings and their fusion, across Logistic Regression, RBF-SVM, LightGBM and an MLP (KNN excluded by design). All evaluation is speaker-independent; hyperparameters and WavLM layers are chosen by nested, speaker-grouped cross-validation.

Within each corpus, the fusion model with Logistic Regression is best (UAR 82.2% on RAVDESS, 74.1% on CREMA-D; chance 16.7%), well calibrated (ECE ≈ 0.02), and WavLM beats eGeMAPS by 22–26 points in every fold. Random splits inflated RAVDESS UAR by up to 9.3 points. Across corpora, every model loses 20–35 points. Whether embeddings lose *relatively* more than interpretable features depends on the direction: they do when trained on the small RAVDESS (+6.1 points larger relative drop, 95% CI [3.1, 9.4]) but not when trained on the more diverse CREMA-D (−18.4 [−27.9, −8.1]). SHAP rankings are progressively less stable across resamples (Spearman 0.78), speaker groups (0.63) and corpora (0.50) for the interpretable model, and much less stable for WavLM (0.68 / 0.34 / 0.10), always far above a random baseline. Feature-group shares (Energy most informative per feature) are stable across corpora. Selecting "consistently important" features did not reliably improve transfer. Explainability also exposed a preprocessing flaw (exploding ratio features), fixed by clipping. We release an interactive demo that runs the real pipeline, explains each prediction, tests robustness, and states its own limits.

**Keywords:** speech emotion recognition, explainable AI, SHAP, eGeMAPS, WavLM, cross-corpus generalisation, explanation stability, speaker independence.

---

## Contents
1. Introduction · 2. Problem statement · 3. Motivation · 4. Objectives · 5. Literature review · 6. Research gap and questions · 7. Contributions · 8. Datasets · 9. Preprocessing · 10. Features · 11. Models · 12. Explainability methods · 13. Experimental design · 14. Results · 15. Comparison with published work · 16. Ablations · 17. Error analysis · 18. Interactive system · 19. Discussion · 20. Limitations · 21. Ethics · 22. Conclusion · 23. Future work · References · Appendices

---

## 1. Introduction

Speech carries emotion through *how* something is said: loudness, pitch movement, voice quality and timing. Speech emotion recognition aims to infer the expressed emotion from these cues. Applications include call-centre analytics, human–robot interaction and assistive technology, but every one of them meets audio recorded under conditions the model never saw.

Two developments shape current SER. First, **self-supervised speech models** such as wav2vec 2.0 and WavLM produce embeddings that, with a simple classifier on top, outperform hand-crafted features within a dataset [4, 5, 7, 8]. Second, **explainable AI (XAI)** methods such as SHAP [14] are increasingly used to show which acoustic properties a model relies on [5, 9]. Both trends are usually evaluated inside one corpus. This project studies what happens to both, accuracy and explanation, when the corpus changes.

## 2. Problem statement

Published SER results and explanations are typically produced on one dataset, often with random train/test splits and a single run. Three problems follow:

1. **Speaker leakage.** When the same actor appears in training and test data, a model can recognise voices rather than emotions and report inflated accuracy [6].
2. **Unknown transfer.** Embeddings win within a dataset, but it is not established whether they also transfer better to a new corpus than interpretable features; the authors of eGeMAPS suggested compact features might generalise better but did not test it [3].
3. **Unverified explanations.** SHAP rankings in SER papers come from one run on one dataset [5, 9], yet SHAP rankings are known to change between runs [10]. If an explanation changes with the training sample or the corpus, conclusions such as "energy features matter most" may not be reliable.

## 3. Motivation

- A model that is accurate only on its own corpus is of little practical use; real deployments face new microphones, rooms and speakers.
- An explanation that changes with the dataset gives a false sense of understanding. Explanations should be **tested**, not only displayed.
- If stable explanations were also *predictive* of what transfers, they could be used to design more robust models, linking XAI to generalisation.
- The project title asks for explainable ML; we make explainability the **object of study**, not a decoration.

## 4. Objectives

1. Build a leak-free, speaker-independent SER pipeline for six emotions on RAVDESS and CREMA-D.
2. Compare interpretable (eGeMAPS) and pretrained (WavLM) representations, and their fusion, across four model families (no KNN).
3. Quantify how much random splits inflate accuracy (rigour baseline).
4. Explain the models with SHAP, cross-checked by permutation importance, and use the explanations to audit the pipeline.
5. Measure cross-corpus accuracy loss (H1), explanation stability across resamples, speakers and corpora (H2), and whether stability-guided feature selection transfers better (H3).
6. Analyse errors by emotion, speaker gender and age, and label quality.
7. Deliver an interactive application that uses the real model and communicates uncertainty honestly.

## 5. Literature review

Ten papers were read in full by the team (detailed notes in `literature/notes/`, matrix in `literature/literature_matrix.md`).

| # | Paper | Key content for us |
|---|---|---|
| 1 | Livingstone & Russo 2018 — RAVDESS | 24 actors, 2 sentences; human audio-only accuracy 63% |
| 2 | Cao et al. 2014 — CREMA-D | 91 actors, 12 sentences; human audio-only accuracy ~41%; Krippendorff α 0.42 |
| 3 | Eyben et al. 2016 — GeMAPS/eGeMAPS | Standard 88-feature interpretable set; speaker-independent LOSO; cross-database generalisation left untested |
| 4 | Pepino et al. 2021 — wav2vec 2.0 for SER | RAVDESS 7-class UAR 84.3% (wav2vec2) vs 57.0% (eGeMAPS); middle layers most useful; speaker normalisation used test-speaker statistics |
| 5 | Dixit et al. 2024 — explaining embeddings with acoustic features | SHAP on eGeMAPS and WavLM; group ranking Energy > Frequency > Spectral > Temporal; within-corpus, single run |
| 6 | Antoniou et al. 2023 — SER reality check | Speaker-dependent vs independent protocols change conclusions; reproducibility problems |
| 7 | Wu et al. 2024 — EMO-SUPERB | Frozen self-supervised encoders + light head as a standard; CREMA-D benchmark |
| 8 | Pastor et al. 2023 — cross-corpus training with SSL | Merging corpora gives small, inconsistent gains; adding CREMA-D lowered IEMOCAP UAR |
| 9 | Norval & Wang 2025 — XAI for SER | SHAP and LIME on spectral features; unreadable feature indices; unclear split |
| 10 | Hwang et al. 2026 — explanation multiplicity in SHAP | SHAP rankings vary across runs; rank-based metrics needed |

**Themes.**
- *Representations.* Embeddings beat eGeMAPS within a dataset [4, 5]; fusion adds little [4].
- *Models.* A frozen encoder plus a simple classifier is the accepted standard [7]; logistic regression [5] and SVM [3] remain common.
- *Explainability.* SHAP is the most used method [5, 9], but stability of its rankings is not measured in the SER papers we read [10].
- *Evaluation.* Speaker leakage and inconsistent protocols are recurring problems [6]; credible work uses speaker-independent splits [3, 4, 5, 7, 8].
- *Generalisation.* Cross-corpus studies report accuracy only [8, 19, 20], not whether explanations change.

## 6. Research gap and questions

### 6.1 Counter-evidence search
Before claiming a gap, we searched for work that might already fill it (Phase 2, `report/02_research_gap_and_methodology.md`):

| Work | Overlap | Remaining gap |
|---|---|---|
| Kaloga & Kodrasi 2025 [15] | Counts features that recur across 6 corpora | No formal stability metric, no cross-corpus train/test, no RAVDESS/CREMA-D |
| Kołodziej et al. 2026 [16] | Feature recurrence across LOSO folds | One Polish corpus; no cross-corpus test |
| Nasr et al. 2025 [17] | Plausibility of saliency explanations, CREMA-D | Plausibility, not stability |
| Phukan et al. 2025 [19], Ibrahim et al. 2024 [20] | Cross-corpus benchmarks incl. RAVDESS/CREMA-D | Accuracy only; no explainability |
| Nfissi et al. 2024 [18] | SHAP-guided feature selection | One corpus, 2 speakers; transfer and stability not tested |
| Springer 2026 article on SHAP/permutation-guided feature selection for SER | — | **Full text not accessible to us; we could not check it** |

We found no study that formally measures SHAP stability across resamples, speaker groups **and** corpora for RAVDESS ↔ CREMA-D, or that tests whether explanation stability predicts transfer. We describe this as an **underexplored question**, not a first: our search is not exhaustive and one relevant article could not be read.

### 6.2 Research question
> When an SER model trained on one acted English corpus is applied to another, do its accuracy and its SHAP explanations hold up, and can explanation stability be used to choose features that transfer better?

| | Hypothesis |
|---|---|
| **H0** (rigour) | Random splits inflate accuracy, more on RAVDESS (24 speakers) than on CREMA-D (91). |
| **H1** | WavLM wins within a corpus, but its *relative* cross-corpus drop is larger than eGeMAPS's. |
| **H2** | SHAP ranking stability decreases: resample > speaker group > corpus > random. |
| **H3** | A "stable core" of consistently important features transfers better than the usual top-k features. |

## 7. Contributions

Stated carefully, without novelty claims beyond what we verified:
1. A leak-free, speaker-independent comparison of eGeMAPS, WavLM and fusion, with four model families, **within and across** RAVDESS and CREMA-D, with speaker-level bootstrap confidence intervals.
2. A quantitative measurement of **SHAP explanation stability** across resamples, disjoint speaker groups and corpora, against a random-ranking baseline, for interpretable and embedding models.
3. A test of **stability-guided feature selection** for cross-corpus transfer (negative/inconclusive result, reported as such).
4. A documented case of **XAI used to debug** a pipeline (exploding ratio features).
5. An **interactive research demo** that runs the real pipeline, explains predictions, simulates perturbations, and reports calibration and typicality.

## 8. Datasets

| | RAVDESS (speech, audio-only) | CREMA-D |
|---|---|---|
| Files | 1,440 | 7,442 |
| Speakers | 24 (12 F, 12 M), ages 21–33 | 91 (43 F, 48 M), ages 20–74, diverse ethnicity |
| Sentences | 2 | 12 |
| Emotions | 8 (incl. calm, surprised) | 6 |
| Sample rate | 48 kHz (5 files stereo) | 16 kHz |
| Mean duration | 3.70 s | 2.54 s |
| Licence | CC BY-NC-SA 4.0 | ODbL 1.0 |

**Why these two.** Both are free, widely used [1, 2, 4, 5, 7, 8], share six emotions, and differ in size, speaker diversity, sentences and recording conditions, which is exactly the shift we want to study. Both use acted, *intended* labels.

**Harmonisation.** Six classes: neutral, happy, sad, angry, fearful, disgust. RAVDESS calm and surprised (384 clips) are dropped. Used: RAVDESS **1,056** clips (neutral 96, other classes 192 each), CREMA-D **7,442** (neutral 1,087, others 1,271).

**Label quality.** For CREMA-D, the audio-only listener majority matches the intended label for only **41.6%** of clips (sad 16.4%, happy 26.0%, neutral 95.7%), in line with [2]. These 3,099 "agreed" clips are used in the error analysis.

**Splits.** Five speaker-independent folds per corpus (`StratifiedGroupKFold`, seed 42), grouped by speaker and stratified by emotion **and gender**; no speaker appears in two folds (checked in code). Cross-corpus experiments train on all clips of one corpus and test on all clips of the other.

## 9. Preprocessing

Identical for both corpora (`src/audio.py`):

| Step | Setting | Reason |
|---|---|---|
| Mono | average channels | 5 RAVDESS files are stereo |
| Resample | 16 kHz, exact polyphase | removes a format confound; required by WavLM |
| Trim edge silence | `top_db = 30`, never below 0.25 s | RAVDESS had ~1.8 s extra silence (mean 3.70 → 1.90 s; CREMA-D 2.54 → 2.39 s) |
| Amplitude normalisation | none | loudness is an emotion cue; corpus gain is handled by standardisation |
| Standardisation | z-score, then **clip to ±5 SD**, inside each pipeline, fitted on training data only | see §14.5 (found through SHAP) |

**Data-quality finding.** 39 CREMA-D clips (mostly *sad* and *disgust*) have no detected pitch; they were kept and tested for shortcut effects (§14.5).

## 10. Features

| Set | Dimension | Tool | Role |
|---|---|---|---|
| **eGeMAPS v02** [3] | 88 named functionals: pitch, loudness, jitter, shimmer, HNR, formants, spectral slope/balance, MFCC 1–4, temporal rates | openSMILE [21] | interpretable features; human-readable SHAP |
| **WavLM base-plus** [13] | 13 layers × 768, frozen, mean-pooled over time | Hugging Face `microsoft/wavlm-base-plus` | strong pretrained representation |
| **Fusion** | 88 + 768 = 856 | concatenation | complementarity test |

eGeMAPS features fall into four groups following [3, 5]: **Energy** (15), **Frequency** (24), **Spectral** (43), **Temporal** (6).

**Domain shift (PCA).** With features standardised on pooled data, the corpora separate visibly along the second principal component for eGeMAPS and form distinct clusters for WavLM layer 6 (`results/figures/p5_domain_shift_pca.png`): both representations encode *which corpus*, not only *which emotion*.

## 11. Models

| Model | Grid (inner CV) | Why |
|---|---|---|
| Logistic Regression (L2, balanced) | C ∈ {0.01, 0.1, 1} | linear baseline; exact SHAP |
| SVM, RBF kernel (balanced) | C ∈ {1, 10} | standard in paralinguistics [3] |
| LightGBM (250 trees, balanced) | min_child_samples ∈ {10, 30} | non-linear interactions; exact TreeSHAP |
| MLP (256 units, early stopping) | α ∈ {0.001, 0.1} | standard neural head on embeddings [4, 7] |

**KNN was excluded by project rule.** It would also be a poor fit here: distance-based methods are sensitive to the corpus offsets shown in §10 and to the 768-dimensional embedding space, and they offer no global explanation.

**Tuning.** For each outer fold, hyperparameters and the WavLM layer are chosen by 3-fold speaker-grouped CV on the training folds only (nested CV). The outer test fold is used once.

## 12. Explainability methods

| Method | Use |
|---|---|
| **Exact linear SHAP** for LogReg: φ = w · (z − z̄_train) on standardised, clipped features | global and local explanations; verified identical to `shap.LinearExplainer` (difference 0.0, additivity error ≈ 10⁻¹⁴) |
| **TreeSHAP** for LightGBM [14] | non-linear comparison |
| **Permutation importance** [22] (10 repeats, drop in test UAR) | independent cross-check |
| **Out-of-fold explanation** | every clip explained by a model that never heard its speaker |
| **Stability metrics**: Spearman, Kendall τ, top-k Jaccard, against a random-ranking null | H2 (following [10]) |
| **Group shares** (Energy, Frequency, Spectral, Temporal), total and per feature | comparison with [5] |

SHAP describes what a **model** uses. It is association, not evidence that a voice property causes an emotion.

## 13. Experimental design

| ID | Experiment | Answers |
|---|---|---|
| E0 | Random split vs speaker split, same LogReg | H0 |
| E1 | Within-corpus, 5 speaker-independent folds × 3 feature sets × 4 models | baseline for H1 |
| E2 | Cross-corpus, both directions, 3 feature sets × 4 models × 2 normalisations | H1 |
| E3 | SHAP stability: resample / disjoint speakers / corpus / random | H2 |
| E4 | Stable-core vs top-k selection (k = 10, 20, 30) → cross-corpus | H3 |
| E5 | Per-class, gender, age and label-quality error analysis | discussion |
| E6 | Precision/recall/F1, ROC-AUC, calibration, overfitting, learning curve | standard evaluation |

**Normalisation variants in E2/E4.** *Source*: the scaler is fitted on the source corpus only (strict). *Corpus*: each corpus is standardised with its own mean and SD, an established unsupervised adaptation [23] that uses unlabelled target features, never target labels.

**Metrics.** UAR (mean per-class recall; primary, robust to RAVDESS's half-size neutral class), macro-F1, accuracy, per-class recall, ROC-AUC (one-vs-rest), ECE [24].

**Statistics.** Cross-corpus 95% CIs from a **speaker-level bootstrap** (1,000 resamples of target speakers; clips of one speaker are not independent). Paired differences use the same resamples. H2 uses Mann–Whitney tests. With 5 folds, within-corpus fold-level tests cannot reach p < 0.05 (minimum Wilcoxon p = 0.0625), so we report consistency across folds instead.

**Leakage rules.** Every selection (layer, hyperparameters, feature subset) uses training/source data only; scaling and clipping are inside pipelines.

## 14. Results

### 14.1 E0 — Speaker leakage (H0 supported)

| Corpus | Features | Random split | Speaker split | Inflation |
|---|---|---|---|---|
| RAVDESS | eGeMAPS | 63.2% | 53.9% | **+9.3** |
| RAVDESS | WavLM L6 | 91.8% | 84.0% | **+7.8** |
| CREMA-D | eGeMAPS | 52.3% | 50.8% | +1.5 |
| CREMA-D | WavLM L6 | 74.2% | 72.5% | +1.7 |

The smaller the speaker pool, the more a random split rewards recognising voices. A RAVDESS score above 90% can be a leakage artefact.

### 14.2 E1 — Within-corpus results

Mean UAR ± SD over 5 speaker-independent folds (`results/figures/p6_within_corpus_uar.png`):

| Corpus | Model | eGeMAPS | WavLM | Fusion |
|---|---|---|---|---|
| RAVDESS | LogReg | **55.4 ± 2.6** | 81.0 ± 5.9 | **82.2 ± 5.1** |
| | SVM | 52.4 ± 5.0 | 78.5 ± 7.5 | 78.8 ± 9.4 |
| | LightGBM | 53.9 ± 4.2 | 76.8 ± 7.9 | 76.3 ± 7.3 |
| | MLP | 51.4 ± 6.1 | 78.6 ± 8.1 | 81.3 ± 4.3 |
| CREMA-D | LogReg | 51.4 ± 3.3 | 73.8 ± 1.9 | **74.1 ± 2.2** |
| | SVM | 52.8 ± 3.8 | 73.4 ± 3.0 | 73.4 ± 3.4 |
| | LightGBM | **53.8 ± 3.7** | 70.3 ± 3.2 | 70.2 ± 4.2 |
| | MLP | 52.0 ± 3.5 | 72.5 ± 1.8 | 73.1 ± 3.6 |

- WavLM beats eGeMAPS by **≈26 points (RAVDESS) and ≈22 points (CREMA-D)**, in all 5 folds of both corpora.
- Fusion adds little (+0.3 to +1.2 for LogReg).
- Logistic Regression is best or tied-best on embeddings; non-linear models do not help on 768-d WavLM features.
- Fold-to-fold variation is larger on RAVDESS (4–5 test speakers per fold).

**WavLM layer.** Inner-CV UAR peaks in the middle layers (RAVDESS layer 5: 79.3%; CREMA-D layers 5–7: ≈71%) and declines toward layer 12 (`p6_wavlm_layer_profile.png`), consistent with [4].

### 14.3 E6 — Standard evaluation of the within-corpus models

**Per-class precision / recall / F1 (LogReg + fusion, out-of-fold):**

| Emotion | RAVDESS P / R / F1 | CREMA-D P / R / F1 |
|---|---|---|
| neutral | 0.76 / 0.87 / 0.81 | 0.77 / 0.86 / 0.81 |
| happy | 0.84 / 0.76 / 0.80 | 0.78 / 0.76 / 0.77 |
| sad | 0.68 / 0.73 / 0.71 | 0.65 / 0.67 / 0.66 |
| angry | 0.89 / 0.89 / 0.89 | 0.84 / 0.83 / 0.83 |
| fearful | 0.82 / 0.79 / 0.80 | 0.69 / 0.64 / 0.67 |
| disgust | 0.90 / 0.88 / 0.89 | 0.71 / 0.68 / 0.70 |
| **macro** | **0.81 / 0.82 / 0.82** | **0.74 / 0.74 / 0.74** |

| Check | RAVDESS | CREMA-D |
|---|---|---|
| Macro ROC-AUC, LogReg fusion / WavLM / eGeMAPS | 0.968 / 0.969 / 0.853 | 0.944 / 0.943 / 0.833 |
| ECE, LogReg fusion / eGeMAPS | **0.025** / 0.079 | **0.021** / 0.040 |
| Train → validation UAR, LogReg fusion | 99.7 → 82.0 | 86.6 → 74.1 |
| Train → validation UAR, LightGBM eGeMAPS | 100.0 → 53.9 | 97.7 → 53.8 |

- The fusion model is **well calibrated in-domain**.
- Every model fits its training speakers better than new ones; LightGBM memorises (≈100% train) without gaining on new speakers.
- **Learning curve** (CREMA-D, LogReg, held-out speakers): WavLM rises from 67.0% (9 training speakers) to 74.6% (72) and is still improving; eGeMAPS plateaus near 51% after ~54 speakers (`p8_e6_learning_curve.png`).

### 14.4 Explainability findings (Phase 7)

**Global importance** (LogReg, mean |SHAP|, out-of-fold; `p7_global_top15.png`):

| Rank | RAVDESS | CREMA-D |
|---|---|---|
| 1 | MFCC 1 mean | equivalent sound level |
| 2 | F1 frequency mean | loudness mean |
| 3 | MFCC 3 mean | F2 amplitude rel. F0, mean |
| 4 | Hammarberg index (voiced) variation | F2 amplitude rel. F0, variation |
| 5 | alpha ratio (unvoiced) | loudness rising slope |

The CREMA-D model relies on **loudness and energy**; the RAVDESS model on **spectral shape and formants**.

**Feature groups vs [5].** Per feature, **Energy is the most informative group in both corpora** (share per feature 0.29 RAVDESS, 0.36 CREMA-D). The full order Energy > Frequency > Spectral > Temporal is reproduced on RAVDESS but only partly on CREMA-D (Spectral > Frequency ≈ Temporal after Energy).

**Direction plausibility.** On CREMA-D, *angry* is pushed by a steeper loudness rise, higher alpha ratio and higher sound level; *sad* most strongly by **lower sound level**. On RAVDESS, *happy* is pushed by higher median pitch and pitch variation. These match textbook vocal-emotion acoustics [3].

**Cross-check.** SHAP vs permutation importance: Spearman **0.79** (RAVDESS) and **0.96** (CREMA-D); top-10 overlap 5/10 and 7/10. Lower agreement on RAVDESS is expected with fewer clips and correlated features.

**XAI as a debugging tool.** The first SHAP run showed single features contributing up to ±170 log-odds. Cause: eGeMAPS "stddevNorm" features (std ÷ mean) explode when the mean is near zero, reaching 86 SD in CREMA-D (138 clips beyond 10 SD). Clipping standardised values to ±5 SD was added to every pipeline and all experiments were re-run: eGeMAPS UAR rose slightly (RAVDESS 54.6 → 55.4%; CREMA-D 51.1 → 51.4%) and the largest contribution fell to ≈10. **Accuracy alone had hidden the problem.**

**Shortcut check.** The 39 no-pitch CREMA-D clips are pushed slightly toward *sad*, but retraining without them leaves UAR unchanged (51.41 vs 51.40%): no harmful shortcut.

### 14.5 E2 — Cross-corpus transfer (H1: direction-dependent)

LogReg, UAR (%) on the other corpus, 95% speaker-bootstrap CI (`p8_e2_cross_corpus.png`). In-domain UAR of the target for reference (pooled out-of-fold): CREMA-D 51.4 / 73.8 / 74.1; RAVDESS 55.4 / 81.1 / 82.2 (eGeMAPS / WavLM / fusion).

| Direction | Norm | eGeMAPS | WavLM | Fusion | Δ relative drop (WavLM − eGeMAPS) |
|---|---|---|---|---|---|
| RAVDESS → CREMA-D | source | 30.1 [28.9, 31.2] | 38.7 [37.0, 40.6] | 42.1 [40.2, 44.0] | **+6.1** [+3.1, +9.4] ✔ |
| RAVDESS → CREMA-D | corpus | 35.6 [34.3, 36.9] | 45.0 [43.3, 46.7] | 48.6 [46.8, 50.6] | **+8.3** [+5.1, +11.6] ✔ |
| CREMA-D → RAVDESS | source | 30.0 [27.0, 33.2] | 58.8 [54.7, 62.9] | 66.3 [62.8, 70.1] | **−18.4** [−27.9, −8.1] ✘ |
| CREMA-D → RAVDESS | corpus | 43.2 [41.2, 45.3] | 67.8 [65.1, 70.5] | 68.3 [65.6, 70.9] | −5.7 [−12.4, +1.4] ✘ |

1. **Every model loses 20–35 points** across corpora.
2. **H1 holds only from RAVDESS.** Trained on 24 actors in one studio, WavLM loses relatively more than eGeMAPS. Trained on CREMA-D's 91 varied actors, WavLM loses much less and keeps 59–68% on RAVDESS. Training-set diversity matters more than the representation type.
3. **WavLM still has the higher absolute UAR in every setting.**
4. **Per-corpus normalisation helps every LogReg configuration** (+2 to +13 points; largest for eGeMAPS CREMA-D → RAVDESS, +13.2).
5. **RBF-SVM collapses under the strict setting** (WavLM RAVDESS → CREMA-D 20.6%, near chance) and recovers to 43.2% with per-corpus normalisation: distance-based kernels are fragile under shifted feature levels.

### 14.6 E3 — Explanation stability (H2 supported)

Median Spearman correlation between global SHAP importance vectors of two models trained on:

| Representation | Two resamples | Two disjoint speaker groups | Two corpora | Random rankings |
|---|---|---|---|---|
| LogReg · eGeMAPS | 0.78 | 0.63 | 0.50 | 0.00 |
| LightGBM · eGeMAPS | 0.76 | 0.44 | 0.20 | 0.00 |
| LogReg · WavLM layer 6 | 0.68 | 0.34 | 0.10 | 0.00 |

All orderings hold (Mann–Whitney, all p < 10⁻¹⁰; `p8_e3_stability.png`).
- Explanations are **never fully stable even on the same data** (0.68–0.78 between resamples), as [10] warns.
- The **interpretable linear model has the most stable explanations**; the most accurate representation (WavLM) has the least transferable explanation: a concrete accuracy-versus-explainability trade-off.
- **Feature-group shares are stable** across corpora (per-feature share, mean ± SD over models: Energy 0.27 ± 0.02 RAVDESS, 0.29 ± 0.03 CREMA-D; the other groups vary within 0.21–0.28). Groups are a more reliable unit of explanation than single features.

### 14.7 E4 — Stable-core feature selection (H3 inconclusive)

Difference in target UAR, stable core − usual top-k (points, 95% CI):

| Direction | Norm | k = 10 | k = 20 | k = 30 |
|---|---|---|---|---|
| RAVDESS → CREMA-D | source | −0.4 [−1.7, +0.9] | **−1.2** [−1.9, −0.5] | **−1.3** [−2.0, −0.6] |
| RAVDESS → CREMA-D | corpus | +1.0 [−0.2, +2.1] | **−1.3** [−2.2, −0.5] | −0.3 [−1.1, +0.6] |
| CREMA-D → RAVDESS | source | **−3.8** [−7.0, −0.7] | +1.4 [−1.1, +3.8] | −0.9 [−3.9, +2.0] |
| CREMA-D → RAVDESS | corpus | +1.5 [−1.8, +4.9] | +2.5 [−0.3, +5.5] | **+5.0** [+2.3, +7.8] |

One significant gain, four significant losses, seven differences indistinguishable from zero; the two subsets share 70–87% of features. **Stability is useful for diagnosing models, but not a reliable selection recipe** in our data. The only clear gain occurs with the diverse source and per-corpus normalisation, the same conditions under which transfer was best in E2.

### 14.8 Final model

The final system is LogReg on fusion features (WavLM layer chosen by inner CV on all data: **layer 7**, C = 0.01), trained on all 8,498 clips of 115 speakers, plus a LogReg on eGeMAPS (C = 1) used for explanation. The same recipe, tested with fold-0 speakers held out, scored **72.5% UAR on RAVDESS** (Phase 6 single-corpus model on the same speakers: 75.0%) and **72.3% on CREMA-D** (71.5%). Pooling cost RAVDESS, which is outnumbered 7 : 1, about 2.5 points, in exchange for a model that has heard both recording styles.

## 15. Comparison with published work

Numbers from other papers use different protocols; comparisons are indicative only.

| Claim in the literature | Our finding |
|---|---|
| Embeddings ≫ eGeMAPS within RAVDESS: 84.3 vs 57.0% UAR, 7 classes [4] | Same gap size: 81.0 vs 55.4% UAR, 6 classes, speaker-independent 5-fold, no test-speaker normalisation |
| Middle layers of self-supervised models carry most emotion information [4] | Confirmed for WavLM (layers 5–7) |
| Energy > Frequency > Spectral > Temporal [5] (RAVDESS, SAVEE) | Reproduced on RAVDESS; only partly on CREMA-D; Energy first per feature in both |
| Compact interpretable features may generalise better across databases (suggested, untested) [3] | Only when trained on the small corpus; not when trained on the diverse corpus |
| Random/speaker-dependent splits change conclusions [6] | Measured: +7.8 to +9.3 points on RAVDESS, +1.5 to +1.7 on CREMA-D |
| Adding CREMA-D to training can lower performance elsewhere [8] | Pooling both corpora cost RAVDESS ~2.5 points on held-out speakers |
| SHAP rankings vary between runs [10] | Measured in SER: Spearman 0.68–0.78 between resamples of the same data |
| Human audio-only accuracy 63% (RAVDESS) [1], ~41% (CREMA-D) [2] | Shown as context only; human and model numbers are not directly comparable (different task, intended labels) |

## 16. Ablations

| Ablation | Result |
|---|---|
| Feature set (eGeMAPS / WavLM / fusion) | WavLM +22–26 over eGeMAPS; fusion +0.3–1.2 over WavLM within corpus, up to +7.5 cross-corpus (CREMA-D → RAVDESS, source norm) |
| WavLM layer (0–12) | Middle layers best; layer 0 is 24 points (RAVDESS) / 13 points (CREMA-D) below the peak |
| Model family | LogReg best on embeddings; LightGBM best on eGeMAPS for CREMA-D (53.8 vs 51.4) but overfits most |
| Clipping (±5 SD) | +0.8 (RAVDESS) / +0.3 (CREMA-D) eGeMAPS UAR; removes ±170 log-odds explanations |
| Normalisation (source vs per-corpus) | +2 to +13 points cross-corpus for LogReg; SVM 20.6 → 43.2 |
| Split type (random vs speaker) | −7.8 to −9.3 points on RAVDESS when leakage is removed |
| Training speakers (9 → 72, CREMA-D) | WavLM +7.6 points, still rising; eGeMAPS +5.8, plateaus |
| Removing 39 no-pitch clips | no change (51.41 vs 51.40%) |
| Feature subset (stable core vs top-k) | mixed: −3.8 to +5.0 points |
| Training data (single corpus vs pooled) | RAVDESS −2.5, CREMA-D +0.8 on fold-0 speakers |

## 17. Error analysis

**Within-corpus confusions** (LogReg + fusion, row-normalised, out-of-fold):
- RAVDESS: fearful → sad **13.5%**, neutral → sad **10.4%**, sad → neutral 9.4%.
- CREMA-D: fearful → sad **17.3%**, disgust → sad **11.4%**, sad → fearful/disgust 11.0% / 9.8%.
- *Sad* is the "sink" class: low-energy deliveries are pulled toward it. SHAP shows why: a quiet *fearful* clip is pushed to *sad* mainly by its low sound level (Phase 7, `p7_local_examples.png`).

**Cross-corpus per-class recall** (best transfer setting: fusion, per-corpus normalisation):

| Emotion | CREMA-D → RAVDESS: in-domain → cross | RAVDESS → CREMA-D: in-domain → cross |
|---|---|---|
| neutral | 0.87 → 0.92 | 0.86 → **0.09** |
| happy | 0.76 → **0.44** | 0.76 → 0.55 |
| sad | 0.75 → **0.36** | 0.67 → 0.74 |
| angry | 0.90 → 0.82 | 0.83 → 0.64 |
| fearful | 0.79 → 0.69 | 0.64 → 0.51 |
| disgust | 0.88 → 0.88 | 0.68 → 0.39 |

*Neutral* collapses from RAVDESS to CREMA-D: RAVDESS neutral is a calm, controlled delivery, CREMA-D neutral is not. The same label does not mean the same acoustics in both corpora.

**Gender and age (LogReg UAR).** WavLM is 4–7 points better for female speakers in-domain (CREMA-D 75.9 vs 71.9; RAVDESS 84.4 vs 77.8); the gap disappears cross-corpus. Age differences are ≤ 3 points. Folds are gender-balanced, so these are observations, not proven bias.

**Label quality.** On CREMA-D clips where listeners agreed with the intended emotion, UAR is 17–18 points higher (WavLM 84.3 vs 67.2%). Per emotion, human recognition rate and model recall correlate (Spearman 0.60): what humans find hard, the model finds hard. CREMA-D scores are partly limited by label noise.

## 18. Interactive system

A Streamlit application (`app/streamlit_app.py`, `report/09_simulation.md`) runs the real preprocessing and the saved models.

- **Input:** held-out dataset clips (speakers the demo model never heard), file upload, or microphone.
- **Output:** predicted emotion, all class scores, and a trust panel (calibration in-domain; explicit out-of-domain warning for uploads; a typicality rank against the 8,498 training clips; disagreement between the fusion and eGeMAPS models).
- **Explanation:** exact linear SHAP for any chosen emotion, a plain-language summary, net evidence per feature group, and the share of the fusion model's evidence that comes from named features vs WavLM.
- **Signal views:** waveform, energy, spectrogram, Mel-spectrogram, pitch contour; 16 interpretable features with training percentiles.
- **Robustness simulation:** re-runs the pipeline on quieter/louder, pitch-shifted and noisy copies.
- **Results pages:** all Phase 6–9 results, interactive.

Pipeline parity was verified: features computed live in the app equal the stored training features (maximum difference 5 × 10⁻⁵). An early fixed out-of-range warning fired on 8.4% of genuine RAVDESS training clips; it was replaced by a rank-based typicality check (warn above the 95th percentile).

## 19. Discussion

**Accuracy does not transfer well, and diversity matters more than representation.** Embeddings are much stronger within a corpus and also stronger in absolute terms across corpora. Whether they *degrade more* depends on what they were trained on: a 768-dimensional representation trained on 24 actors in one studio absorbs that studio's style, while one trained on 91 varied actors learns cues that carry over. The learning curve agrees: WavLM was still improving at 72 speakers.

**Explanations transfer partly.** Rankings agree far more than chance but noticeably less across corpora than within. The two corpora "explain" emotion through different features (loudness vs spectral shape), which is consistent with their different recording levels and speaker pools rather than with different emotions. **Group-level explanations are the stable unit**: Energy is the most informative group per feature everywhere. Practical consequence: SER papers should report explanation stability, and favour group-level statements over single top features.

**Interpretability trades off with explanation stability.** The interpretable linear model is less accurate but has the most stable explanations; WavLM is the most accurate with the least stable ones. This is a measurable form of the accuracy-explainability trade-off.

**XAI earned its place.** It exposed a numerical flaw that accuracy hid, ruled out a suspected shortcut (no-pitch clips), explained the dominant confusion (quiet → sad), and identified loudness reliance, which the robustness simulation now lets users test.

**Simple adaptation is effective.** Per-corpus standardisation, which needs no target labels, recovers 2–13 points. Part of the cross-corpus gap is a shift in feature levels, not in emotion.

## 20. Limitations

- **Two corpora only**, both acted English speech; the cross-corpus findings concern one pair, and H1's direction dependence needs a third corpus to generalise.
- **Acted emotions** are exaggerated relative to everyday speech; labels are intended, not felt.
- **One Springer 2026 article** in the gap search could not be accessed; our gap statement is qualified accordingly.
- **One seed** per within-corpus configuration (LogReg and SVM are deterministic; LightGBM and MLP were not repeated across seeds), and small hyperparameter grids for laptop compute. Phase 2 had planned 5 seeds.
- With 5 folds, within-corpus differences are reported as fold consistency, not significance tests.
- **WavLM explanations** concern unnamed dimensions; their stability says nothing about human-readable meaning.
- The **app** is a local research demo; microphone recordings are out-of-domain and its scores for them are rough.

## 21. Ethics

- **Data:** consenting professional actors; licences respected (RAVDESS non-commercial). No audio, features or models are redistributed; the repository contains code, metadata and download instructions only.
- **Validity of claims:** the system recognises *acted, expressed* emotion in short clips. It does not measure how a person feels and must not be used for decisions about people (hiring, policing, insurance, health).
- **Fairness:** small accuracy differences by gender were observed; CREMA-D has more demographic variety than RAVDESS, but neither represents the general population or other languages.
- **Privacy:** the app processes recordings locally and does not store them.
- **Honest communication:** the app states calibration, out-of-domain risk and model disagreement rather than showing a bare confident label.

## 22. Conclusion

With speaker-independent evaluation, WavLM-based models recognise six acted emotions far better than interpretable eGeMAPS features within a corpus (up to 82.2% UAR), and remain better in absolute terms across corpora. But all models lose 20–35 points when the corpus changes, and whether embeddings degrade more than interpretable features depends on the diversity of the training corpus. SHAP explanations are partly stable: rankings change with the training sample and more with the corpus, especially for embeddings, while feature-group importance is stable. Choosing features by explanation stability did not reliably improve transfer. Explainability proved most valuable as an auditing tool: it uncovered a preprocessing flaw, ruled out a shortcut and explained the main errors. The research question is answered as follows: **accuracy does not hold up well across corpora, explanations hold up only partly (best at group level), and explanation stability is a diagnostic, not a feature-selection recipe.**

## 23. Future work

1. Add a third and fourth corpus (e.g., a non-acted or non-English corpus) to test H1's direction dependence and H2 more widely.
2. Repeat within-corpus runs over several seeds and use more folds for formal paired tests.
3. Probe WavLM dimensions with eGeMAPS features (as in [5]) to give the embedding explanations names.
4. Test speaker-level or utterance-level normalisation and domain-adversarial training for transfer.
5. Quantify the robustness simulation over all held-out clips (gain, pitch, noise) as a formal experiment.
6. Use soft (listener-distribution) labels for CREMA-D, as in [7].
7. Calibrate scores under domain shift (e.g., temperature scaling on a small labelled target set).

---

## References

[1] S. R. Livingstone and F. A. Russo, "The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)," *PLOS ONE*, vol. 13, no. 5, e0196391, 2018.

[2] H. Cao, D. G. Cooper, M. K. Keutmann, R. C. Gur, A. Nenkova and R. Verma, "CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset," *IEEE Trans. Affective Computing*, vol. 5, no. 4, pp. 377–390, 2014.

[3] F. Eyben, K. R. Scherer, B. W. Schuller, J. Sundberg, E. André, C. Busso, L. Y. Devillers, J. Epps, P. Laukka, S. S. Narayanan and K. P. Truong, "The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing," *IEEE Trans. Affective Computing*, vol. 7, no. 2, pp. 190–202, 2016.

[4] L. Pepino, P. Riera and L. Ferrer, "Emotion Recognition from Speech Using wav2vec 2.0 Embeddings," *Proc. Interspeech*, 2021. arXiv:2104.03502.

[5] S. Dixit, D. M. Low, G. Elbanna, F. Catania and S. S. Ghosh, "Explaining Deep Learning Embeddings for Speech Emotion Recognition by Predicting Interpretable Acoustic Features," arXiv:2409.09511, 2024.

[6] N. Antoniou, A. Katsamanis, T. Giannakopoulos and S. Narayanan, "Designing and Evaluating Speech Emotion Recognition Systems: A Reality Check Case Study with IEMOCAP," arXiv:2304.00860, 2023 (published at ICASSP 2023 according to our literature matrix; venue to be confirmed from the PDF).

[7] H. Wu, H.-C. Chou, K.-W. Chang, L. Goncalves, J. Du, J.-S. R. Jang, C.-C. Lee and H.-Y. Lee, "EMO-SUPERB: An In-depth Look at Speech Emotion Recognition," arXiv:2402.13018, 2024.

[8] M. A. Pastor, D. Ribas, A. Ortega, A. Miguel and E. Lleida, "Cross-Corpus Training Strategy for Speech Emotion Recognition Using Self-Supervised Representations," *Applied Sciences*, vol. 13, no. 16, 9062, 2023. doi:10.3390/app13169062.

[9] M. Norval and Z. Wang, "Explainable Artificial Intelligence Techniques for Speech Emotion Recognition: A Focus on XAI Models," *Inteligencia Artificial*, vol. 28, no. 76, pp. 85–123, 2025. doi:10.4114/intartif.vol28iss76pp85-123.

[10] H. Hwang, S. Lee, L. Rosenblatt, S. E. Whang and J. Stoyanovich, "Explanation Multiplicity in SHAP: Characterization and Assessment," arXiv:2601.12654, 2026.

[13] S. Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing," *IEEE J. Selected Topics in Signal Processing*, vol. 16, no. 6, pp. 1505–1518, 2022.

[14] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

[15] Y. Kaloga and I. Kodrasi, "Towards Interpretable Emotion Recognition: Identifying Key Features with Machine Learning," arXiv:2508.04230, 2025.

[16] Kołodziej et al., "Emotion Recognition Using Acoustic Features and Deep Learning: A Speaker-Independent Study," *Signals*, vol. 7, no. 4, 2026.

[17] Nasr, Ren and Johnson, "Beyond Saliency: Grounding Explanations of Speech Emotion Recognition in Acoustic Cues," arXiv:2511.11691, 2025.

[18] Nfissi et al., "Iterative Feature Boosting for Explainable Speech Emotion Recognition," arXiv:2405.20172, 2024.

[19] Phukan et al., "Rethinking Cross-Corpus Speech Emotion Recognition Benchmarking," *Proc. APSIPA ASC*, 2025.

[20] Ibrahim et al., "What Does it Take to Generalize SER Model Across Datasets?," arXiv:2406.09933, 2024.

[21] F. Eyben, M. Wöllmer and B. Schuller, "openSMILE: The Munich Versatile and Fast Open-Source Audio Feature Extractor," *Proc. ACM Multimedia*, 2010.

[22] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, pp. 5–32, 2001.

[23] B. Schuller, B. Vlasenko, F. Eyben, M. Wöllmer, A. Stuhlsatz, A. Wendemuth and G. Rigoll, "Cross-Corpus Acoustic Emotion Recognition: Variances and Strategies," *IEEE Trans. Affective Computing*, vol. 1, no. 2, pp. 119–131, 2010.

[24] C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, "On Calibration of Modern Neural Networks," *Proc. ICML*, 2017.

[25] G. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

[26] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[27] M. Mauch and S. Dixon, "pYIN: A Fundamental Frequency Estimator Using Probabilistic Threshold Distributions," *Proc. ICASSP*, 2014.

> **Before submission:** check entries [15]–[20] against the papers themselves (full author lists, exact titles) and complete them; they are recorded here as noted in Phase 2. References [11]–[12] are intentionally unused (numbering kept stable with the literature matrix).

---

## Appendix A — Hypotheses at a glance

| Hypothesis | Verdict | Key evidence |
|---|---|---|
| H0: leakage inflates accuracy, more on RAVDESS | **Supported** | +7.8 to +9.3 (RAVDESS) vs +1.5 to +1.7 (CREMA-D) |
| H1: WavLM drops relatively more cross-corpus | **Direction-dependent** | +6.1 / +8.3 points from RAVDESS; −18.4 / −5.7 from CREMA-D |
| H2: resample > speaker > corpus > random | **Supported** for all 3 models | e.g. 0.78 > 0.63 > 0.50 > 0.00 (LogReg eGeMAPS), p < 10⁻¹⁰ |
| H3: stable core transfers better | **Inconclusive** | 1 gain, 4 losses, 7 null among 12 settings |

## Appendix B — Reproducibility

| Item | Value |
|---|---|
| Hardware | MacBook (Apple silicon), CPU + MPS for WavLM extraction |
| Software | Python 3.11, scikit-learn 1.9.1, LightGBM 4.7.0, openSMILE 2.6.0, transformers 5.20.1, torch 2.14.1, SHAP 0.51.0, Streamlit 1.66.0 (full list: `requirements.txt`) |
| Random seeds | folds 42; models 0 |
| Notebooks | `01_dataset_investigation` → `02_feature_extraction` → `03_modelling` → `04_explainability` → `05_experiments` → `06_evaluation` → `07_final_model` |
| App | `python -m src.app_assets` then `streamlit run app/streamlit_app.py` |
| Not in the repository | audio, extracted features, trained models (regenerated by the notebooks) |

## Appendix C — Repository map

```
data/README.md            download instructions (no audio committed)
data/metadata/            parsed labels, speakers, folds
src/                      audio · data · features · models · explain · experiments · evaluation · predict · app_*
notebooks/01–07           one notebook per phase step
results/                  CSV/JSON results and figures for every phase
report/                   phase reports 02–09, this report, assessment and viva material
app/streamlit_app.py      interactive demo
literature/               notes on the 10 papers, matrix, synthesis
```
