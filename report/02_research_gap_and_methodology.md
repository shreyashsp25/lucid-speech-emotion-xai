# Phase 2 — Research Gap and Methodology

Team Lucid · Speech Emotion Recognition with Explainable ML
Builds on: `literature/literature_matrix.md`, `literature/synthesis.md`
**Status: draft for team discussion. Nothing is final until the team confirms the chosen direction.**

---

## 1. What we checked before choosing (counter-evidence search)

A gap is only a gap if nobody has filled it. Our Phase 1 synthesis suggested that **SER papers do not test whether their explanations are stable**. So we searched specifically for recent work that might already do this.

| Paper found in Phase 2 | What it does | Does it close our gap? |
|---|---|---|
| Kaloga & Kodrasi (2025), *Towards Interpretable Emotion Recognition: Identifying Key Features with ML*, arXiv 2508.04230 | Aggregates feature importance across 4 models and **6 corpora** (French, Italian, German, English); speaker-independent; counts which descriptors recur | **Partly.** It checks whether features recur across datasets, but uses no formal stability metric, runs no cross-corpus train/test, and doesn't use RAVDESS or CREMA-D (English TESS only, 2 speakers) |
| Kołodziej et al. (2026), *Emotion Recognition Using Acoustic Features and Deep Learning: A Speaker-Independent Study*, Signals 7(4) | Polish corpus, 9 speakers; reports how often selected features recur **across LOSO folds**; compares classical vs wav2vec2/WavLM | **Partly.** Fold-level recurrence only; one corpus; no cross-corpus test; no SHAP stability |
| Nasr, Ren & Johnson (2025), *Beyond Saliency*, arXiv 2511.11691 | Grounds saliency explanations in expert acoustic cues; mainly CREMA-D | **No.** Studies explanation *plausibility* rather than *stability*; saliency, not SHAP on named features |
| Phukan et al. (2025), *Rethinking Cross-Corpus SER Benchmarking*, APSIPA ASC 2025 | Cross-corpus benchmark including **RAVDESS and CREMA-D**; pretrained models only | **No.** No handcrafted features; no explainability |
| Ibrahim et al. (2024), *What Does it Take to Generalize SER Across Datasets?*, arXiv 2406.09933 | 11 English corpora incl. RAVDESS/CREMA-D, Whisper | **No.** Accuracy only; no explainability |
| Nfissi et al. (2024), *Iterative Feature Boosting for Explainable SER*, arXiv 2405.20172 | SHAP-guided feature selection on TESS | **No.** One dataset, 2 speakers, random split (likely leaky), stability not measured |
| *Explainability guided feature selection for SER using SHAP and permutation importance* (Springer, 2026) | — | **Unknown.** We could not access the full text. **Must be checked before the final report.** |

**Conclusion of the search:** others have looked at whether important features *recur*, but only within single corpora or by simple counting. We found **no study that formally measures SHAP explanation stability across random seeds, speaker folds *and* corpora on RAVDESS ↔ CREMA-D, or that links explanation stability to cross-corpus accuracy**. We will describe this as an **underexplored question**, not as "the first ever", because our search is not exhaustive.

---

## 2. The five candidate directions

| | Direction | Problem | Supporting papers | Label |
|---|---|---|---|---|
| **A** | Explanation stability | SHAP rankings in SER come from one run on one dataset (P5, P9); SHAP is known to vary between runs (P10) | P5, P9, P10, Kaloga 2025, Kołodziej 2026 | Underexplored (partially studied) |
| **B** | Interpretable vs pretrained features across corpora | Embeddings win within a dataset (P4, P5), but the GeMAPS authors' claim that compact sets generalise better across corpora was never tested (P3) | P3, P4, P5, P8, Phukan 2025 | Underexplored for this corpus pair |
| **C** | Replicate and extend P5's "Energy > Frequency > Spectral > Temporal" finding | Shown only within RAVDESS/SAVEE | P5 | Replication + extension |
| **D** | Measure leakage inflation | Random splits inflate accuracy (P6) | P1, P6 | Engineering / rigour baseline |
| **E** | Stability-guided feature selection | SHAP-based selection exists (Nfissi 2024) but is never tested for whether it *transfers* to a new corpus | Nfissi 2024, P3, P8 | Underexplored |

### Comparison

| Criterion | A | B | C | D | E |
|---|---|---|---|---|---|
| Originality | High | Medium | Low–Med | Low | Medium–High |
| Feasibility on MacBook | High | Medium (WavLM extraction ~1 h CPU) | High | High | High |
| Uses both our datasets | Yes | Yes | Yes | Yes | Yes |
| Makes XAI central (project title) | **Yes** | Partly | Yes | No | **Yes** |
| Viva defensibility | High | High | Medium | Medium | High |
| Risk | Results may show "stable" (still a valid finding) | Either outcome is reportable | Low novelty | Low novelty | Gain may be small (still reportable) |

---

## 3. Recommendation: combine A + B + E, with D as a rigour baseline

A, B and E answer one connected question, and D makes the evaluation trustworthy. Together they fit the project title: XAI is the subject of the research, not decoration.

### Research question
> **When a speech emotion recognition model trained on one acted English corpus is applied to another, do its accuracy and its SHAP explanations hold up — and can explanation stability be used to choose features that transfer better?**

### Sub-questions and hypotheses

| | Sub-question | Hypothesis (falsifiable) |
|---|---|---|
| **RQ1** | How much accuracy is lost when moving between RAVDESS and CREMA-D, for interpretable (eGeMAPS) vs pretrained (WavLM) features? | **H1:** WavLM wins within-corpus (replicates P4, P5). The *relative* cross-corpus drop is smaller for eGeMAPS than for WavLM. |
| **RQ2** | How stable are SHAP global feature rankings across (a) random seeds, (b) speaker folds, (c) corpora? | **H2:** Stability decreases in the order seed > speaker fold > corpus. Corpus-level rankings agree significantly less than within-corpus rankings, but more than a random-ranking baseline. |
| **RQ3** | Does selecting features that are *consistently* important (a "stable core") transfer better than selecting the top features by importance alone? | **H3:** A model restricted to the stable core loses less accuracy cross-corpus than an equal-sized top-k-by-importance model. |
| (rigour) | How much do random splits inflate accuracy on each corpus? | **H0:** Random splits inflate accuracy, more on RAVDESS (24 speakers) than on CREMA-D (91). |

Any outcome is a finding. If H2 is rejected, for example if explanations turn out stable across corpora, that is good news for SER explainability and still a valid result.

### Proposed contribution (stated carefully)
1. A leak-free, speaker-independent comparison of eGeMAPS and WavLM features **within and across** RAVDESS and CREMA-D on 6 harmonised emotions.
2. A **quantitative stability analysis of SHAP explanations** for SER across seeds, speaker folds and corpora, using rank-based metrics (following P10's recommendation that magnitude metrics hide changes in the top features).
3. A test of whether **stability-guided feature selection** improves cross-corpus transfer.
4. An interactive demo that shows predictions together with **how reliable** each explanation is.

---

## 4. Methodology

### 4.1 Data (detailed in Phase 3)
- **RAVDESS** speech audio, 1,440 clips, 24 actors. **CREMA-D**, 7,442 clips, 91 actors.
- **Harmonised 6 classes:** neutral, happy, sad, angry, fearful, disgust. RAVDESS *calm* and *surprised* are dropped for cross-corpus work and kept for an optional within-RAVDESS 8-class run.
- Resample everything to **16 kHz mono**, as WavLM requires.
- `speaker_id` is parsed from filenames and used for every split.

### 4.2 Features
| Set | Dim | Tool | Why |
|---|---|---|---|
| **eGeMAPS v02** | 88 | `opensmile` | Interpretable, standard (P3); SHAP results are readable |
| **WavLM-base**, mean-pooled, chosen layer(s) | 768 | `transformers` (frozen) | Strong pretrained representation (P5, P7, P8) |
| Fusion (eGeMAPS + WavLM) | 856 | — | Tests complementarity (P4) |

Normalisation is **fitted on training data only**. Per-speaker normalisation is reported separately because it uses test-speaker statistics (P4 caveat).

### 4.3 Models (no KNN)
| Role | Model | Why |
|---|---|---|
| Baseline | **Logistic Regression** (L2) | Simple, linear, exact SHAP via LinearExplainer; P5 used it |
| Strong classical | **SVM (RBF)** | Standard in paralinguistics (P3) |
| Strong tree ensemble | **LightGBM** or **XGBoost** | Handles non-linear interactions; fast, exact TreeSHAP |
| Neural | **MLP** (small) | Standard head on frozen embeddings (P4, P7) |

Hyperparameters are tuned with **nested, speaker-grouped CV** and are never tuned on the test set.

### 4.4 Explainability
- **Global:** mean |SHAP| per feature (and per class) → ranking.
- **Local:** per-clip SHAP waterfall plots (for the demo and error analysis).
- **Cross-check:** permutation importance, to see whether a second XAI method agrees.
- **Stability metrics:** Spearman and Kendall rank correlation, **top-k Jaccard overlap** (k = 5, 10, 20), sign agreement, all compared against a **random-ranking null distribution** (P10).
- Explanations are described as *associations the model uses*, **not** causes of emotion.

### 4.5 Experiments

| ID | Experiment | Answers |
|---|---|---|
| **E0** | Random split vs speaker-grouped split, each corpus | H0 |
| **E1** | Within-corpus: speaker-grouped 5-fold × 5 seeds × {eGeMAPS, WavLM, fusion} × {LR, SVM, GBM, MLP} | baseline for H1 |
| **E2** | Cross-corpus: train RAVDESS → test CREMA-D, and the reverse | H1 |
| **E3** | SHAP stability: compare rankings across seeds, folds, corpora | H2 |
| **E4** | Stable-core vs top-k feature selection (selection inside training only) → cross-corpus test | H3 |
| **E5** | Error analysis: confusion matrices, per-emotion recall, explanations of misclassified clips | discussion |

**Metrics:** UAR (macro recall) as primary, plus macro-F1, accuracy, per-class recall and confusion matrices. Reported as **mean ± std over seeds and folds**, with **bootstrap 95% CIs** and paired tests (e.g., Wilcoxon) for key comparisons.

**Reference points:** human audio-only accuracy is 63% for RAVDESS (P1) and ~41% for CREMA-D (P2).

### 4.6 Pipeline
```
audio → resample 16 kHz → speaker-id + label parsing → harmonise 6 classes
      → features (eGeMAPS | WavLM | fusion) → speaker-grouped CV / cross-corpus split
      → train (LR | SVM | GBM | MLP) → evaluate (UAR, F1, CI)
      → SHAP global + local → stability metrics → stable-core selection → re-evaluate
      → demo app (prediction + explanation + reliability)
```

### 4.7 Compute
- eGeMAPS for ~8,900 clips takes minutes on CPU.
- WavLM-base extraction is a one-off of roughly 1 hour on an M-series Mac. Features are cached to disk and git-ignored.
- All models train in minutes. No GPU is needed.

### 4.8 Syllabus links
Supervised classification · Logistic Regression · SVM · Decision-tree ensembles / boosting · MLP · feature selection · PCA (optional, for visualising embeddings) · cross-validation and hyperparameter tuning · overfitting and leakage · evaluation metrics · bias (speaker and corpus effects) · explainability.

---

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| CREMA-D labels are noisy (α = 0.42, P2) | Report results with intended labels; sensitivity run on high-agreement clips only |
| RAVDESS test folds are small → high variance | Many seeds and folds; report CIs, not single numbers |
| Class imbalance after harmonising | Class-weighted training; UAR as primary metric |
| SHAP on 768 WavLM dimensions is not human-readable | Stability metrics still apply; human interpretation focuses on eGeMAPS (optionally P5-style probing) |
| Unread Springer 2026 paper may overlap | Check its full text before the final report; adjust the contribution wording if needed |
| Results don't support hypotheses | Report honestly; rejected hypotheses are findings |

---

## 6. What we will *not* claim
- Not "first ever" or "state-of-the-art".
- Not that a feature *causes* an emotion.
- No comparison of our numbers with papers that used random splits unless we note the difference.

---

## 7. Decisions needed from the team
1. **Approve the research question** (A + B + E with D as baseline), or pick a different direction from Section 2.
2. **Gradient boosting library:** LightGBM or XGBoost. (Recommendation: LightGBM, which is faster on Mac.)
3. **Label policy for CREMA-D:** intended labels with an agreement-subset check (recommended), or perceived labels.
