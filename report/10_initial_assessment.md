# Initial Assessment (6 marks) — Slide-by-slide content

Team Lucid · Target length: **10–12 minutes, 11 slides**. Each slide lists what to show, what to say, and who presents it.

The marking criteria are covered as follows: problem (slide 2), motivation (slide 3), objectives (slide 4), literature survey (slides 5–6), research gap (slide 7), dataset justification (slide 8), methodology (slide 9), architecture/workflow (slide 10), proposed contribution (slide 11).

---

### Slide 1 — Title (Pranav)
- **Show:** *Emotion Recognition Using Explainable Machine Learning: do accuracy and explanations survive a change of corpus?* Team names and roll numbers.
- **Say:** "We recognise emotions from speech, explain every prediction, and test whether both the predictions and the explanations still hold on a different dataset."

### Slide 2 — Problem (Pranav)
- **Show:** three boxes: *Speaker leakage* · *Unknown transfer to new data* · *Explanations never checked*.
- **Say:** Most SER results come from one dataset, often with random splits that let the same speaker appear in training and test. Explanations (SHAP) are reported from one run on one dataset, although SHAP rankings are known to vary between runs (Paper 10).

### Slide 3 — Motivation (Pranav)
- **Show:** an example: model trained in one studio → used with a new microphone → accuracy and explanation both change.
- **Say:** A model or explanation that only works on its own dataset gives a false sense of understanding. We make explainability the object of study, not decoration.

### Slide 4 — Objectives (Piriya)
1. Leak-free, speaker-independent SER for 6 emotions on RAVDESS and CREMA-D.
2. Compare interpretable eGeMAPS vs pretrained WavLM features, four model families (no KNN).
3. Measure leakage inflation.
4. Explain models with SHAP, cross-checked by permutation importance.
5. Test cross-corpus transfer, explanation stability and stability-guided feature selection.
6. Build an honest interactive demo.

### Slide 5 — Literature survey: what is known (Piriya)
| Theme | Papers | Finding |
|---|---|---|
| Datasets | P1 RAVDESS, P2 CREMA-D | Humans reach 63% / ~41% on audio alone |
| Interpretable features | P3 eGeMAPS | 88 standard features; cross-database generalisation untested |
| Pretrained embeddings | P4, P7, P8 | Beat hand-crafted features within a dataset (84.3 vs 57.0% UAR, P4) |
| Explaining SER | P5, P9 | SHAP on named features; Energy > Frequency > Spectral > Temporal (P5) |

### Slide 6 — Literature survey: what goes wrong (Praveen)
| Problem | Papers |
|---|---|
| Speaker leakage, inconsistent protocols | P6 |
| Merging corpora gives small or negative gains | P8 |
| SHAP rankings vary between runs; single runs reported | P10, P5, P9 |
| Explanations with unreadable feature indices | P9 |

### Slide 7 — Research gap (Praveen)
- **Show:** counter-evidence table (Kaloga & Kodrasi 2025; Kołodziej 2026; Phukan 2025; Ibrahim 2024; Nfissi 2024): each covers part of the problem, none measures SHAP stability across resamples, speakers *and* corpora for RAVDESS ↔ CREMA-D or links it to transfer.
- **Research question:** *When an SER model trained on one corpus is applied to another, do its accuracy and its SHAP explanations hold up, and can explanation stability be used to choose features that transfer better?*
- **Hypotheses:** H0 leakage inflates (more on RAVDESS) · H1 embeddings drop relatively more cross-corpus · H2 stability resample > speaker > corpus > random · H3 stable features transfer better.
- **Say:** "We call this underexplored, not a first: our search was careful but not exhaustive."

### Slide 8 — Dataset justification (Pranav)
| | RAVDESS | CREMA-D |
|---|---|---|
| Speakers | 24, young, one accent | 91, ages 20–74, diverse |
| Sentences | 2 | 12 |
| Clips used (6 emotions) | 1,056 | 7,442 |
| Recording | 48 kHz studio | 16 kHz |
- **Say:** both free and standard; they share six emotions; they differ in size, diversity and recording, so they give a realistic cross-corpus test. CREMA-D listener votes let us study label quality.

### Slide 9 — Methodology (Piriya)
- Preprocessing: mono, 16 kHz, trim silence (anything 30 dB below the loudest point counts as silence), no volume normalisation.
- Features: eGeMAPS (88), WavLM base-plus (768, layer chosen by inner CV), fusion.
- Models: LogReg, RBF-SVM, LightGBM, MLP; **KNN excluded** (rule; also distance-based methods are fragile under corpus shift and give no global explanation).
- Evaluation: 5 speaker-independent folds (grouped by speaker, stratified by emotion and gender); nested CV for tuning; UAR primary; speaker-level bootstrap CIs.
- XAI: exact SHAP (linear and tree), permutation importance, rank-stability metrics vs a random baseline.
- Experiments E0–E5 (leakage, within-corpus, cross-corpus, stability, stable-core selection, error analysis).

### Slide 10 — Architecture / workflow (Praveen)
```
audio ─► mono · 16 kHz · trim (−30 dB) ─┬─► eGeMAPS (88) ─┐
                                        └─► WavLM layer ──┴─► speaker-independent CV / cross-corpus
                                                      ─► LogReg | SVM | LightGBM | MLP
                                                      ─► UAR, F1, CI
                                                      ─► SHAP ─► stability ─► stable-core selection
                                                      ─► demo app (prediction + explanation + reliability)
```
Tools: Python, scikit-learn, LightGBM, openSMILE, Hugging Face transformers, SHAP, Streamlit; macOS, VS Code, Jupyter, GitHub.

### Slide 11 — Proposed contribution and plan (all)
1. Leak-free comparison of interpretable vs pretrained features within and across corpora.
2. Quantitative SHAP stability analysis for SER.
3. Test of stability-guided feature selection for transfer.
4. Honest interactive demo with explanations and reliability.
- **Timeline:** Phases 1–10 (literature → report). **Work split:** see `10_team_plan.md`.

---

## Likely questions at this stage
- *Why not deep learning end-to-end?* → Frozen self-supervised encoder + simple classifier is the accepted standard (P7) and runs on a laptop; it also keeps explanations exact.
- *Why SHAP over LIME?* → Exact for our models, deterministic; LIME adds sampling noise, which would confuse a stability study.
- *What if the hypotheses fail?* → Every outcome is reportable; a failed hypothesis is a finding.
