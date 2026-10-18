# Phase 1 Synthesis — What the 10 Papers Tell Us Together

Team Lucid · Speech Emotion Recognition with Explainable ML
Based on: `literature_matrix.md` and `notes/notes_paper01–10.md`

---

## 1. Common datasets
- Acted, studio-recorded English corpora dominate: **RAVDESS** (Papers 1, 4, 5, 8, 9), **CREMA-D** (2, 7, 8), IEMOCAP (4, 6, 7, 8).
- Human listeners are far from perfect on audio alone: **63% on RAVDESS** (P1) and **~41% on CREMA-D** (P2). These are useful reference points when judging a model's accuracy.
- CREMA-D's intended labels often disagree with what listeners perceive (Krippendorff's α = 0.42, P2).

## 2. Common feature representations
| Family | Example | Strength | Weakness |
|---|---|---|---|
| Handcrafted, standardised | eGeMAPS, 88 features (P3) | Each feature has physical meaning; compact | Lower in-dataset accuracy (P4, P5) |
| Generic spectral | MFCC, chroma, mel (P9) | Easy to compute | When flattened, explanations become unreadable indices (P9) |
| Pretrained self-supervised | wav2vec 2.0, HuBERT, WavLM (P4, P5, P7, P8) | Highest in-dataset accuracy | Hundreds of uninterpretable dimensions; generalisation doubted (P4's own repo) |

Consistent finding: **pretrained embeddings > eGeMAPS within a dataset** (P4: 84.3% vs 57.0% UAR on RAVDESS; P5: higher F1 for every emotion). Fusing the two gives small gains (P4), which suggests handcrafted prosody still adds complementary information.

## 3. Common models
Mostly simple models on top of features: SVM (P3), logistic regression (P5), small dense heads on frozen embeddings (P4, P7, P8). Frozen-encoder + small-head is the accepted standard (P7). This suits laptop compute.

## 4. Explainability approaches
- **Interpretable-by-design features** (P3).
- **Post-hoc SHAP** on interpretable features or embeddings (P5), and SHAP/LIME on spectral inputs (P9).
- **Probing** embeddings with acoustic features (P5).
- What is missing in the SER papers we read:
  - **No paper checks whether SHAP rankings are stable** across seeds, folds or datasets. P5 and P9 each report a single ranking.
  - P10 shows SHAP rankings can change substantially between runs. Model-induced variation is larger on **small datasets**, and RAVDESS is a small dataset.
  - P9's explanations name anonymous feature indices, so a reader cannot interpret them.

## 5. Evaluation practices and known problems
- **Speaker leakage**: random splits put the same speaker in train and test (P6). Credible work uses speaker-independent splits (P3 LOSO, P4 actor split, P5, P7, P8).
- **Inconsistent protocols** (classes dropped, different folds, WA vs UA) make published numbers hard to compare (P6).
- **Speaker normalisation** with test-speaker statistics can inflate results (P4).
- **Reporting single runs** without variance is common (P6, P7).

## 6. Generalisation across datasets
- Merging corpora gives **small, inconsistent gains**, and adding CREMA-D actually **lowered** IEMOCAP performance (P8).
- The GeMAPS authors suggested compact feature sets might generalise better across corpora but **did not test it** (P3).
- P5's finding that energy features carry the most information was shown only **within** RAVDESS and SAVEE.

## 7. Disagreements and tensions
- Accuracy vs interpretability: embeddings win on accuracy (P4, P5), while eGeMAPS wins on readability (P3).
- Merging corpora helps in some setups and hurts in others (P8 vs its own cited prior work).
- Bigger models are not always better (P7: DeCoAR 2 beats wav2vec 2.0).

## 8. Overall weaknesses in the literature (as seen in our 10 papers)
1. Explanations are reported from **one run on one dataset**, and their stability is not measured.
2. Cross-corpus work reports **accuracy only**, not whether what the model relies on changes between corpora.
3. Claims about compact features generalising better are **untested** for RAVDESS ↔ CREMA-D.
4. Evaluation protocols are inconsistent, and speaker leakage is common.

## 9. Candidate gaps to take into Phase 2
These are candidates only. Phase 2 must search for counter-evidence before we claim novelty, because 10 papers is not the whole field.

| Direction | Supported by | Provisional label |
|---|---|---|
| A. Are SER explanations (SHAP rankings) stable across seeds, speaker folds and corpora? | P5, P9, P10 | Genuine-gap candidate |
| B. Do interpretable features (eGeMAPS) lose less accuracy than embeddings when moving RAVDESS ↔ CREMA-D? | P3, P4, P8 | Underexplored |
| C. Does P5's energy > frequency > spectral > temporal ranking hold on CREMA-D and cross-corpus? | P5 | Replication + extension |
| D. How much do random splits inflate accuracy on RAVDESS vs CREMA-D? | P1, P6 | Engineering / rigour baseline |

**Observation:** A, B and C can be combined into one research question: *"Do the accuracy and the explanations of an SER model learned on one corpus hold up on another?"* D would then be a supporting experiment. The final choice belongs to Phase 2.

---

## What remains before Phase 1 is complete
- Each team member reads their assigned papers and checks these notes against the PDFs, correcting anything wrong.
- Resolve the two **TO VERIFY** venues (papers 6 and 7).
- Team agrees that this synthesis reflects the papers.
