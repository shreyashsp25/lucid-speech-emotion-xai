# Phase 7 — Explainable AI

Team Lucid · Results from `notebooks/04_explainability.ipynb` (19 Oct 2026). Raw numbers: `results/phase7_summary.json`, `results/phase7/`.

---

## 1. Method

| Item | Choice | Why |
|---|---|---|
| Model explained | **Logistic Regression on eGeMAPS** (main) and LightGBM on eGeMAPS (comparison) | eGeMAPS features have physical names, so explanations are readable. WavLM's 768 unnamed dimensions are not; their *stability* is measured in Phase 8 |
| Explainer | **SHAP**, exact in both cases: linear SHAP for LogReg, TreeSHAP for LightGBM | No sampling noise from the explainer itself (cf. P10) |
| Linear SHAP formula | contribution = weight × (clip value − training mean), in standardised, clipped units (log-odds) | Verified identical to `shap.LinearExplainer` (max difference 0.0; additivity error ≈ 10⁻¹⁴) |
| Out-of-fold | each clip is explained by a model trained **without its speaker**, with the Phase 6 hyperparameters | Explanations describe models that generalise to new speakers, not memorised ones |
| Second method | **Permutation importance** (drop in test UAR when a feature is shuffled, 10 repeats) | Cross-checks SHAP with a completely different method |
| Feature groups | Energy (15) · Frequency (24) · Spectral (43) · Temporal (6), following GeMAPS (P3) / P5 | To test P5's group ranking |

The explained models reproduce the Phase 6 scores (out-of-fold UAR: RAVDESS 55.4% LogReg / 53.9% LightGBM; CREMA-D 51.4% / 53.7%), so we are explaining the models we evaluated.

> SHAP describes what the **model** uses. It is association, not proof that a voice property causes an emotion.

## 2. A preprocessing flaw found through explainability

The first SHAP run showed single features contributing up to **±170 log-odds** to one clip, against typical values below 1. The cause: eGeMAPS "stddevNorm" features are std ÷ mean, and they explode when the mean is near zero. In CREMA-D, 138 clips (1.9%) had values beyond 10 SD, the worst at 86 SD.

**Fix:** clip standardised features to ±5 SD inside every pipeline (fitted on training data only), then re-run all of Phase 6.

| | Before | After |
|---|---|---|
| eGeMAPS LogReg UAR, RAVDESS | 54.6% | 55.4% |
| eGeMAPS LogReg UAR, CREMA-D | 51.1% | 51.4% |
| Largest single SHAP contribution | ≈170 | ≈10 |

**Lesson:** accuracy alone hid the problem. The explanation exposed it. This is a concrete case of XAI being used to *debug* a model, not just to decorate it.

## 3. Global importance (top features, LogReg)

`results/figures/p7_global_top15.png`

| Rank | RAVDESS | CREMA-D |
|---|---|---|
| 1 | mfcc1 mean (Spectral) | **equivalent sound level** (Energy) |
| 2 | F1 frequency mean (Frequency) | **loudness mean** (Energy) |
| 3 | mfcc3 mean (Spectral) | F2 amplitude rel. F0, mean (Spectral) |
| 4 | Hammarberg index (voiced) variation (Spectral) | F2 amplitude rel. F0, variation (Spectral) |
| 5 | alpha ratio (unvoiced) (Spectral) | loudness rising slope (Energy) |
| 6 | equivalent sound level (Energy) | spectral flux (unvoiced) (Spectral) |
| 7 | HNR mean (Energy) | mfcc2 mean (Spectral) |
| 8 | loudness variation (Energy) | mfcc2 (voiced) mean (Spectral) |
| 9 | mfcc2 (voiced) mean (Spectral) | spectral flux mean (Spectral) |
| 10 | mfcc4 mean (Spectral) | F3 amplitude rel. F0, variation (Spectral) |

The CREMA-D model leans on **loudness / energy**. The RAVDESS model leans on **spectral shape (MFCCs) and vowel formants (F1)**.

## 4. Feature groups vs Paper 5

Share of total |SHAP| (LogReg); in brackets, the share divided by group size (importance per feature):

| Group | RAVDESS | CREMA-D |
|---|---|---|
| Energy (15 features) | 19.7% (per feature: **0.29**) | 23.5% (per feature: **0.36**) |
| Frequency (24) | 27.9% (0.25) | 18.5% (0.18) |
| Spectral (43) | 46.2% (0.24) | 53.4% (0.29) |
| Temporal (6) | 6.2% (0.22) | 4.7% (0.18) |

- **Per feature, Energy is the most informative group in both corpora**, in line with P5's finding that Energy carries the most emotion information.
- After that, the order **differs between corpora**: RAVDESS is Frequency > Spectral > Temporal, while CREMA-D is Spectral > Frequency ≈ Temporal. P5's full ranking (Energy > Frequency > Spectral > Temporal) is **reproduced on RAVDESS but only partly on CREMA-D**. P5 studied only RAVDESS and SAVEE, so this is new evidence that the ranking depends on the corpus.
- In raw totals, Spectral dominates simply because it has 43 of the 88 features.

## 5. What drives each emotion (direction of effect)

From `results/phase7/per_emotion_drivers.csv`. ↑ = a higher value pushes **toward** the emotion, ↓ = away.

| Emotion | CREMA-D (strongest drivers) | Plausible acoustic reading |
|---|---|---|
| angry | loudness rising slope ↑, alpha ratio ↑, equivalent sound level ↑, median pitch ↓ | Loud, fast-rising energy, more high-frequency energy: a tense, high-effort voice |
| sad | **sound level ↓** (by far the strongest single driver), spectral flux ↓ | Quiet, low-effort, little spectral change |
| disgust | loudness mean ↑, loudness range ↑ | Energy dynamics |
| happy | spectral flux ↑, (pitch-related) | Lively, changing spectrum |
| fearful | formant amplitudes (F2/F3 rel. F0) ↑ | Changed voice quality / breathiness |
| neutral | sound level ↑, spectral flux ↓ | Steady, unremarkable delivery |

- **The directions match textbook vocal-emotion acoustics** (P3's literature review): anger is louder with steeper energy rises, and sadness is quieter. This is a plausibility check on the explanations, not a proof.
- On RAVDESS, *happy* is driven by **higher median pitch** and pitch variation (↑), also consistent with the literature.

## 6. Cross-check: SHAP vs permutation importance

| Corpus | Spearman (88 features) | Top-10 overlap |
|---|---|---|
| RAVDESS | 0.79 | 5 / 10 |
| CREMA-D | **0.96** | 7 / 10 |

The two methods broadly agree, strongly so on CREMA-D. Agreement is lower on RAVDESS, which has fewer clips (noisier permutation estimates) and many **correlated features**: permutation importance under-rates a feature whose information a correlated "twin" still carries, while SHAP splits credit between them.

## 7. Local explanations (`results/figures/p7_local_examples.png`)

Typical clips were chosen (median evidence), not extreme ones.
- **`1068_MTI_ANG_XX.wav`, correctly predicted angry:** pushed toward angry by unvoiced spectral flux, formant amplitudes, loudness and the loudness rising slope.
- **`1013_WSI_FEA_XX.wav`, fearful mispredicted as sad:** pushed toward sad mainly by a **low sound level** (−38 dB) and steady loudness. The model treats "quiet" as "sad", and a quiet fearful delivery falls into that trap. This matches the main confusion in Phase 6 (fearful → sad, 17%) and where human listeners also struggled (Phase 3).

## 8. Shortcut check: the 39 "no-pitch" CREMA-D clips

| | With these clips | Model retrained without them |
|---|---|---|
| UAR (on the other 7,403 clips) | 51.41% | 51.40% |
| Sad recall | 57.8% | 58.2% |

The pitch features do push no-pitch clips slightly toward sad (+0.09 log-odds on average), but removing those clips changes nothing measurable. **No harmful "no pitch → sad" shortcut.**

## 9. First look at our research question (H2): do the two corpora agree?

| Model | Spearman of global importances (RAVDESS vs CREMA-D) | Top-10 Jaccard | Shared top-10 features |
|---|---|---|---|
| LogReg | 0.54 | **0.11** (2 of 10 shared) | equivalent sound level, mfcc2 (voiced) |
| LightGBM | 0.24 | 0.11 | pitch variation, unvoiced spectral flux |

The overall importance profiles are moderately correlated, but the **top features largely differ** between corpora. This fits the corpus shift already seen in Phase 5 (PCA) and in this phase: formant-amplitude features sit at very different levels in the two corpora (median F1 amplitude −83 vs −150 dB rel. F0), likely from different recording conditions.

**This is a preview, not a result.** Phase 8 (E3) tests it properly: stability across seeds and folds *within* a corpus versus *across* corpora, against a random-ranking baseline.

## 10. Limitations
- SHAP on correlated features splits credit among them; feature-level rankings are less stable than group-level ones.
- Explanations are for eGeMAPS models (≈55% / 51% UAR), not for the stronger WavLM models (≈81% / 74%). The interpretable model is less accurate, a trade-off we state explicitly.
- Direction and plausibility checks compare against the literature; they do not establish causation.
