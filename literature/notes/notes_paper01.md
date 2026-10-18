# Paper 01 — RAVDESS

**Citation:** Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. *PLOS ONE*, 13(5), e0196391. https://doi.org/10.1371/journal.pone.0196391

**Free PDF:** https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391 (open access)
**Dataset download:** https://zenodo.org/records/1188976 — licence CC BY-NC-SA 4.0 (non-commercial, share-alike)
**Type:** Dataset paper (no ML experiments)

---

## What they did (plain language)
They recorded 24 professional actors saying two fixed sentences in many emotions and at two intensities, then had hundreds of ordinary people rate every recording to check that the intended emotion actually comes across.

## Dataset design
| Item | Value |
|---|---|
| Actors | 24 (12 M, 12 F), aged 21–33; odd IDs male, even IDs female |
| Sentences | 2 only: "Kids are talking by the door", "Dogs are sitting by the door" |
| Speech emotions | 8: neutral, calm, happy, sad, angry, fearful, surprise, disgust |
| Intensity | normal and strong (neutral has no strong version) |
| Modalities | audio-visual, video-only, audio-only |
| Files | 7,356 total; **1,440 audio-only speech files** (60 per actor) — the part we use |
| Audio | Rode NTK mic at 20 cm, **48 kHz, 16-bit** WAV, peak-normalised to −3 dBFS |
| Recording | Professional studio, actors induced emotion via acting techniques |

## Validation (how good are the labels?)
- 247 untrained raters; each file rated **10 times** for emotion category, intensity and genuineness.
- **Human accuracy, audio-only speech: 63%** (95% CI 62–64%). Audio-visual 81%, video-only 75%.
- Inter-rater agreement (Fleiss' κ): 0.62 for strong-intensity speech, 0.53 for normal.
- Test–retest (72 extra raters): κ ≈ 0.73 average.
- **Calm and neutral are often confused** with each other and with sad/happy.

## Preprocessing / Features / Models / XAI
Not Reported — this is a dataset paper; no classifiers were trained.

## Evaluation protocol
Perceptual validation only (human raters). No ML split is defined by the authors.

## Limitations (stated or evident)
- Only 24 speakers and **2 sentences** → models can memorise speaker voice and wording; random splits leak speaker identity.
- Acted, studio-quality speech; not spontaneous or noisy.
- Raters were undergraduates in one region (not general population).
- Intensity and genuineness ratings had low single-rater reliability.
- Note: paper text writes the licence as "CC BY-NA-SC 4.0"; the Zenodo page states the correct **CC BY-NC-SA 4.0**.

## Relevance to Team Lucid
- One of our two core datasets. Human audio-only accuracy (63%) gives a **reference ceiling** for interpreting our model's accuracy.
- The small speaker count is the reason we **must use speaker-independent (actor-grouped) splits**.
- Calm/neutral confusion explains why some papers merge them (e.g., Paper 04) — a choice we must justify.
- Emotion set overlaps CREMA-D on 6 classes (drop calm, surprise) → enables cross-corpus tests.

## Viva-ready points
- "Why not random split on RAVDESS?" → 24 speakers, 60 clips each; random split puts the same voice in train and test.
- "What does 63% human accuracy tell you?" → The task is genuinely ambiguous; >90% claims on RAVDESS deserve scrutiny of the split.
