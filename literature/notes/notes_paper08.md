# Paper 08 — Cross-Corpus Training Strategy for SER Using Self-Supervised Representations

**Citation:** Pastor, M. A., Ribas, D., Ortega, A., Miguel, A., & Lleida, E. (2023). Cross-Corpus Training Strategy for Speech Emotion Recognition Using Self-Supervised Representations. *Applied Sciences*, 13(16), 9062. https://doi.org/10.3390/app13169062

**Free PDF:** https://www.mdpi.com/2076-3417/13/16/9062 (open access)
**Type:** Cross-corpus method paper

---

## What they did (plain language)
They asked: if you add other emotion datasets to the training data, does a model get better on your target dataset? They pooled EmoDB (German), RAVDESS and CREMA-D with IEMOCAP, using HuBERT and WavLM features.

## Datasets
| Corpus | Language | Size | Speakers |
|---|---|---|---|
| EmoDB | German | 16 min | 10 |
| RAVDESS | English | 42 min | 24 |
| CREMA-D | English | 203 min | 91 |
| IEMOCAP | English | 420 min | 10 (main target) |
4 classes: neutral, happiness, anger, sadness.

## Preprocessing
Not Reported (no resampling/normalisation details given).

## Features
HuBERT base and WavLM base (768-dim). Layer used: Not Reported.

## Models
Pooling + linear; **CNN + self-attention pooling** (best); CT-Transformer.

## XAI method
Not Reported.

## Evaluation protocol
5-fold CV, speaker-disjoint train/test, gender-balanced test sets; metric UAR.

## Reported results (IEMOCAP test, CNN, UAR %)
| Training data | HuBERT | WavLM |
|---|---|---|
| IEMOCAP only | 64.94 | 65.90 |
| + EmoDB + RAVDESS | 65.75 | **66.90** |
| + CREMA-D too | 64.72 | 65.06 |

- **Adding CREMA-D hurt** (authors blame its high variability).
- RAVDESS as target: 67.04 → 70.92 with extended training.
- EmoDB as target: no gain.
- A few minutes of in-domain data gave large relative gains.

## Limitations
- Gains are small; authors say more research is needed.
- Which dataset properties help or hurt is left open.
- Internal inconsistencies: confusion-matrix accuracy (~62%) does not match the reported UAR (66.90%).

## Relevance to Team Lucid
- Uses **both our datasets** and shows that **naively merging corpora can hurt** — RAVDESS and CREMA-D are not interchangeable.
- Supports a careful **train-on-one, test-on-the-other** design rather than blind merging.
- Reports only accuracy; **what the model learns from each corpus (explanations) is not examined.**

## Viva-ready points
- "Why not just merge RAVDESS and CREMA-D?" → This paper shows adding CREMA-D reduced performance; dataset shift matters.
