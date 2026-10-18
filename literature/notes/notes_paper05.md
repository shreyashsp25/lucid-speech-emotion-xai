# Paper 05 — Explaining Deep Learning Embeddings for SER via Interpretable Acoustic Features

**Citation:** Dixit, S., Low, D. M., Elbanna, G., Catania, F., & Ghosh, S. S. (2024). Explaining Deep Learning Embeddings for Speech Emotion Recognition by Predicting Interpretable Acoustic Features. *arXiv preprint* arXiv:2409.09511 (submitted to IEEE).

**Free PDF:** https://arxiv.org/abs/2409.09511
**Type:** XAI / probing method paper

---

## What they did (plain language)
WavLM embeddings classify emotion very well but are 1,024 meaningless numbers. The authors used SHAP to find which embedding dimensions matter most for each emotion, then checked which **real acoustic features** (from eGeMAPS) those important dimensions can predict. If the important dimensions predict "loudness" well, they infer loudness information is what the model relies on.

## Datasets
RAVDESS and SAVEE (both acted, English, lab-recorded).

## Preprocessing
Speaker normalisation before classification.

## Features
- eGeMAPSv02 (openSMILE), grouped into Energy, Frequency, Spectral, Temporal.
- WavLM Large, mean-pooled to one **1,024-dim** vector per clip.

## Models
One **binary classifier per emotion** (emotion vs neutral), L2-regularised **Logistic Regression**. Probing: linear ridge regression from top dimensions → each eGeMAPS feature.

## XAI method
**SHAP** to rank features/dimensions per emotion + probing classifiers.

## Evaluation protocol
**Speaker-independent 5-fold nested CV** (C tuned inside training). Metric: F1.

## Reported results (F1, RAVDESS / SAVEE)
| Emotion | eGeMAPS | WavLM |
|---|---|---|
| Anger | 93.1 / 99.4 | 97.7 / 99.4 |
| Fear | 92.6 / 97.9 | 97.5 / 99.4 |
| Joy | 88.8 / 98.7 | 96.8 / 99.4 |
| Sadness | 79.0 / 70.0 | 83.0 / 97.5 |
| Disgust | 89.1 / 62.4 | 99.1 / 98.8 |

- WavLM > eGeMAPS for every emotion.
- Information in important WavLM dims: **Energy > Frequency > Spectral > Temporal**, consistent across emotions and both datasets.
- Note: numbers are for **binary emotion-vs-neutral** tasks, not the usual multi-class task — much easier; not comparable to 7/8-class accuracies.

## Limitations (stated)
- Shows what acoustic information is *encoded*, not proven to be what the model *uses*.
- Not tested on other datasets/languages or naturalistic speech.
- Only one embedding model and one handcrafted set.

## Relevance to Team Lucid
- Closest prior work to an **"explainable SER"** project using eGeMAPS + SHAP on RAVDESS.
- Uses only **within-dataset** analysis; whether the Energy > Frequency > … ranking holds on **CREMA-D** or **across corpora** is untested — a concrete extension.
- Shows a method to make embeddings interpretable; we could reuse the idea.
- SHAP rankings computed once — **stability of those rankings** across folds/seeds is not reported (links to Paper 10).

## Viva-ready points
- "Has explainable SER with SHAP been done?" → Yes (this paper), but within-dataset and without checking explanation stability.
- "Why are their F1 scores so high?" → Binary emotion-vs-neutral tasks, small acted datasets.
