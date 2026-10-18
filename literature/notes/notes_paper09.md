# Paper 09 — XAI Techniques for SER: A Focus on XAI Models

**Citation:** Norval, M., & Wang, Z. (2025). Explainable Artificial Intelligence Techniques for Speech Emotion Recognition: A Focus on XAI Models. *Inteligencia Artificial*, 28(76), 85–123. https://doi.org/10.4114/intartif.vol28iss76pp85-123

**Free PDF:** https://journal.iberamia.org/index.php/intartif/article/view/1809 (open access)
**Type:** Applied XAI paper

---

## What they did (plain language)
They trained CNN, CNN-LSTM and XGBoost models (and an ensemble) on an Afrikaans emotional speech corpus, then applied SHAP and LIME to explain predictions.

## Dataset
Custom **Afrikaans** corpus (authors' earlier work): ~800 clips, ~100 per emotion. Emotions (Plutchik): anger, anticipation, disgust, fear, joy, sadness, surprise, trust. Number of speakers: Not Reported.

## Preprocessing
MFCC normalisation; augmentation by pitch shift and time stretch.

## Features
MFCC, chroma, mel spectrogram, spectral contrast, tonnetz — flattened into tens of thousands of numeric inputs.

## Models
CNN (Conv1D), CLSTM (Conv1D + LSTM), XGBoost, stacking ensemble.

## XAI method
**SHAP** (DeepExplainer for CNN/CLSTM, TreeExplainer for XGBoost) and **LIME** (tabular).

## Evaluation protocol
Random 20% validation split; **speaker independence not stated** (likely speaker-dependent).

## Reported results
- No per-model accuracy/F1 tables for the Afrikaans corpus in the text.
- Claims UAR 89.4% / WAR 92.1% on RAVDESS, but the RAVDESS setup is not described → **not verifiable**.

## Explanation findings
- Top features reported as anonymous indices (e.g., "Feature36766") with **no mapping back to acoustic meaning**.
- No quantitative SHAP-vs-LIME comparison despite the title's framing.

## Limitations
- Explanations are not interpretable (feature indices, not named acoustic properties).
- Inconsistent epochs and data shapes across sections; split protocol unclear.
- Single corpus; no cross-corpus test; no stability check of explanations.

## Relevance to Team Lucid
- A **cautionary example**: applying SHAP to thousands of unnamed spectral values gives explanations nobody can read.
- Strengthens the case for **named, physically meaningful features** (eGeMAPS) and for **evaluating** explanations, not just plotting them.

## Viva-ready points
- "What's wrong with just applying SHAP?" → Without meaningful features and a leak-free split, SHAP can only explain shortcuts, and the output may be unreadable (as here).
