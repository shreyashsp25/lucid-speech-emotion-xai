# Paper 04 — Emotion Recognition from Speech Using wav2vec 2.0 Embeddings

**Citation:** Pepino, L., Riera, P., & Ferrer, L. (2021). Emotion Recognition from Speech Using wav2vec 2.0 Embeddings. *Proc. Interspeech 2021*, 3400–3404. https://www.isca-archive.org/interspeech_2021/pepino21_interspeech.html

**Free PDF:** https://arxiv.org/abs/2104.03502 (arXiv) / ISCA Archive
**Code:** https://github.com/habla-liaa/ser-with-w2v2
**Type:** Method paper (pretrained speech representations)

---

## What they did (plain language)
They took wav2vec 2.0 — a model pretrained on lots of unlabelled English speech — froze it, and used its internal layers as features. A small neural network learns how much to trust each layer and then classifies emotion. It beat hand-crafted features by a wide margin.

## Datasets
- **IEMOCAP** — 4 classes (anger, happy+excited, sad, neutral); no-majority samples dropped.
- **RAVDESS** — calm and neutral **merged** → 7 classes.

## Preprocessing
- IEMOCAP clips trimmed to 15 s; inputs capped at 8 s (IEMOCAP) / 5 s (RAVDESS).
- **Speaker normalisation** (per-speaker mean/std) for main results; global normalisation as comparison.

## Features
- wav2vec 2.0 **base** (768-dim, 12 transformer layers), frozen. Two versions: PT (pretrained only) and FT (fine-tuned for ASR).
- All 13 layer outputs combined with **learned weights**.
- Baselines: eGeMAPS (opensmile-python) and magnitude spectrogram.

## Models
Two per-frame dense layers (128 units) → time-average → softmax. Variants: LSTM head; fusion with eGeMAPS branch.

## XAI method
None formal; they inspect the **learned layer weights** (middle layers weighted most).

## Evaluation protocol
- RAVDESS: **actor-split** — actors 1–19 train, 20–21 val, 22–24 test (speaker-independent). (Paper's text lists overlapping validation/test ranges — likely a typo.)
- IEMOCAP: 5-fold leave-one-session-out.
- Metric: average recall (UAR), mean ± std over 5 seeds.

## Reported results (UAR %, speaker-normalised)
| Features | IEMOCAP | RAVDESS |
|---|---|---|
| eGeMAPS | 52.4 | 57.0 |
| Spectrogram | 49.8 | 44.5 |
| wav2vec2-PT, all layers | **67.2** | **84.3** |
| wav2vec2-FT, all layers | 63.8 | 68.7 |

- Global normalisation drops RAVDESS to 75.7 (dense). eGeMAPS fusion → 77.5.
- With a **random 5-fold split** on RAVDESS (setup of an earlier paper) they report 84.1 — note that this setup is not speaker-independent.

## Limitations
- Only base model tested; inputs truncated.
- Speaker normalisation assumes many test clips per speaker are available (not realistic for a single new user).
- Test set on RAVDESS is only 3 actors → high variance.
- Authors' code repo warns models are **very unlikely to generalise** beyond clean, acted training data.

## Relevance to Team Lucid
- Strongest evidence that **pretrained embeddings beat eGeMAPS** within a dataset.
- ASR-fine-tuned model is worse → emotion cues (e.g., pitch) get discarded by ASR training.
- eGeMAPS fusion helps a little → handcrafted prosody still carries complementary information.
- Raises our question: does the embedding advantage **survive cross-corpus** (RAVDESS ↔ CREMA-D)?

## Viva-ready points
- "Why not just use wav2vec2?" → Higher in-dataset accuracy but 768+ uninterpretable dimensions; generalisation unverified.
- "What is speaker normalisation and why is it a caveat?" → Uses test-speaker statistics; inflates results if unrealistic for deployment.
