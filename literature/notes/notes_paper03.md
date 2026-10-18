# Paper 03 — GeMAPS / eGeMAPS

**Citation:** Eyben, F., Scherer, K. R., Schuller, B. W., Sundberg, J., André, E., Busso, C., Devillers, L. Y., Epps, J., Laukka, P., Narayanan, S. S., & Truong, K. P. (2016). The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing. *IEEE Transactions on Affective Computing*, 7(2), 190–202. https://doi.org/10.1109/TAFFC.2015.2457417

**Free PDF:** https://opus.bibliothek.uni-augsburg.de/opus4/files/38554/0214.pdf (University of Augsburg repository, CC-BY 4.0)
**Type:** Feature-standard paper

---

## What they did (plain language)
Experts from psychology, phonetics and engineering agreed on a small, standard list of voice measurements that are known to change with emotion (pitch, loudness, voice quality, spectral balance, timing). They implemented it in the openSMILE toolkit and showed that this small set performs nearly as well as feature sets with thousands of features.

## Feature set
- **GeMAPS: 62 parameters** from 18 low-level descriptors (LLDs):
  - Frequency: pitch (F0, semitones), jitter, formants F1–F3, F1 bandwidth
  - Energy/amplitude: shimmer, loudness, harmonics-to-noise ratio (HNR)
  - Spectral balance: alpha ratio, Hammarberg index, spectral slopes, formant relative energies, H1–H2, H1–A3
  - Temporal: loudness-peak rate, voiced/unvoiced segment lengths, pseudo-syllable rate
- **eGeMAPS (extended): 88 parameters** — adds MFCC 1–4, spectral flux, F2–F3 bandwidths, equivalent sound level.
- Each parameter has a physical meaning → **interpretable by design**.

## Datasets
6 corpora: FAU AIBO, TUM-AVIC, EMO-DB, GEMEP, GeSiE (singing), VAM. (Not RAVDESS or CREMA-D.)

## Preprocessing
Audio scaled to [−1, +1]; frames of 20 ms or 60 ms with 10 ms hop. Speaker-wise z-normalisation used for the main results.

## Models
Linear SVM (WEKA SMO), results averaged over a range of complexity values C.

## XAI method
None formal — interpretability comes from the feature design itself.

## Evaluation protocol
**Leave-One-Speaker(-group)-Out (LOSO)**, 8 speaker groups → speaker-independent. Training data up-sampled to balance classes. Metric: UAR.

## Reported results
- Binary arousal, average over 5 corpora: **eGeMAPS 79.7% UAR**, GeMAPS 79.6%, ComParE (6,373 features) 78.0%.
- Binary valence: eGeMAPS 66.4%, ComParE 67.2%.
- Per-corpus categorical tasks: large sets usually win, but eGeMAPS uses **<2% of the features**.
- eGeMAPS ≥ GeMAPS always → MFCC and spectral flux add value, especially for valence.

## Limitations
- Only binary arousal/valence evaluated in summary tables.
- Only parameters reliably extractable in clean audio were included.
- **Authors explicitly leave cross-database generalisation untested**: "it should be investigated whether the proposed minimalistic sets are able to obtain better generalisation in cross-database classification experiments."

## Relevance to Team Lucid
- eGeMAPS is our natural **interpretable feature baseline** (88 named features → readable SHAP plots).
- Section 2 argues that small feature sets may **generalise better across corpora** than huge ones — but the paper does not test this. That open question is directly testable with RAVDESS ↔ CREMA-D.
- Its LOSO protocol is the evaluation standard we should follow.

## Viva-ready points
- "Why eGeMAPS instead of 1000s of features?" → Interpretable, standardised, less over-fitting risk, near-equal performance.
- "Why not only eGeMAPS?" → Valence results lag large sets and pretrained embeddings (see Papers 04, 05).
