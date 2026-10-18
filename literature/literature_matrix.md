# Literature Matrix — Team Lucid

Speech Emotion Recognition with Explainable ML · Phase 1
"Not Reported" = the paper does not state it. "TO VERIFY" = check against the PDF's first page.
Detailed notes per paper: `literature/notes/notes_paperNN.md`.

---

## A. Bibliographic details

| # | Title (short) | Authors | Year | Venue | Free PDF | Reader |
|---|---|---|---|---|---|---|
| 1 | RAVDESS | Livingstone, Russo | 2018 | PLOS ONE 13(5) e0196391 | [PLOS](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391) | Pranav |
| 2 | CREMA-D | Cao, Cooper, Keutmann, Gur, Nenkova, Verma | 2014 | IEEE TAC 5(4) 377–390 | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4313618) | Pranav |
| 3 | GeMAPS / eGeMAPS | Eyben, Scherer, Schuller, Sundberg, André, Busso, Devillers, Epps, Laukka, Narayanan, Truong | 2016 | IEEE TAC 7(2) 190–202 | [Augsburg](https://opus.bibliothek.uni-augsburg.de/opus4/files/38554/0214.pdf) | Piriya |
| 4 | wav2vec 2.0 embeddings for SER | Pepino, Riera, Ferrer | 2021 | Interspeech 2021 | [arXiv](https://arxiv.org/abs/2104.03502) | Piriya |
| 5 | Explaining DL embeddings via acoustic features | Dixit, Low, Elbanna, Catania, Ghosh | 2024 | arXiv 2409.09511 | [arXiv](https://arxiv.org/abs/2409.09511) | Piriya |
| 6 | SER reality check (IEMOCAP) | Antoniou, Katsamanis, Giannakopoulos, Narayanan | 2023 | arXiv 2304.00860 (venue TO VERIFY) | [arXiv](https://arxiv.org/pdf/2304.00860) | Praveen |
| 7 | EMO-SUPERB | Wu, Chou, Chang, Goncalves, Du, Jang, Lee, Lee | 2024 | arXiv 2402.13018 (venue TO VERIFY) | [arXiv](https://arxiv.org/pdf/2402.13018) | Pranav |
| 8 | Cross-corpus training with SSL | Pastor, Ribas, Ortega, Miguel, Lleida | 2023 | Applied Sciences 13(16) 9062 | [MDPI](https://www.mdpi.com/2076-3417/13/16/9062) | Pranav |
| 9 | XAI techniques for SER | Norval, Wang | 2025 | Inteligencia Artificial 28(76) 85–123 | [Iberamia](https://journal.iberamia.org/index.php/intartif/article/view/1809) | Praveen |
| 10 | Explanation multiplicity in SHAP | Hwang, Lee, Rosenblatt, Whang, Stoyanovich | 2026 | arXiv 2601.12654 | [arXiv](https://arxiv.org/abs/2601.12654) | Praveen |

---

## B. Data, features and models

| # | Dataset(s) | Preprocessing | Features | Models |
|---|---|---|---|---|
| 1 | RAVDESS (creates it): 24 actors, 2 sentences, 8 emotions, 1,440 speech audio files, 48 kHz | Peak-normalised −3 dBFS | Not Reported (dataset paper) | Not Reported |
| 2 | CREMA-D (creates it): 91 actors, 12 sentences, 6 emotions, 7,442 clips | Not Reported | Not Reported (dataset paper) | Not Reported |
| 3 | FAU AIBO, TUM-AVIC, EMO-DB, GEMEP, GeSiE, VAM | [−1,1] scaling; speaker z-norm; class up-sampling | **GeMAPS (62) / eGeMAPS (88)** via openSMILE | Linear SVM |
| 4 | IEMOCAP (4 cls), RAVDESS (7 cls, calm+neutral merged) | Truncation 8 s / 5 s; speaker normalisation | **wav2vec 2.0 base** (all 13 layers, learned weights); baselines eGeMAPS, spectrogram | Dense head; LSTM; eGeMAPS fusion |
| 5 | RAVDESS, SAVEE | Speaker normalisation | **eGeMAPSv02**; **WavLM Large** (1,024-d, mean-pooled) | Logistic Regression (binary per emotion); ridge probes |
| 6 | IEMOCAP (4 cls, 5,531 utt.) | — (re-runs others' code) | Various (others' models) | Various (others' models) |
| 7 | IEMOCAP, **CREMA-D**, MSP-IMPROV, MSP-PODCAST, BIIC-NNIME, BIIC-PODCAST | Fixed speaker-independent splits | 15 frozen SSL models (wav2vec2, HuBERT, WavLM, XLS-R…); FBANK | Lightweight head, soft labels |
| 8 | EmoDB, **RAVDESS**, **CREMA-D**, IEMOCAP (4 cls) | Not Reported | HuBERT base, WavLM base | Linear; CNN+attention pooling; CT-Transformer |
| 9 | Custom Afrikaans corpus (~800 clips, 8 emotions) | MFCC norm; pitch-shift / time-stretch augmentation | MFCC, chroma, mel, spectral contrast, tonnetz (flattened) | CNN, CLSTM, XGBoost, stacking |
| 10 | Tabular datasets (not speech) | — | — | Several model classes |

---

## C. Explainability, evaluation and results

| # | XAI method | Evaluation protocol | Reported results (with context) |
|---|---|---|---|
| 1 | Not Reported | Human perceptual validation (10 ratings/file) | Human audio-only speech accuracy **63%**; AV 81% |
| 2 | Not Reported | Human crowd rating (~9.8/clip) | Human audio-only accuracy **40.9%**; AV 63.6%; Krippendorff α 0.42 |
| 3 | None (interpretable features) | **Speaker-independent LOSO** (8 groups); UAR | Binary arousal avg UAR: eGeMAPS **79.7%** vs ComParE 78.0%; valence 66.4% vs 67.2% |
| 4 | Layer-weight inspection only | **Actor split** on RAVDESS (19/2/3); session-out on IEMOCAP; UAR, 5 seeds | RAVDESS 7-cls UAR: wav2vec2-PT **84.3%** vs eGeMAPS 57.0%; IEMOCAP 67.2% vs 52.4% |
| 5 | **SHAP** + probing | **Speaker-independent 5-fold nested CV**; F1 | Binary emotion-vs-neutral F1 on RAVDESS: WavLM 83–99%, eGeMAPS 79–93%. Info ranking: Energy > Frequency > Spectral > Temporal |
| 6 | Not Reported | Proposes 10-fold leave-one-speaker-out; reproduces others | Reproduced WA 63.8–78.2%; dropping "happy" hides a large drop |
| 7 | Not Reported | Fixed speaker-independent splits; macro-F1; single run | All SSL > FBANK; XLS-R-1B best on average |
| 8 | Not Reported | 5-fold, speaker-disjoint; UAR | IEMOCAP UAR WavLM: 65.90 → 66.90 (+EmoDB+RAVDESS) → **65.06 with CREMA-D added** |
| 9 | **SHAP + LIME** | Random 20% split; speaker independence not stated | No per-model metrics for Afrikaans corpus; RAVDESS claim (UAR 89.4%) setup not described |
| 10 | SHAP (studied) | Dual-seed repeated runs; magnitude/rank/set metrics | SHAP rankings vary across runs/seeds; L2 hides top-feature churn |

---

## D. Limitations and relevance

| # | Key limitation | Relevance to Team Lucid |
|---|---|---|
| 1 | 24 speakers, 2 sentences, acted; calm/neutral confusion | Core dataset; forces speaker-independent splits; human ceiling 63% |
| 2 | Low rater agreement audio-only; intended ≠ perceived labels | Core dataset; 91 speakers → stable evaluation; human ceiling ~41% |
| 3 | Cross-database generalisation explicitly left untested | Interpretable baseline; its open question is testable with our two corpora |
| 4 | Speaker normalisation uses test-speaker stats; 3 test actors; generalisation doubted by authors | Embeddings ≫ eGeMAPS in-dataset — does it hold cross-corpus? |
| 5 | Within-dataset only; single SHAP run; binary tasks | Closest prior work; extension to CREMA-D, cross-corpus and stability is open |
| 6 | One dataset; no controlled SD-vs-SI experiment | Our evaluation rules (speaker-independent, keep all classes, UA+WA) |
| 7 | No explainability; RAVDESS absent; single runs | Reference speaker-independent CREMA-D setup; frozen-feature design is standard |
| 8 | Small, inconsistent gains; dataset properties not analysed | Naive merging of RAVDESS+CREMA-D can hurt → use train-on-one/test-on-other |
| 9 | Unreadable feature indices; unclear split; no stability check | Cautionary example: use named features and evaluate explanations |
| 10 | Not speech | Tools to measure explanation stability (rank metrics, dual seeds) |
