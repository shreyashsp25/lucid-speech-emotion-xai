# Phase 5 — Audio Preprocessing and Feature Extraction

Team Lucid · Measured by `notebooks/02_feature_extraction.ipynb` (19 Oct 2026). Summary: `results/phase5_feature_summary.json`.

---

## 1. Preprocessing (identical for both corpora) — `src/audio.py`

| Step | Setting | Reason |
|---|---|---|
| Mono | average channels | 5 RAVDESS files are stereo (Phase 3) |
| Resample | **16 kHz** (exact polyphase) | RAVDESS is 48 kHz, CREMA-D 16 kHz; removes a format confound; required by WavLM |
| Trim edge silence | `top_db = 30`, never below 0.25 s | RAVDESS clips carried extra leading/trailing silence |
| Amplitude normalisation | **none** | Loudness is an emotion cue; corpus gain differences are handled by feature standardisation in Phase 6 |

**Effect on duration (mean):**

| | Original | After trim |
|---|---|---|
| RAVDESS | 3.70 s | **1.90 s** |
| CREMA-D | 2.54 s | **2.39 s** |

Trimming removed ~1.8 s of silence from RAVDESS and only ~0.15 s from CREMA-D. RAVDESS clips are now *shorter* than CREMA-D's (see `results/figures/p5_trimmed_durations.png`), because RAVDESS sentences are short and studio-quiet. **Duration-related features still differ between corpora**, so in Phases 7–8 we will check whether temporal eGeMAPS features (speech rate, segment lengths) act as corpus shortcuts.

## 2. Features — `src/features.py`

| Set | Shape per corpus | Tool | Check |
|---|---|---|---|
| **eGeMAPS v02** | RAVDESS 1,440 × 88 · CREMA-D 7,442 × 88 | openSMILE (Functionals) | 0 missing values, 0 constant columns, row order = metadata |
| **WavLM base-plus** (frozen, mean-pooled) | RAVDESS 1,440 × 13 × 768 · CREMA-D 7,442 × 13 × 768 | Hugging Face `microsoft/wavlm-base-plus`, Apple GPU (MPS) | all values finite, row order = metadata |

All 13 hidden layers are stored (float16). The layer used for classification is chosen in Phase 6 by cross-validation **inside the training data only**.

Runtime on a MacBook Air (Apple GPU): eGeMAPS ≈ 4 min, WavLM ≈ 10 min for all 8,882 clips.

## 3. Domain-shift check (PCA)

`results/figures/p5_domain_shift_pca.png` shows both corpora projected onto the top 2 principal components (features standardised on the pooled data).

| Feature set | PC1 | PC2 | What the plot shows |
|---|---|---|---|
| eGeMAPS | 29% | 7.5% | RAVDESS and CREMA-D separate along **PC2**: they overlap partly but form distinct bands |
| WavLM layer 6 | 23% | 6.7% | RAVDESS forms a **tight, separate cluster**; strong corpus separation |

**Interpretation:** both feature sets encode *which corpus* a clip came from, not only *which emotion*. The corpus separation looks stronger for WavLM. That is consistent with H1 (pretrained embeddings may transfer worse across corpora), but a PCA picture is not evidence; **H1 will be tested by the cross-corpus experiments in Phase 8**.

## 4. Data-quality finding: 39 CREMA-D clips with no detected pitch

For **39 CREMA-D clips** (0 in RAVDESS) openSMILE found no voiced frames: F0, HNR and voiced-segment rate are all 0, and loudness is about a third of the corpus average. They form the small isolated cluster in the eGeMAPS PCA plot.
- Mostly **sad** (19 of 36 in the cluster) and **disgust** (11): plausibly very soft or breathy productions.
- Spread across actors (e.g., 1075, 1014, 1034, 1050).
- Examples: `1006_IEO_SAD_LO.wav`, `1014_IEO_SAD_LO.wav`, `1001_ITH_SAD_XX.wav`.

**Decision:** keep them. They are genuine recordings, and removing them would remove mostly one emotion. They are flagged for error analysis (E5), and a sensitivity check without them will be reported. **Action for the team:** listen to 2–3 of these files to confirm they contain audible speech.

## 5. Files
- `src/audio.py`, `src/features.py`, `notebooks/02_feature_extraction.ipynb`
- `results/phase5_feature_summary.json`, `results/figures/p5_trimmed_durations.png`, `results/figures/p5_domain_shift_pca.png`
- Local only (git-ignored, regenerate with the notebook): `data/processed/egemaps_*.csv`, `data/processed/wavlm_*.npy`
