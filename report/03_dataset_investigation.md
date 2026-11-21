# Phase 3 — Dataset Investigation

Team Lucid · All numbers below were measured from the downloaded files by `notebooks/01_dataset_investigation.ipynb` (19 Oct 2026). Raw summary: `results/phase3_dataset_summary.json`.

---

## 1. Overview

| | RAVDESS (speech, audio-only) | CREMA-D (AudioWAV) |
|---|---|---|
| Files | **1,440** (all readable) | **7,442** (all readable) |
| Speakers | 24 (12 female, 12 male) | 91 (43 female, 48 male) |
| Clips per speaker | exactly 60 | 76–82 (median 82) |
| Sentences | 2 | 12 |
| Emotions | 8 | 6 |
| Sample rate | 48 kHz | 16 kHz |
| Channels | 1,435 mono, **5 stereo** | all mono |
| Bit depth | 16-bit PCM | 16-bit PCM |
| Duration (s) | mean 3.70, range 2.94–5.27 | mean 2.54, range 1.27–5.00 |
| Clips < 1 s | 0 | 0 |
| Duplicates | 0 | 0 |
| Licence | CC BY-NC-SA 4.0 | ODbL 1.0 |

Both corpora match their published file counts, and no files are corrupt.

## 2. Things the papers didn't tell us (and what we'll do)

| Finding | Why it matters | Action (Phase 5) |
|---|---|---|
| **Sample rates differ** (48 kHz vs 16 kHz) | Features would differ just from the recording format, which is a corpus confound | Resample everything to **16 kHz** (also required by WavLM) |
| **5 RAVDESS files are stereo**: `03-01-02-01-01-02-01`, `03-01-08-01-02-02-01` (actor 01), `03-01-02-01-02-02-05` (actor 05), `03-01-03-01-02-01-20`, `03-01-06-01-01-02-20` (actor 20) | Inconsistent input shape | Convert to mono by averaging channels |
| **RAVDESS clips are ~1.2 s longer** on average | Duration-related features (e.g., eGeMAPS temporal features) could separate corpora rather than emotions | Trim leading/trailing silence (anything 30 dB below the loudest point counts as silence); compare durations after trimming; watch for duration features in SHAP |

## 3. Emotion labels and harmonisation

| Harmonised label | RAVDESS | CREMA-D |
|---|---|---|
| neutral | **96** | 1,087 |
| happy | 192 | 1,271 |
| sad | 192 | 1,271 |
| angry | 192 | 1,271 |
| fearful | 192 | 1,271 |
| disgust | 192 | 1,271 |
| **Total used** | **1,056** | **7,442** |
| Dropped | calm 192, surprised 192 | — |

- RAVDESS neutral has half the clips because neutral was recorded at only one intensity. This is why **UAR** is our main metric.
- CREMA-D neutral is smaller because neutral was not recorded at the low/medium/high levels of sentence 1.
- CREMA-D intensity: 455 clips each at low / medium / high, and 6,077 "unspecified".

## 4. CREMA-D label quality (crowd agreement)

The audio-only majority vote from `summaryTable.csv` matches the intended emotion for **41.6%** of clips. That is consistent with the ~41% human audio-only accuracy reported in the CREMA-D paper (P2).

| Intended emotion | Listeners agreed |
|---|---|
| neutral | 95.7% |
| angry | 60.6% |
| fearful | 32.0% |
| disgust | 27.0% |
| happy | 26.0% |
| **sad** | **16.4%** |

- Listeners chose **"neutral" for 3,897 of 7,442 clips**. Much intended emotion is perceived as neutral when only the voice is heard.
- 644 clips have tied votes (e.g., `N:S`). Our rule counts agreement only when the single majority letter equals the intended emotion, so tied clips count as "not agreed". This is a strict, conservative rule.
- **High-agreement subset** for the robustness check: **3,099 clips** (neutral 1,040, angry 770, fearful 407, disgust 343, happy 330, sad 209). This subset is heavily imbalanced, so results on it must be read through UAR and per-class recall.

**Implication:** a CREMA-D model that "fails" on sad or happy may be failing where human listeners also fail. Error analysis (E5) will compare model confusions with listener confusions.

## 5. Speaker-independent folds

Created with `StratifiedGroupKFold` (5 folds, seed 42), grouped by speaker and stratified by **emotion and gender**.

- **No speaker appears in more than one fold**: checked in code for both corpora.
- RAVDESS: 4–5 speakers per fold, gender split 2–3 / 2–3 in every fold.
- CREMA-D: 18–19 speakers per fold, gender split 8–9 / 9–10.
- Class proportions are near-identical across folds.
- Fold assignments are saved in `data/metadata/*.csv` (`fold` column), so **every experiment uses the same folds**.

Cross-corpus experiments (E2) use **all** harmonised clips of one corpus for training and all of the other for testing. No fold split is needed there, because the speakers are entirely different people.

## 6. Ethics and licensing
- Both datasets use consenting professional actors. Labels are *acted* emotions, not real emotional states, which limits real-world claims.
- RAVDESS is non-commercial (CC BY-NC-SA), so our demo app is for academic use only.
- CREMA-D demographics: 61 Caucasian, 22 African American, 7 Asian, 1 unknown; ages 20–74. RAVDESS has only young adults (21–33) with a neutral North American accent. This demographic mismatch is part of the cross-corpus shift we study.
- No audio is redistributed in our repository; only metadata and download instructions.

## 7. Limitations noted
- Acted, clean, English-only speech.
- RAVDESS's small speaker count means wide confidence intervals per fold.
- CREMA-D intended labels are often not what listeners perceive (Section 4).
