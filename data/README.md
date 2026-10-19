# Data — download instructions

Audio files are **not** stored in this repository (licences and size). Each team member downloads them locally into `data/raw/`, which is git-ignored.

Run all commands from the project root (`lucid-speech-emotion-xai/`).

---

## 1. RAVDESS — speech, audio-only (215 MB)

- Source: https://zenodo.org/records/1188976 (file `Audio_Speech_Actors_01-24.zip`)
- Licence: **CC BY-NC-SA 4.0** (non-commercial, share-alike; cite Livingstone & Russo, 2018)

```bash
mkdir -p data/raw/ravdess
curl -L -C - -o data/raw/ravdess.zip \
  "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
unzip -q data/raw/ravdess.zip -d data/raw/ravdess
rm data/raw/ravdess.zip
find data/raw/ravdess -name "*.wav" | wc -l      # expect 1440
```

If `curl` fails (some college networks block Zenodo), download the zip in a browser from the Zenodo page, move it to `data/raw/ravdess.zip`, and run the `unzip` line.

Expected layout: `data/raw/ravdess/Actor_01/03-01-01-01-01-01-01.wav … Actor_24/…`

---

## 2. CREMA-D — audio WAV only (~0.5 GB)

- Source: https://github.com/CheyneyComputerScience/CREMA-D
- Licence: **Open Database License (ODbL 1.0)**; cite Cao et al., 2014
- The repo stores audio with **Git LFS**. A plain zip download gives empty placeholder files, so use the commands below. They download only the WAV folder, not the full 7.5 GB with videos.

```bash
brew install git-lfs            # once per laptop
git lfs install                 # once per laptop

cd data/raw
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/CheyneyComputerScience/CREMA-D.git cremad
cd cremad
git lfs pull --include="AudioWAV/*,VideoDemographics.csv,processedResults/summaryTable.csv"
cd ../../..
find data/raw/cremad/AudioWAV -name "*.wav" | wc -l   # expect 7442
```

If `git lfs pull` reports a **bandwidth / quota error**, stop and tell the team. We will choose an alternative mirror and document it here.

Expected layout: `data/raw/cremad/AudioWAV/1001_DFA_ANG_XX.wav …`, plus `data/raw/cremad/VideoDemographics.csv` and `data/raw/cremad/processedResults/summaryTable.csv`.

---

## 3. Build the metadata

Open `notebooks/01_dataset_investigation.ipynb` with the **Python (lucid-ser)** kernel and choose **Restart & Run All**. It creates:

| File | Contents | In git? |
|---|---|---|
| `data/metadata/ravdess.csv` | one row per clip: speaker, gender, emotion, harmonised label, sample rate, duration, fold | yes |
| `data/metadata/cremad.csv` | the same, plus age/race and listener-agreement columns | yes |
| `results/phase3_dataset_summary.json` | counts and checks used in the report | yes |
| `results/figures/p3_*.png` | distribution plots | yes |

The metadata tables contain no audio, only labels and file paths, so they are safe to commit and share.
