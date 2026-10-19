# Phase 4 — Environment and Repository

Team Lucid · Set up during Phase 1 and extended as each phase needed new tools.

---

## 1. Tools and why we use them

| Tool | Version | Why |
|---|---|---|
| macOS + Homebrew | — | Package manager to install Python, Git and command-line tools |
| Python | **3.11** | Best-supported version for the audio/ML libraries (librosa, openSMILE, PyTorch); one fixed version means identical results for all members |
| Virtual environment (`.venv`) | — | Keeps project packages separate from the rest of the laptop; `requirements.txt` recreates it exactly |
| Jupyter (kernel *Python (lucid-ser)*) | — | Interactive exploration and experiments; notebooks are numbered in run order |
| VS Code | — | Editing notebooks and modules; Markdown preview for reports |
| Git + GitHub CLI (`gh`) | Git 2.56 | Version control; private repo `shreyashsp25/lucid-speech-emotion-xai` |
| Git LFS | — | Needed only to download CREMA-D audio |

## 2. Packages, added only when the method required them

| Phase | Packages | Purpose |
|---|---|---|
| 1 | `jupyter`, `ipykernel`, `pandas` | Notebooks, tables |
| 3 | `soundfile`, `scikit-learn`, `matplotlib` | Read audio headers, grouped CV folds, plots |
| 5 | `librosa`, `opensmile`, `torch`, `transformers`, `tqdm` | Resampling/trimming, eGeMAPS, WavLM embeddings, progress bars |
| 6–7 | `lightgbm`, `shap` | Gradient boosting (decision recorded in Phase 2), explanations |
| 9 | `streamlit`, `plotly` | Interactive demo |

Exact versions are pinned in `requirements.txt` (`pip freeze`).

## 3. Repository layout
```
literature/   notes on 10 papers, matrix, synthesis
report/       phase write-ups (this file is 04)
notebooks/    01_dataset_investigation, 02_feature_extraction, ...
src/          data.py (metadata, folds), audio.py (preprocessing), features.py (eGeMAPS, WavLM),
              models.py, evaluation.py, explain.py, experiments.py, predict.py, app_utils.py, app_assets.py
data/         README.md (download steps), metadata/ (labels + folds, committed)
              raw/ and processed/ (audio and features, git-ignored)
results/      summaries (.json), per-phase tables (phase6–9/) and figures (.png)
models/       trained models (git-ignored)
app/          Streamlit demo (Phase 9); settings in .streamlit/config.toml
```
Rule: **notebooks explore, `src/` holds reusable code.** Anything used by more than one notebook moves into `src/`.

## 4. What is never committed (`.gitignore`)
`.venv/`, audio files (`*.wav`, `*.mp3`, `*.zip`), `data/raw/`, `data/processed/`, paper PDFs, trained model files, `.env` and other secrets. Dataset licences (RAVDESS CC BY-NC-SA 4.0, CREMA-D ODbL) are respected by sharing download instructions instead of audio.

## 5. Reproducibility practices
- Fixed random seeds (`seed=42` for the speaker folds; `seed=0` for models, bootstraps and resampling), so every run reproduces the same numbers.
- Speaker folds saved in `data/metadata/*.csv`, so every experiment uses identical splits.
- Feature files are regenerated from code; already-extracted files are skipped.
- Commit messages follow a type prefix: `docs:`, `feat:`, `fix:`, `chore:`.

## 6. Setting up a team member's laptop
```bash
brew install python@3.11 git git-lfs gh
git lfs install
gh auth login
gh repo clone shreyashsp25/lucid-speech-emotion-xai
cd lucid-speech-emotion-xai
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name lucid-ser --display-name "Python (lucid-ser)"
```
Then download the data (`data/README.md`) and run the notebooks in order.

**Daily routine:** `cd` into the project → `source .venv/bin/activate` → `git pull` → work → `git add` / `commit` / `push`.
