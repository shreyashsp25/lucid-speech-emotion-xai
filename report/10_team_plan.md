# Team Plan — Ownership, Shared Understanding and Final Checklist

Team Lucid · Each member **owns** some parts (presents them, answers first in the viva), but **everyone must be able to explain the whole system**: the examiner may ask anyone anything.

---

## 1. Ownership

| Member | Owns | Code and files | Papers to know best |
|---|---|---|---|
| **Pranav Sajith** | Data, evaluation protocol, modelling results | `src/data.py`, `src/models.py`, `src/evaluation.py`; notebooks 01, 03, 06; reports 03, 06; E0, E1, E6 | P1 RAVDESS, P2 CREMA-D, P7 EMO-SUPERB, P8 cross-corpus SSL |
| **Piriya Shreyash Shanmukh** | Preprocessing, features, explainability, stability (H2) | `src/audio.py`, `src/features.py`, `src/explain.py`; notebooks 02, 04; reports 05, 07; E3 | P3 eGeMAPS, P4 wav2vec 2.0, P5 explaining embeddings |
| **Praveen Raj** | Cross-corpus experiments, feature selection, error analysis, app | `src/experiments.py`, `src/predict.py`, `src/app_*.py`, `app/streamlit_app.py`; notebooks 05, 07; reports 08, 09; E2, E4, E5 | P6 reality check, P9 XAI for SER, P10 SHAP multiplicity |

Presentation split: see `10_initial_assessment.md` and `10_final_presentation.md` (each slide has a presenter). Viva: each question in `10_viva_preparation.md` has a lead.

## 2. Making sure everyone understands everything

| Session (≈1 h each) | Led by | The other two must be able to… |
|---|---|---|
| 1. Data → folds → leakage | Pranav | explain UAR, speaker-independent folds, why random splits inflate RAVDESS more |
| 2. Audio → eGeMAPS / WavLM → SHAP | Piriya | explain one eGeMAPS feature from each group, why middle WavLM layers, how linear SHAP is computed, the clipping fix |
| 3. Cross-corpus → stability → app | Praveen | explain H1's two directions, per-corpus normalisation, H2 numbers, why H3 failed, how the app avoids misleading users |
| 4. Mock viva | rotate | answer 10 random questions from **another member's** section without notes |

**Rule for the mock viva:** pick questions with a random number generator; the person answering must not be the owner. Repeat until everyone can answer the "Quick-fire numbers" table in `10_viva_preparation.md` from memory.

Each member should also run the full demo once on their own (see §3).

## 3. Setting up the teammates' laptops (pending item)

On each teammate's Mac:
```bash
git clone https://github.com/shreyashsp25/lucid-speech-emotion-xai.git
cd lucid-speech-emotion-xai
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Then download the data (`data/README.md`) and run notebooks 01 → 02 → 03 → 07 in order, then `python -m src.app_assets` and `streamlit run app/streamlit_app.py`.
**Faster alternative:** copy `data/processed/`, `data/raw/` and `models/` from the Mac where the experiments were run, with a USB drive or AirDrop (never via GitHub). The app then runs immediately.

If the repository is private, add teammates (and later the faculty) as collaborators: GitHub → repository → Settings → Collaborators → Add people.

## 4. Final submission checklist

| Item | Status |
|---|---|
| Final report `report/10_final_report.md` | written; **team to read and correct** |
| References verified (full authors, titles, venues; numbered 1–25) | done |
| Venue of Paper 6 confirmed (ICASSP 2023) | done |
| Try once more to access the Springer 2026 SHAP/permutation feature-selection article; if read, update report §6.1 | **to do** |
| Phase reports 02–09 | done |
| Notebooks 01–07 run top to bottom with outputs | done |
| App runs on at least two laptops | **to do** (§3) |
| Final presentation `Lucid_Final_Presentation.pptx` (22 slides, speaker notes) | done; **rehearse once with timing**; initial-assessment deck optional |
| Learning guide *Lucid from zero* (PDF in `report/`, local only; replaces the earlier Word guides) | done; every member reads it |
| All 25 reference PDFs collected in `Machine learning/papers/` (01–25) | done |
| Backup demo `.wav` file on the desktop | **to do** |
| Mock viva held (§2) | **to do** |
| Repository shared with the faculty (collaborator or public after grading) | **to do** |
| Nothing secret or large in Git (no audio, features, models, `.env`) | checked |
