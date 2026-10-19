# Emotion Recognition Using Explainable Machine Learning

**Team Lucid** · Speech Emotion Recognition (SER) with Explainable AI

| Member | Roll number |
|---|---|
| Pranav Sajith | BL.SC.U4CSE24039 |
| Piriya Shreyash Shanmukh | BL.SC.U4CSE24037 |
| Praveen Raj | BL.SC.U4CSE24040 |

---

## About the project
We are building a research-driven Speech Emotion Recognition system. It classifies emotion from voice recordings and **explains** which acoustic properties drove each prediction. We follow a literature-first workflow, so the research gap and methodology are chosen from evidence before any code is written.

**Research question:** When a speech emotion recognition model trained on one acted English corpus is applied to another, do its accuracy and its SHAP explanations hold up, and can explanation stability be used to choose features that transfer better?

**Workflow:** Literature → Research gap → Methodology → Data → Features → Models → Explainability → Experiments → Simulation → Report/Viva

## Current status
| Phase | Status |
|---|---|
| 1. Literature survey | ✅ Notes, matrix and synthesis complete |
| 2. Research gap & methodology | ✅ Direction approved — see [`report/02_research_gap_and_methodology.md`](report/02_research_gap_and_methodology.md) |
| 3. Dataset investigation | ✅ [`report/03_dataset_investigation.md`](report/03_dataset_investigation.md) |
| 4. Environment & repository | ✅ Set up |
| 5. Audio preprocessing & features | ✅ [`report/05_preprocessing_and_features.md`](report/05_preprocessing_and_features.md) |
| 6. Modelling | ✅ [`report/06_modelling.md`](report/06_modelling.md) |
| 7. Explainable AI | ✅ [`report/07_explainability.md`](report/07_explainability.md) |
| 8. Experiments & error analysis | ✅ [`report/08_experiments.md`](report/08_experiments.md) |
| 9. Interactive simulation | ✅ Streamlit app — see [`report/09_simulation.md`](report/09_simulation.md) |
| 10. Report & viva | 🔄 Drafted — [`report/10_final_report.md`](report/10_final_report.md), viva prep, assessment plans, team plan; team review pending |

## Literature survey
- [`literature/literature_matrix.md`](literature/literature_matrix.md): comparison of 10 verified, open-access papers
- [`literature/synthesis.md`](literature/synthesis.md): common methods, weaknesses and candidate research gaps
- [`literature/notes/`](literature/notes/): detailed notes on each paper

## Datasets (not included in this repository)
| Dataset | Speakers | Clips | Licence | Source |
|---|---|---|---|---|
| RAVDESS (speech, audio-only) | 24 | 1,440 | CC BY-NC-SA 4.0 | https://zenodo.org/records/1188976 |
| CREMA-D | 91 | 7,442 | Open Database License | https://github.com/CheyneyComputerScience/CREMA-D |

Audio files are **not** committed. Download instructions: [`data/README.md`](data/README.md).

## Repository structure
```
literature/     literature matrix, synthesis, per-paper notes
notebooks/      numbered Jupyter notebooks (exploration & experiments)
src/            reusable code: preprocessing, features, models, XAI, evaluation
data/           metadata only (raw audio stays local, git-ignored)
experiments/    experiment configurations and logs
results/        figures and result tables
app/            interactive demo application
report/         report and presentation material
```

## Setup (macOS)
```bash
git clone https://github.com/shreyashsp25/lucid-speech-emotion-xai.git
cd lucid-speech-emotion-xai
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Principles
- Speaker-independent evaluation (no speaker in both training and test data)
- No fabricated results, citations or novelty claims
- Explanations are evaluated, not just plotted

## Run the demo app

```bash
source .venv/bin/activate
pip install streamlit==1.66.0 plotly==7.1.0     # once
python -m src.app_assets                         # once: held-out demo model (~3–5 min)
streamlit run app/streamlit_app.py
```
Needs the trained model from `notebooks/07_final_model.ipynb` and the local data (`data/raw`, `data/processed`).
