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

**Workflow:** Literature → Research gap → Methodology → Data → Features → Models → Explainability → Experiments → Simulation → Report/Viva

## Current status
| Phase | Status |
|---|---|
| 1. Literature survey | ✅ Notes, matrix and synthesis complete (team review in progress) |
| 2. Research gap & methodology | ⏳ Next |
| 3. Dataset investigation | ⬜ |
| 4. Environment & repository | ✅ Set up |
| 5. Audio preprocessing & features | ⬜ |
| 6. Modelling | ⬜ |
| 7. Explainable AI | ⬜ |
| 8. Experiments & error analysis | ⬜ |
| 9. Interactive simulation | ⬜ |
| 10. Report & viva | ⬜ |

## Literature survey
- [`literature/literature_matrix.md`](literature/literature_matrix.md): comparison of 10 verified, open-access papers
- [`literature/synthesis.md`](literature/synthesis.md): common methods, weaknesses and candidate research gaps
- [`literature/notes/`](literature/notes/): detailed notes on each paper

## Datasets (not included in this repository)
| Dataset | Speakers | Clips | Licence | Source |
|---|---|---|---|---|
| RAVDESS (speech, audio-only) | 24 | 1,440 | CC BY-NC-SA 4.0 | https://zenodo.org/records/1188976 |
| CREMA-D | 91 | 7,442 | Open Database License | https://github.com/CheyneyComputerScience/CREMA-D |

Audio files are **not** committed. Download instructions will be added in Phase 3.

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
