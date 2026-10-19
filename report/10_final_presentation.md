# Final Assessment (14 marks) — Presentation and Live Demo Plan

Team Lucid · Target: **15 minutes talk + 5 minutes live demo + questions**. Figures are in `results/figures/`.

Marking criteria covered: working system (demo), research contribution (slides 3, 12), real results (6–10), explainability (7–9), professional simulation (demo), comparisons (11), analysis (6–11), report (`10_final_report.md`), presentation, live demo.

---

## Slides

| # | Title | Show | Key message | Presenter |
|---|---|---|---|---|
| 1 | Title | project title, team | — | Pranav |
| 2 | Problem and question | 3 problems; research question; H0–H3 | Do accuracy and explanations survive a corpus change? | Pranav |
| 3 | What we did | pipeline diagram (report §18 / app About page) | Speaker-independent, nested CV, exact SHAP, 2 corpora, no KNN | Piriya |
| 4 | Data | dataset table; `p3_emotion_counts.png` | 1,056 + 7,442 clips; 6 emotions; CREMA-D listeners agree on 41.6% | Pranav |
| 5 | Leakage (H0) | E0 bar chart (app → Model performance) | Random splits inflate RAVDESS by up to 9.3 points | Pranav |
| 6 | Within-corpus results | `p6_within_corpus_uar.png`; confusion matrix | Fusion LogReg 82.2% / 74.1% UAR; WavLM +22–26 over eGeMAPS in every fold; fearful → sad is the main error | Pranav |
| 7 | Explainability | `p7_global_top15.png`; group shares | CREMA-D model uses loudness, RAVDESS model spectral shape; Energy most informative per feature; SHAP ≈ permutation (0.79 / 0.96) | Piriya |
| 8 | XAI as debugging | before/after table (±170 → ≈10 log-odds) | SHAP exposed exploding ratio features that accuracy hid | Piriya |
| 9 | Explanation stability (H2) | `p8_e3_stability.png` | 0.78 > 0.63 > 0.50 > 0.00; WavLM least stable; groups stable | Piriya |
| 10 | Cross-corpus (H1) | `p8_e2_cross_corpus.png` | −20 to −35 points; H1 holds from RAVDESS, reverses from CREMA-D: training diversity matters | Praveen |
| 11 | Stable-core selection (H3) + errors | E4 forest plot; neutral collapse (recall 0.09) | H3 inconclusive; neutral and sad break across corpora; human-hard = model-hard (ρ = 0.60) | Praveen |
| 12 | Standard evaluation | ROC-AUC 0.97 / 0.94; ECE 0.025 / 0.021; overfitting; learning curve | Well calibrated in-domain; still improving with more speakers | Pranav |
| 13 | Comparison with literature | report §15 table | Same embedding gap as P4; P5's ranking partly reproduced; P3's suggestion only one-way | Piriya |
| 14 | Limitations and ethics | report §20–21 | Two acted English corpora; over-confident under shift; not for decisions about people | Praveen |
| 15 | Conclusion | 3 bullet answer to the research question | Accuracy doesn't transfer well; explanations partly (best at group level); stability is a diagnostic | all |

---

## Live demo script (5 minutes, Praveen drives, others narrate)

**Before the session**
- `source .venv/bin/activate && streamlit run app/streamlit_app.py`, then open the app once and analyse one clip, so openSMILE and WavLM are already loaded.
- Close other apps; set the screen to 100% zoom; turn off notifications.
- Test the microphone in the browser (allow access).

**Steps**
1. **Analyse a recording → Dataset example → CREMA-D → angry.** Point out: speaker never heard by this model; true label shown; prediction and scores; trust panel (calibrated in-domain).
2. **Explanation tab.** The plain-language sentence; blue/orange bars; "explain the evidence for" → switch to *sad* to show why it is *not* sad; group chart; eGeMAPS vs WavLM evidence share.
3. **Signal tab.** Spectrogram and pitch contour: "this is exactly what the model receives".
4. **Robustness simulation.** Run it: discuss which conditions change the prediction (volume tests the loudness reliance found by SHAP).
5. **Record with microphone.** One sentence, angry voice, then sad voice. Point to the out-of-domain warning and the typicality check: "the app tells you when not to trust it".
6. **Cross-corpus & errors page.** H1 chart and the neutral collapse. Close with "accuracy within a dataset is not the whole story".

**If something fails**
- Microphone blocked → use **Upload a file** with a prepared `.wav` recorded earlier (keep one on the desktop).
- App crash or slow load → show the screenshots in `results/figures/` and the notebook outputs; restart with `Ctrl + C` and `streamlit run app/streamlit_app.py`.
- No internet is needed once WavLM has been downloaded once.

---

## Questions to expect after the demo
See `10_viva_preparation.md`, especially A10, B18, B20, C1, C5, C6 and C15.
