# Final Assessment (14 marks) — Presentation and Live Demo Plan

Team Lucid · Target: **15 minutes talk + 5 minutes live demo + questions**. Figures are in `results/figures/`.

Slides: [`Lucid_Final_Presentation.pptx`](Lucid_Final_Presentation.pptx) (23 slides, speaker notes on every slide). Each content slide shows its section and a progress bar.

Marking criteria covered: working system (demo, slides 19–20), research contribution (slides 5–6, 21), real results (9–17), explainability (12–13, 15), professional simulation (demo), comparisons (18), analysis (9–17), report (`Lucid_Final_Report.docx`), presentation, live demo.

---

## Slides

| # | Section | Title | Key message | Presenter |
|---|---|---|---|---|
| 1 | — | Title | Do accuracy and explanations survive a change of corpus? | Pranav |
| 2 | — | Roadmap of this talk | Five parts: basics, problem, method, results, demo & wrap-up | Pranav |
| 3 | 01 The basics | The task: hear a short clip, name the emotion | 6 emotions, chance 16.7%; emotion changes loudness, pitch, voice quality | Pranav |
| 4 | 01 The basics | Three ideas the rest of the talk relies on | Features (eGeMAPS vs WavLM), fair testing (speaker-independent, UAR), SHAP | Pranav |
| 5 | 02 The problem | Three problems with how SER is usually evaluated | Leakage, unknown transfer, unchecked explanations → research question | Pranav |
| 6 | 02 The problem | Four hypotheses, stated before the experiments | H0–H3 | Pranav |
| 7 | 03 Our method | Two acted English corpora, six shared emotions | 1,056 + 7,442 clips; CREMA-D listeners agree on 41.6% | Pranav |
| 8 | 03 Our method | Pipeline: leak-free from audio to explanation | Speaker-independent, nested CV, exact SHAP, no KNN | Piriya |
| 9 | 04 What we found | Results at a glance | Two supported, one direction-dependent, one inconclusive | Praveen |
| 10 | 04 What we found | H0: random splits inflate RAVDESS by up to 9 points | +9.3 / +7.8 on RAVDESS vs +1.5 / +1.7 on CREMA-D | Pranav |
| 11 | 04 What we found | Within a corpus, WavLM beats eGeMAPS by 22–26 points | Fusion LogReg 82.2% / 74.1% UAR; WavLM ahead in 5/5 folds | Pranav |
| 12 | 04 What we found | What the models listen to differs by corpus | CREMA-D = loudness, RAVDESS = spectral shape; Energy best per feature; SHAP ≈ permutation | Piriya |
| 13 | 04 What we found | SHAP found a bug that accuracy had hidden | ±170 → ≈10 log-odds after clipping; UAR barely changed | Piriya |
| 14 | 04 What we found | H1 depends on direction | −20 to −35 points; +6.1 from RAVDESS, −18.4 from CREMA-D; per-corpus norm +2 to +13 | Praveen |
| 15 | 04 What we found | H2 supported: explanations are only partly stable | 0.78 > 0.63 > 0.50 > 0.00; WavLM least stable; groups stable | Piriya |
| 16 | 04 What we found | H3 inconclusive; errors follow acting style and labels | 1 gain / 4 losses / 7 null; neutral recall 0.09; human-hard = model-hard | Praveen |
| 17 | 04 What we found | Standard checks | ROC-AUC 0.97 / 0.94; ECE 0.025 / 0.021; learning curve still rising | Pranav |
| 18 | 04 What we found | How our findings compare with published work | Same embedding gap as P4; P5 partly reproduced; P3 only one way | Piriya |
| 19 | 05 Demo & wrap-up | Live demo | Real model, SHAP and "why X and not Y?", what-if studio, two-dataset lens, stress test, trust notes, calibrate to your voice | Praveen |
| 20 | 05 Demo & wrap-up | Built for real voices: three tested fixes | Simulated-mic training +4 to +15 on new CREMA-D speakers (no studio loss); calibration 69 → 77%; corrections halve happy-called-angry; notes add the fourth fix, teach it your voice (78 → 85% on new RAVDESS speakers) | Pranav |
| 21 | 05 Demo & wrap-up | Limitations and ethics | Two acted corpora; not for decisions about people | Praveen |
| 22 | 05 Demo & wrap-up | Answer to the research question | Accuracy doesn't travel; explanations partly; stability is a diagnostic; XAI earned its place | all |
| 23 | 05 Demo & wrap-up | Thank you. Questions? | Future work | all |

---

## Live demo script (5 minutes, Praveen drives, others narrate)

**Before the session**
- `source .venv/bin/activate && streamlit run app/streamlit_app.py`, then open the app once and analyse one clip, so openSMILE and WavLM are already loaded.
- Close other apps; set the screen to 100% zoom; turn off notifications.
- Test the microphone in the browser (allow access).

**Steps**
1. **Analyse a voice → Dataset clip → CREMA-D → angry.** Point out the console: the emotion, the score ring, all six scores, the waveform; "matches the performed emotion"; the speaker was never heard by this model; the three trust notes (calibrated in-domain).
2. **Why tab.** The plain-language sentence; blue/orange bars; set **Compared with** to *happy*: "why angry and not happy?" (shared loudness cancels, what separates the two remains); group chart; eGeMAPS vs WavLM evidence share.
3. **What-if studio.** Lower *Loudness attack* and *Overall sound level*: watch the answer move. Pick *sad* under "smallest single change" and read the result aloud.
4. **Two-dataset lens.** Switch to RAVDESS → neutral: the RAVDESS-only, CREMA-D-only and pooled models often disagree. "This is our research question on one clip."
5. **Stress test**, then **Emotion compass.** Run the stress test; on the compass the modified copies appear as small dots, showing how far the answer drifts (volume tests the loudness reliance found by SHAP).
6. **Record.** Point to the *microphone-robust model* tag. Open **Step 1** and say the three calm sentences. Open **Step 2** and say one sentence in each of the six emotions ("teach it your voice"); do this before the viva and keep the browser tab open, because it lasts for the session. Then **Step 3**: one sentence, angry voice, then sad voice; compare with "without calibration or teaching". If it is wrong, use **Wrong answer?** to teach it and record again. Point to the trust notes: "the app tells you how far to trust it".
7. **New-dataset test page.** H1 chart and the neutral collapse. Close with "accuracy within a dataset is not the whole story".

**If something fails**
- Microphone blocked → use **Upload a file** with a prepared `.wav` recorded earlier (keep one on the desktop).
- App crash or slow load → show the screenshots in `results/figures/` and the notebook outputs; restart with `Ctrl + C` and `streamlit run app/streamlit_app.py`.
- No internet is needed once WavLM has been downloaded once.

---

## Questions to expect after the demo
See `10_viva_preparation.md`, especially A10, B18, B20, C1, C5, C6 and C15.
