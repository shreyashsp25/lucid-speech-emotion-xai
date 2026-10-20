# Phase 9 — Interactive simulation (research demo app)

Team Lucid · `app/streamlit_app.py` (Streamlit + Plotly) · helpers in `src/app_utils.py`, `src/studio.py`, `src/app_assets.py`, `src/predict.py`.

## 1. How to run

```bash
source .venv/bin/activate
python -m src.app_assets                         # once, ~3–5 min: builds the held-out demo model
streamlit run app/streamlit_app.py               # opens http://localhost:8501
```

Needs `models/final_models.joblib` (from `notebooks/07_final_model.ipynb`), the cached features in `data/processed/` and the audio in `data/raw/` (for dataset examples). The first analysis takes about 20 s while openSMILE and WavLM load; after that each clip takes a few seconds.

## 2. What the app contains

**Look and navigation.** The app has its own visual identity ("Lucid voice emotion lab"): navigation across the top, a dark "console" panel that shows the predicted emotion in large type with a score ring, all six scores as bars, and the clip's own waveform trace; Bricolage Grotesque headings with Instrument Sans text; the same emotion colours as the report and slides.

| Page | Content | Source |
|---|---|---|
| **Analyse a voice** | Input: dataset clip (held-out speaker), file upload (WAV/FLAC/OGG/MP3) or microphone. Console with prediction, score ring, six score bars and waveform; three trust notes (recording conditions and calibration, typicality, second opinion); eight tabs (below) | live, real models |
| · Why | Exact linear SHAP of the eGeMAPS model for any chosen emotion ("why / why not"); plain-language summary; net evidence per feature group; share of the fusion model's evidence from eGeMAPS vs WavLM | live |
| · Emotion compass | The six scores placed on a pleasantness × energy circle; the clip's score-weighted position; after a stress test, every modified copy is added so the drift is visible | live |
| · What-if studio | Eight sliders (loudness, sound level, loudness attack, pitch, pitch movement, HNR, alpha ratio, speech rate) shift features in training standard deviations; the eGeMAPS model is recomputed exactly; "smallest single change" search that would turn the clip into another emotion | live, exact (linear model) |
| · Two-dataset lens | The same clip judged by three eGeMAPS models trained on RAVDESS only, CREMA-D only and both (fold-0 speakers always held out), each with its top three reasons: the Phase 8 cross-corpus effect on one clip | live, trained once per session |
| · Timeline | eGeMAPS model on 1-second windows every 0.25 s, drawn as stacked emotion scores above the energy curve | live |
| · Voice profile | Radar of eight named features as percentiles among the 8,498 training clips, against the median profile of any emotion; all 88 features on request | live |
| · Signal | Waveform + energy, STFT spectrogram, 64-band Mel-spectrogram, pitch contour (pYIN); duration, median pitch, pitch range, voiced share | live |
| · Stress test | Re-runs the whole pipeline on 6 modified copies (−12 / +6 dB, ±2 semitones, noise at 20 / 10 dB SNR) and shows the scores as a heat map | live |
| **How it decides** | Global SHAP top-15 per corpus and model; group shares (total and per feature); SHAP vs permutation agreement; RAVDESS-vs-CREMA-D importance scatter; per-emotion drivers; no-pitch shortcut check | Phase 7 results |
| **Performance** | All 12 model × feature results with fold SD; leakage experiment; interactive confusion matrices (out-of-fold); ROC-AUC, calibration, overfitting, learning curve; final-recipe held-out check | Phases 6, 8 (E6), 9 |
| **New-dataset test** | Cross-corpus UAR with bootstrap CIs and in-domain reference; H1 table; explanation stability (H2) box plots; stable-core selection (H3) forest plot; per-class drops, gender/age, label quality | Phase 8 |
| **About** | Pipeline diagram (reads the WavLM layer from the saved model), design choices, the interactive tools, limitations, ethics | — |

Every chart is drawn from the trained models or the saved result files; nothing is hard-coded or simulated.

**Honesty notes for the new tools.** The compass positions are a readable layout loosely following the valence–arousal arrangement used in psychology, not a measurement. The what-if studio moves one feature at a time, while real voices change several together, so it probes the model rather than creating a realistic voice. The timeline windows are shorter than the clips the model was trained on, so each window is rougher than the whole-clip result. All three caveats are written on screen.

## 3. Design decisions for honest predictions

1. **Real pipeline, verified.** The app calls the same `load_audio` → openSMILE → WavLM code as training. For a held-out RAVDESS clip, the features computed live in the app equal the stored training features (largest difference 0.00005, float rounding).
2. **Honest dataset demo.** The final model has heard every corpus clip, so playing one would overstate accuracy. `src/app_assets.py` trains the *same recipe* with all fold-0 speakers held out (5 RAVDESS + 18 CREMA-D speakers). Dataset examples come only from those speakers and use that model, so the true label is known **and** the speaker is new. On these speakers the recipe scored 72.5% (RAVDESS) and 72.3% (CREMA-D) UAR (Phase 9 notebook).
3. **"Model score", not "confidence".** Scores are shown as a score ring and six bars, with three trust notes underneath:
   - dataset clips: in-domain calibration is good (ECE 0.025 / 0.021, Phase 8 E6);
   - uploads and microphone: always warned that this is outside the training data, where cross-corpus UAR fell to 39–68% (Phase 8 E2) and scores are over-confident.
4. **Typicality check instead of a fixed threshold.** The first version warned when more than 10 of 88 features fell outside the 1–99% training range. Tested on the training clips themselves, that rule fired on **8.4% of RAVDESS clips**, too often to be useful. The app now ranks the clip against all training clips and warns only when it is more atypical than 95% of them.
5. **Two models, disagreement shown.** The prediction comes from the fusion model; the explanation from the interpretable eGeMAPS model. When the two disagree, the app says so rather than hiding it.
6. **Explanation caveats on screen.** Contributions are relative to the average training clip; correlated features (loudness and sound level) can get opposite signs; group-level evidence is more stable than single features (Phase 8, E3); SHAP shows association, not cause.

## 4. Stress test: why it is scientifically useful

Phase 7 found that the CREMA-D model relies on loudness (quiet → *sad*). Recording level also depends on the microphone and distance, not only on emotion. The simulation lets anyone check this live: if a copy 12 dB quieter flips the prediction to *sad*, the model is using recording level. Pitch shifts test reliance on speaker voice range (male vs female), and added noise tests recording quality. These are the same failure modes behind the cross-corpus drop in Phase 8.

## 5. Testing done

- October 2026 redesign: every page and all eight Analyse tabs were run in a headless browser on a test copy with the real saved models, real eGeMAPS features, real held-out RAVDESS and CREMA-D clips and the real `microsoft/wavlm-base-plus`; no errors. Checked: dataset clip, file upload, stress test (compass drift appears), what-if sliders, timeline, lens, a 400-pixel-wide phone layout. Example: a held-out RAVDESS neutral clip was called sad by the main model; in the two-dataset lens the RAVDESS-only model called it neutral while the CREMA-D-only and pooled models said sad.
- Earlier version: all five pages were rendered headless in a test copy of the project with the real Phase 6–9 result files and the real saved model; no errors on any page, tab or input mode (dataset example, random clip, file upload, robustness run).
- The test environment could not download WavLM weights, so it used a random stand-in for WavLM there; predictions shown in that test are therefore not meaningful. On the Mac the app uses the real `microsoft/wavlm-base-plus`.
- Pitch display: background hiss in the silent lead-in of some RAVDESS clips was being tracked as a flat 65 Hz "pitch"; frames pinned at the pitch-search floor are now dropped.

## 6. Live-demo plan (for the final assessment)

1. *Dataset clip* → CREMA-D → angry → show the console, then **Why** (loudness rising slope, sound level).
2. **What-if studio** → lower loudness attack and sound level → watch the answer move; show the smallest change that turns it sad.
3. **Two-dataset lens** → RAVDESS → neutral: the three training sources often disagree (our research question on one clip).
4. **Stress test** → then **Emotion compass** to show how far the copies drift.
5. *Record* → one sentence, happy then sad → point to the out-of-domain warning and the typicality check.
6. **New-dataset test** → H1 chart and neutral collapse (recall 0.09).
7. **Performance** → leakage chart (why speaker-independent splits matter).

## 7. Limitations

- Microphone recordings are out-of-domain; the app's scores for them are rough, as stated on screen.
- The explanation is for the eGeMAPS model; the fusion model's WavLM part (most of its evidence) has no human-readable names, which the app shows explicitly as the eGeMAPS vs WavLM evidence share.
- Runs locally only (needs the trained model, features and audio). It is a research demo, not a product.
