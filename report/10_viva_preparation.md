# Viva Preparation — Team Lucid

Each question has a **short answer** (what to say first, ~15 seconds) and a **deeper explanation** (if the examiner asks "why?" or "explain more"). Every number comes from our results; the source file is given where useful.

Owner initials show who leads the answer (**PS** Pranav Sajith · **PY** Piriya Shreyash Shanmukh · **PR** Praveen Raj). **Everyone must be able to answer every question**; see `10_team_plan.md`.

---

## A. Easy (definitions and project overview)

**A1. What does your project do? (all)**
- *Short:* It recognises six acted emotions from short speech clips and explains each prediction, then tests whether both the accuracy and the explanations survive when the model is moved to a different dataset.
- *Deeper:* Datasets RAVDESS and CREMA-D; features eGeMAPS (88 named acoustic features) and WavLM (pretrained speech embeddings); models LogReg, SVM, LightGBM, MLP; explanations with SHAP; speaker-independent evaluation; a Streamlit app runs the real model.

**A2. What is your research question? (all)**
- *Short:* When an emotion model trained on one corpus is applied to another, do its accuracy and its SHAP explanations hold up, and can explanation stability help choose features that transfer?
- *Deeper:* Three hypotheses: H1 (embeddings drop relatively more cross-corpus), H2 (explanation stability: resample > speaker > corpus > random), H3 (stable features transfer better), plus H0 (leakage inflation).

**A3. Which emotions, and why only six? (PS)**
- *Short:* Neutral, happy, sad, angry, fearful, disgust: the six both corpora share.
- *Deeper:* RAVDESS also has calm and surprised (384 clips), which CREMA-D lacks; keeping them would make cross-corpus testing impossible. Six classes means chance = 16.7%.

**A4. What is UAR and why use it? (PS)**
- *Short:* Unweighted Average Recall: the average of the per-emotion recalls.
- *Deeper:* It weights every emotion equally. RAVDESS neutral has half the clips of other emotions (96 vs 192), so plain accuracy would reward ignoring neutral. UAR equals scikit-learn's balanced accuracy. We also report macro-F1 and accuracy.

**A5. What is eGeMAPS? (PY)**
- *Short:* A standard set of 88 interpretable voice features (pitch, loudness, voice quality, formants, spectral balance, timing), extracted with openSMILE.
- *Deeper:* Defined by Eyben et al. 2016 (Paper 3) as a minimal, theory-based set for affective computing. Each feature has a physical meaning, so its SHAP explanation is readable. We group them into Energy (15), Frequency (24), Spectral (43) and Temporal (6).

**A6. What is WavLM? (PY)**
- *Short:* A self-supervised speech model pretrained by Microsoft on large amounts of unlabelled speech; we use its hidden states as features.
- *Deeper:* `microsoft/wavlm-base-plus`, 12 transformer layers + the CNN output = 13 layers of 768 values per frame. We keep it frozen, average over time, and pick one layer by inner cross-validation. We do not fine-tune it.

**A7. What is SHAP? (PY)**
- *Short:* A method that splits a prediction into contributions from each feature, based on Shapley values from game theory.
- *Deeper:* Each feature's contribution is its average marginal effect over feature coalitions; contributions sum to the prediction minus the average prediction. For a linear model with independent features it has an exact closed form: weight × (value − training mean). For trees, TreeSHAP is exact.

**A8. What is speaker-independent evaluation? (PS)**
- *Short:* No speaker appears in both training and test data.
- *Deeper:* We split by speaker with `StratifiedGroupKFold` (5 folds, stratified by emotion and gender). This tests generalisation to new people, which is what any real use requires.

**A9. Which model is best? (PS)**
- *Short:* Logistic Regression on fused eGeMAPS + WavLM features: 82.2% UAR on RAVDESS, 74.1% on CREMA-D.
- *Deeper:* WavLM alone is almost as good (81.0 / 73.8). Non-linear models did not help on embeddings.

**A10. What does your app do? (PR)**
- *Short:* It records or loads a clip, runs the real pipeline, shows the predicted emotion with class scores, explains why with SHAP, and warns when the result should not be trusted.
- *Deeper:* It also shows spectrogram, Mel-spectrogram and pitch, compares voice features with the training data, runs a robustness simulation (volume, pitch, noise), and presents all our experimental results interactively.

**A11. What is a confusion matrix? (PS)**
- *Short:* A table of true emotion vs predicted emotion.
- *Deeper:* Row-normalised, the diagonal is per-class recall. Ours shows fearful → sad as the main error (13.5% RAVDESS, 17.3% CREMA-D).

**A12. Why did you not use KNN? (PR)**
- *Short:* It was excluded by the project rules, and it is also a poor fit for this problem.
- *Deeper:* KNN classifies by distance to stored training clips, so (1) it is very sensitive to the corpus shift we study: our PCA shows the two corpora sit in different regions, and distance-based methods fail first (our RBF-SVM fell to 20.6% cross-corpus without normalisation); (2) in 768 dimensions distances become less informative; (3) it gives no global model to explain, and SHAP on KNN is expensive and approximate; (4) it would tend to match a clip to the same speaker's other clips, rewarding voice identity.

---

## B. Medium (method choices and results)

**B1. Why these two datasets? (PS)**
- *Short:* Free, widely used, share six emotions, and differ in exactly the ways we want to study.
- *Deeper:* RAVDESS: 24 actors, 2 sentences, studio-clean, 48 kHz. CREMA-D: 91 actors of varied age and ethnicity, 12 sentences, 16 kHz. Different size, diversity and recording conditions make a realistic but controlled cross-corpus test.

**B2. How did you preprocess the audio? (PY)**
- *Short:* Mono, resample to 16 kHz, trim edge silence; no volume normalisation.
- *Deeper:* 5 RAVDESS files were stereo; resampling removes a 48 vs 16 kHz format confound and is required by WavLM; trimming (top_db 30) removed ~1.8 s of extra silence from RAVDESS. We did not normalise volume because loudness is an emotion cue; standardisation is done later inside each training pipeline.

**B3. How much did leakage inflate results? (PS)**
- *Short:* Up to 9.3 points on RAVDESS, about 1.5 points on CREMA-D.
- *Deeper:* Same LogReg, random split vs speaker split: RAVDESS eGeMAPS 63.2 vs 53.9, WavLM 91.8 vs 84.0; CREMA-D 52.3 vs 50.8, 74.2 vs 72.5. Fewer speakers → each test speaker's voice is more familiar → more inflation. A RAVDESS score above 90% can be a leakage artefact.

**B4. How did you tune hyperparameters without leakage? (PS)**
- *Short:* Nested cross-validation: tuning inside the training folds only, grouped by speaker.
- *Deeper:* For each of the 5 outer folds, a 3-fold GroupKFold on the training speakers picks C (or the LightGBM/MLP parameter) and the WavLM layer. Scaling and clipping are inside the pipeline. The outer test fold is used once.

**B5. Which WavLM layer works best and why? (PY)**
- *Short:* The middle layers, 5–7.
- *Deeper:* Inner-CV UAR peaks at layer 5 on RAVDESS (79.3%) and plateaus at layers 5–7 on CREMA-D (~71%), then falls toward layer 12. Lower layers are closer to raw acoustics; the top layers specialise in the pre-training objective (predicting masked speech units), which is more about content than paralinguistics. This matches Paper 4. The final model chose layer 7.

**B6. Why is Logistic Regression your main model? (PS)**
- *Short:* It is the best or tied-best and its SHAP explanation is exact.
- *Deeper:* On 768-d self-supervised embeddings classes are close to linearly separable; SVM, LightGBM and MLP did not improve. LightGBM memorised training speakers (≈100% train UAR) without gaining on new ones. Linear SHAP needs no sampling, so explanation noise comes only from the data, which is essential for the stability study.

**B7. Does fusion help? (PY)**
- *Short:* Barely within a corpus (+0.3 to +1.2), more across corpora (up to +7.5).
- *Deeper:* Within a corpus WavLM already captures what eGeMAPS measures. Across corpora, CREMA-D → RAVDESS with the source scaler: WavLM 58.8 vs fusion 66.3. Named prosodic features add some robustness when the domain changes.

**B8. What did SHAP tell you the models use? (PY)**
- *Short:* The CREMA-D model relies mostly on loudness and sound level; the RAVDESS model on spectral shape (MFCCs) and formants.
- *Deeper:* Top CREMA-D features: equivalent sound level, loudness mean, F2 amplitude. Directions are plausible: angry has a steeper loudness rise and more high-frequency energy; sad is quieter. Per feature, Energy is the most informative group in both corpora, as Paper 5 found.

**B9. How did you check that SHAP is trustworthy? (PY)**
- *Short:* A second method, plus checks of exactness and plausibility.
- *Deeper:* (1) Our linear SHAP equals `shap.LinearExplainer` (difference 0.0). (2) Permutation importance agrees: Spearman 0.79 (RAVDESS), 0.96 (CREMA-D). (3) Directions match known vocal acoustics. (4) Every clip is explained by a model that never heard its speaker. (5) We measured stability (E3).

**B10. What is the clipping fix? (PY)**
- *Short:* SHAP showed some features contributing ±170 log-odds; we found exploding ratio features and clipped standardised values to ±5 SD.
- *Deeper:* eGeMAPS "stddevNorm" = standard deviation ÷ mean; with a mean near zero it reaches 86 SD in CREMA-D. One such value could decide a clip. After clipping inside every pipeline and re-running everything, eGeMAPS UAR rose slightly and the largest contribution fell to ~10. Accuracy alone had hidden the problem: XAI used for debugging.

**B11. What happened cross-corpus? (PR)**
- *Short:* Every model lost 20–35 UAR points.
- *Deeper:* Best LogReg results: RAVDESS → CREMA-D 48.6% (fusion, per-corpus normalisation); CREMA-D → RAVDESS 68.3%. Training on the larger, more varied CREMA-D transfers much better.

**B12. Was H1 supported? (PR)**
- *Short:* Only in one direction.
- *Deeper:* From RAVDESS, WavLM's relative drop was larger than eGeMAPS's by 6.1 points [3.1, 9.4] (source norm). From CREMA-D it was 18.4 points *smaller* [−27.9, −8.1]. Interpretation: a high-dimensional representation over-fits a narrow training corpus but learns transferable cues from a diverse one. WavLM had the higher absolute UAR in every case.

**B13. What is per-corpus normalisation and is it cheating? (PR)**
- *Short:* Each corpus is standardised with its own mean and standard deviation. It uses the target's unlabelled features, never its labels, so it is not label leakage.
- *Deeper:* It is a standard unsupervised adaptation in cross-corpus SER (Schuller et al. 2010). We always report both versions; the strict "source" version uses no target information at all. It recovered 2–13 points, showing part of the gap is a shift in feature levels (e.g., recording volume).

**B14. Were the explanations stable (H2)? (PY)**
- *Short:* Partly, and in the predicted order.
- *Deeper:* Median Spearman of importance rankings for LogReg-eGeMAPS: 0.78 between resamples, 0.63 between disjoint speaker groups, 0.50 between corpora, 0.00 for random rankings. LightGBM: 0.76 / 0.44 / 0.20; WavLM: 0.68 / 0.34 / 0.10. All orderings significant (p < 10⁻¹⁰). Feature-group shares stayed stable.

**B15. Did stability-guided feature selection work (H3)? (PR)**
- *Short:* No clear benefit: inconclusive.
- *Deeper:* Across 12 settings: 1 significant gain (+5.0, CREMA-D → RAVDESS, k = 30, per-corpus norm), 4 significant losses (up to −3.8), 7 not different from zero. The two subsets overlap 70–87%. We report it as a negative result.

**B16. What are the main errors? (PR)**
- *Short:* Low-energy emotions are pulled to sad; neutral breaks across corpora.
- *Deeper:* Fearful → sad 17.3% on CREMA-D. SHAP shows a quiet fearful clip is pushed to sad by its low sound level. RAVDESS → CREMA-D, neutral recall falls to 0.09 because RAVDESS neutral is calm and controlled, CREMA-D neutral is not.

**B17. Is your model biased? (PR)**
- *Short:* We saw a small gender gap in-domain; we report it as an observation, not proven bias.
- *Deeper:* WavLM UAR is 4–7 points higher for female speakers (CREMA-D 75.9 vs 71.9; RAVDESS 84.4 vs 77.8); the gap vanishes cross-corpus. Age effects ≤ 3 points. Folds are gender-balanced. With 24 / 91 actors we cannot generalise; acted English speech from mostly young North American actors is itself a population bias.

**B18. How good is your model's confidence? (PS)**
- *Short:* Well calibrated within the training corpora (ECE ≈ 0.02), but over-confident on new recording conditions.
- *Deeper:* ECE 0.025 (RAVDESS) and 0.021 (CREMA-D) for the fusion model on new speakers. The app therefore calls it a "model score", gives a strength label, warns for uploads and microphone input, and ranks how unusual the clip is.

**B19. Is the model overfitting? (PS)**
- *Short:* All models fit their training speakers better than new speakers; validation scores on new speakers are the honest numbers.
- *Deeper:* LogReg fusion: 99.7 → 82.0 (RAVDESS), 86.6 → 74.1 (CREMA-D). The gap is larger on RAVDESS (≈19 training speakers per fold). LightGBM reaches 100% train UAR without better validation. The learning curve shows the gap closing as speakers are added.

**B20. How does your app avoid misleading users? (PR)**
- *Short:* It uses the real model, says when it is out of its depth, and shows disagreement and uncertainty.
- *Deeper:* Dataset demos use a model trained without those speakers; uploads get an explicit out-of-domain warning; a typicality check ranks the clip against all 8,498 training clips; model disagreement is shown; probabilities are labelled "model score"; SHAP caveats are on screen. We verified live features equal stored training features (difference 5 × 10⁻⁵).

---

## C. Difficult (critical and research-level)

**C1. Your model beats humans on CREMA-D (74% vs 41%). Isn't that suspicious? (PS)**
- *Short:* The two numbers measure different things, so we don't claim the model "beats" humans.
- *Deeper:* Humans heard each clip once, without training on these actors, and were scored against the *intended* label; the model was trained on thousands of clips from the same corpus and can exploit corpus regularities (recording level, sentence set) that do not transfer: cross-corpus it falls to 39–68%. Clips humans disagreed on are also hard for the model (67% vs 84% UAR).

**C2. Isn't SHAP on correlated features misleading? (PY)**
- *Short:* It can split or flip credit between correlated features, which is why we emphasise group-level explanations.
- *Deeper:* Our linear SHAP is interventional (assumes independence). Correlated pairs such as loudness and sound level can get opposite signs that cancel. Evidence: per-feature rankings change across corpora (Spearman 0.50), but group shares stay stable. Permutation importance also under-rates a feature whose "twin" carries the same information, explaining lower agreement on RAVDESS.

**C3. Why not explain WavLM directly? (PY)**
- *Short:* Its 768 dimensions have no human meaning, so we measure their stability but explain with eGeMAPS.
- *Deeper:* We computed SHAP on WavLM for E3 (it was the least stable: 0.10 across corpora). For human-readable explanation, the app shows how much of the fusion model's evidence comes from named features vs WavLM. Future work: probe WavLM dimensions with eGeMAPS (Paper 5's approach) to give them names.

**C4. Why LogReg-based SHAP instead of LIME? (PY)**
- *Short:* For a linear model SHAP is exact and deterministic; LIME is a sampled local approximation whose output changes between runs.
- *Deeper:* Our research question is about explanation *stability*. Using an explainer that adds its own randomness would mix explainer noise with model/data variation. Exact SHAP isolates the variation we want to measure. We cross-checked with permutation importance, a model-agnostic method.

**C5. How do you know the cross-corpus drop is about emotion and not recording differences? (PR)**
- *Short:* We don't fully separate them; we show that part of it is recording-level shift.
- *Deeper:* Evidence: (1) PCA shows corpus clusters; (2) per-corpus normalisation (which only removes level/scale differences) recovers 2–13 points; (3) formant-amplitude features sit at very different levels (median F1 amplitude −83 vs −150 dB rel. F0); (4) the label "neutral" is performed differently. The remaining gap mixes acting style, speakers, sentences and recording conditions.

**C6. Your H1 result goes both ways. Isn't that just noise? (PR)**
- *Short:* No: both directions have confidence intervals that exclude zero in the source setting, with opposite signs.
- *Deeper:* +6.1 [3.1, 9.4] and −18.4 [−27.9, −8.1], from speaker-level bootstrap with paired resamples. The direction dependence is a finding: it suggests training-set diversity controls how well embeddings transfer. But it rests on one corpus pair; a third corpus is needed to generalise it.

**C7. Why bootstrap over speakers rather than clips? (PS)**
- *Short:* Clips from the same speaker are not independent, so clip-level bootstrap would give falsely narrow intervals.
- *Deeper:* A speaker contributes 60–82 correlated clips. Resampling speakers with replacement treats the speaker as the unit of generalisation, which matches our claim ("new people"). Paired resamples are used for differences between methods.

**C8. With 5 folds you can't get p < 0.05. How can you claim WavLM is better? (PS)**
- *Short:* We report it as consistent in 5/5 folds on both corpora, not as a significance test.
- *Deeper:* The minimum two-sided Wilcoxon p-value with 5 pairs is 0.0625. The difference is large (22–26 points) and consistent in all 10 fold comparisons. Cross-corpus claims use bootstrap CIs, which do support significance statements.

**C9. Isn't pooling both corpora in the final model inconsistent with your cross-corpus findings? (PR)**
- *Short:* No; the cross-corpus experiments are the evaluation; the final model is the deployment choice, and we tested its recipe on held-out speakers.
- *Deeper:* Pooling cost RAVDESS about 2.5 points (72.5 vs 75.0) and helped CREMA-D slightly (72.3 vs 71.5) on fold-0 speakers, but gives one model that has heard both recording styles; E2 showed single-corpus models break on the other corpus. Paper 8 also reported that adding CREMA-D can lower other corpora.

**C10. What exactly is novel here? (all)**
- *Short:* We don't claim a "first". We measured something we did not find measured: how stable SHAP explanations of SER models are across resamples, speakers and corpora, and whether stability helps transfer, for RAVDESS ↔ CREMA-D.
- *Deeper:* Our search found related work counting recurring features within corpora (Kaloga & Kodrasi 2025; Kołodziej 2026) and cross-corpus benchmarks without explainability, but no formal stability measurement linked to transfer. One relevant 2026 article could not be accessed, so the claim is "underexplored", stated carefully.

**C11. What would you do with more time or compute? (all)**
- *Short:* More corpora, more seeds, explanations for WavLM via probing, and better domain adaptation.
- *Deeper:* A non-acted or non-English corpus to test H1/H2 generality; 5 seeds and leave-one-speaker-out for formal tests; probing WavLM with eGeMAPS; domain-adversarial training or speaker normalisation; soft labels from CREMA-D listener votes; calibration under shift; the robustness simulation as a formal experiment over all held-out clips.

**C12. If H3 failed, why include it? (PR)**
- *Short:* Negative results are results; leaving them out would bias the report.
- *Deeper:* H3 was a falsifiable hypothesis stated before the experiment (Phase 2). It tells practitioners that stable explanations are a diagnostic, not an automatic route to robust features. Hiding it would overstate what XAI can do.

**C13. Your RAVDESS test folds have 4–5 speakers. Are the results reliable? (PS)**
- *Short:* They are noisier, so we report fold SDs and use CREMA-D (18–19 speakers per fold) as the stable reference.
- *Deeper:* RAVDESS fold SD is ±5–9 points vs ±2–3 for CREMA-D. That is also why single-split RAVDESS numbers in the literature should be read carefully.

**C14. Could the model be recognising sentences rather than emotions? (PR)**
- *Short:* Unlikely to be the main driver: every emotion is spoken with the same sentences.
- *Deeper:* In both corpora each actor speaks each sentence in every emotion, so sentence identity is balanced across classes and cannot predict the label. Sentence differences between corpora (2 vs 12 sentences) do contribute to the cross-corpus shift.

**C15. Why should anyone trust an emotion recogniser at all? (all)**
- *Short:* They shouldn't use it for decisions about people; it recognises acted expression in clean clips.
- *Deeper:* Our own results show 20–35-point drops under domain shift, gender gaps and label ambiguity. Responsible use means research and assistive contexts with human oversight, transparency about limits, and no inference of internal emotional state.

---

## D. The real-voice upgrade (calibration, corrections, robust training, contrastive explanations)

**D1. Your app said "sad" for normal laptop speech. Why? (PS)**
- *Short:* Domain shift. The model was trained on studio audio and leans on loudness; a laptop microphone records quieter and band-limited speech, which resembles the studio *sad* clips.
- *Deeper:* Phase 7 showed loudness and sound level drive the CREMA-D model; Phase 8 measured feature shifts between corpora. On simulated laptop audio the studio model called neutral speech sad in 17% (CREMA-D speakers) to 55% (RAVDESS speakers) of cases (`report/09_simulation.md` §9).

**D2. How does "calibrate to my voice" work, and is it cheating? (PY)**
- *Short:* Three calm recordings form a baseline; a second model judges each clip as *clip − baseline*; the answer is the geometric mean of both models. It uses no labels from the user, so it is not cheating; it is per-corpus normalisation done for one speaker.
- *Deeper:* Held-out speakers: 69.3 → 76.6 (RAVDESS), 71.8 → 74.9 (CREMA-D); RAVDESS → CREMA-D 42.3 → 51.7. Mean of 10 random baselines; baseline clips removed from the test set. The geometric mean beat the arithmetic mean in 3 of 4 tests and needs both models to agree.

**D3. What does "Wrong answer? I was …" do, and what went wrong at first? (PR)**
- *Short:* The corrected clip is added, with weight 100, to the relative model's training data and only the last layer is re-fitted in seconds. The first version also re-fitted the original model, which pulled all of a new speaker's clips toward the corrected emotion (52 → 42% in one test), so it was rejected.
- *Deeper:* Simulated users correcting each mistake: after 3 corrections +7.2 (RAVDESS), +2.3 (CREMA-D), +4.9 (CREMA-D → RAVDESS) and +1.0 (RAVDESS → CREMA-D) points; happy called angry 21% → 10% in CREMA-D → RAVDESS. Weight chosen on fold-1 speakers. Corrections stay in the browser session.

**D4. Can your model detect excitement? (all)**
- *Short:* No. There is no "excited" class in either corpus. Excitement and anger share high pitch, loudness and fast speech; energy is easy to hear, positivity much harder. Through corrections, a user's excitement is learned as *happy*.
- *Deeper:* The contrastive explanation "why angry and not happy?" shows which features separate the two for a given clip; the shared energy cues cancel out in it.

**D5. What is the microphone-robust model, and how do you know it is not just memorising your simulation? (PS)**
- *Short:* The same recipe trained on every clip twice: as recorded and through a random simulated microphone and room. It was tested on held-out speakers under three fixed conditions, one of them (far room) partly outside the training ranges.
- *Deeper:* Gains with 95% speaker-bootstrap CIs: CREMA-D laptop +3.7 [1.6, 5.6], phone +7.8 [4.3, 11.1], far room +15.4 [10.5, 20.2]; RAVDESS far room +18.3 [12.1, 27.1]. No loss on studio audio (+0.2, +2.1). Honest caveats: simulated, not real microphones; the RAVDESS laptop and phone CIs include zero; across corpora on clean audio it was slightly, not significantly, worse.

**D6. How do you compute "why angry and not happy?" (PY)**
- *Short:* SHAP for angry minus SHAP for happy. For a multinomial Logistic Regression this is exactly each feature's contribution to log P(angry) − log P(happy), relative to the average training clip.
- *Deeper:* log-odds of k over j = (w_k − w_j)·z + (b_k − b_j); subtracting the same quantity for the average clip gives Σ (w_k,i − w_j,i)(z_i − z̄_i), which is the difference of the two classes' linear SHAP values. Features that push both emotions equally (shared energy cues) cancel.

**D7. Can someone reproduce these numbers? (PR)**
- *Short:* Yes: `python -m src.voice_experiments calibration`, `corrections` and `robust` write `results/phase9/voice_*.csv`. The simulated-microphone features come from `python -m src.robust_features`.
- *Deeper:* The original audio was downloaded again from the official sources; re-extracted features matched the stored ones exactly (eGeMAPS within 10⁻⁹, WavLM correlation 1.00000).

**D8. Your app misread your own voice. What did you do about it, and is it not just tuning to yourself? (PS)**
- *Short:* Speaker adaptation, a standard remedy for a speaker unlike the training speakers: after calibrating, the user says each of the six emotions once, and both models re-fit their last layer with those six examples. We tested it on speakers the models never heard, not on ourselves.
- *Deeper:* One other clip of each emotion was taught, and the speaker's remaining clips were scored. UAR rose in all ten tests (e.g. RAVDESS 77.9 → 85.0, +7.1 [1.2, 13.3]; CREMA-D → RAVDESS 68.3 → 76.5, +8.2 [5.5, 11.4]); eight intervals exclude zero. Happy called angry in CREMA-D → RAVDESS fell from 26% to 10%. Single corrections still teach only the relative model, because one correction of one emotion pulled all of a new speaker's clips towards it (section 8); a balanced set of six does not. On twelve recordings of our own voice: 11 correct against 8, reported as a small check, not as evidence (`python -m src.voice_experiments enrolment`).

---

## E. Quick-fire numbers to memorise

| Fact | Value |
|---|---|
| Clips used | RAVDESS 1,056 (24 speakers) · CREMA-D 7,442 (91) |
| Best within-corpus UAR | 82.2% RAVDESS · 74.1% CREMA-D (LogReg fusion) |
| eGeMAPS within-corpus UAR | 55.4% · 51.4% (LogReg) |
| Leakage inflation | up to +9.3 (RAVDESS) · +1.7 (CREMA-D) |
| Best WavLM layers | 5–7 (final model: 7) |
| Cross-corpus best | R → C 48.6% · C → R 68.3% (fusion, per-corpus norm) |
| H1 | +6.1 [3.1, 9.4] from RAVDESS · −18.4 [−27.9, −8.1] from CREMA-D |
| Stability (LogReg eGeMAPS) | 0.78 / 0.63 / 0.50 / 0.00 |
| ECE | 0.025 · 0.021 |
| Macro ROC-AUC (fusion) | 0.968 · 0.944 |
| SHAP vs permutation | 0.79 · 0.96 |
| Human audio-only agreement | 63% RAVDESS · 41.6% CREMA-D |
| Main confusion | fearful → sad (13.5% · 17.3%) |
| Calibration (held-out) | 69.3 → 76.6 RAVDESS · 71.8 → 74.9 CREMA-D |
| Robust model, far room | 29.6 → 47.9 RAVDESS · 33.4 → 48.8 CREMA-D |
| Corrections (3) | happy called angry 21% → 10% (CREMA-D → RAVDESS) |
| Teach it your voice (1 clip per emotion) | RAVDESS 77.9 → 85.0; CREMA-D → RAVDESS 68.3 → 76.5; dataset clips 80.5 → 86.9 |
| Close calls (top two within 15 points) | 11–13% of clips; top answer right 33–40%, truth among the two 73–88% |
