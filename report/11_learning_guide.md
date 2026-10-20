# The Story of Your Project — a learning guide from zero

Team Lucid · Read this like a story, in order. Each chapter first explains a basic idea with an everyday example, then shows **where it appears in your project**. Every number is from your own results.

> **The whole project in one sentence:**
> We taught a computer to hear emotion in a voice, made it explain *why* it decided, and then tested whether both the decision and the explanation still hold when the voices come from a different recording studio.

---

## Chapter 1 — What is machine learning?

### The basic idea
Imagine teaching a child to tell cats from dogs. You don't write rules like "if ears are pointy then cat". You show **hundreds of labelled examples** ("this is a cat", "this is a dog"), and the child figures out the pattern.

Machine learning (ML) is the same:

| Word | Meaning | Child example |
|---|---|---|
| **Example** (sample) | one thing to learn from | one photo |
| **Label** | the correct answer | "cat" |
| **Features** | numbers that describe the example | ear shape, size, fur |
| **Model** | the "brain" that learns a pattern from features → label | the child's understanding |
| **Training** | showing labelled examples so the model learns | studying photos |
| **Prediction** | the model's answer for a new example | "I think it's a dog" |
| **Testing** | checking predictions on examples it has **never seen** | a surprise quiz |

This kind of ML, with correct answers given, is called **supervised classification**: the model chooses one class from a fixed list.

### In your project
- **Example** = one short audio clip of an actor saying a sentence.
- **Label** = the emotion the actor was asked to perform: **neutral, happy, sad, angry, fearful, disgust** (6 classes).
- **Features** = numbers describing the voice (Chapter 4).
- **Models** = Logistic Regression, SVM, LightGBM, MLP (Chapter 6).
- **Prediction** = "this clip sounds *angry*".

**Chance level:** with 6 classes, random guessing is right 1 time in 6 = **16.7%**. Any result must be compared with this.

---

## Chapter 2 — What is sound, and how does emotion live in it?

### The basic idea
Sound is air vibrating. A microphone measures the air pressure **many times per second**; each measurement is a number. The list of numbers is the **waveform**.

- **Sample rate** = how many measurements per second. 16,000 per second = **16 kHz**.
- **Mono** = one channel (one microphone). **Stereo** = two (left and right).
- **Pitch (F0)** = how high or low the voice is (how fast the vocal folds vibrate). Women and children usually have higher pitch.
- **Loudness / energy** = how strong the vibration is.
- **Spectrogram** = a picture of sound: time on the x-axis, frequency (low to high) on the y-axis, brightness = how much energy. It shows *which* frequencies are present *when*.
- **Formants (F1, F2, F3)** = the resonances of your mouth and throat; they make "ee" sound different from "oo".
- **MFCCs** = a compact summary of the *shape* of the spectrum (the "colour" or timbre of the voice).

### How emotion shows in the voice
| Emotion | Typical voice |
|---|---|
| Angry | loud, sharp rises in loudness, tense, more high-frequency energy |
| Sad | quiet, low energy, slow, little change |
| Happy | higher and more varied pitch, lively |
| Fearful | breathy or shaky voice quality |
| Neutral | steady, unremarkable |

Your model **found these same patterns by itself** (Chapter 9). That is a good sign that it learned something real.

---

## Chapter 3 — The data: two "schools" of actors

### The basic idea
A **dataset** (or **corpus**) is a collection of labelled examples. Its quality and variety decide what a model can learn.

### Your two datasets
| | RAVDESS | CREMA-D |
|---|---|---|
| Actors (speakers) | **24**, young, one North American accent | **91**, ages 20–74, many backgrounds |
| Sentences | 2 | 12 |
| Clips you used | **1,056** | **7,442** |
| Recording | 48 kHz studio | 16 kHz |

Think of them as **two different schools**: RAVDESS is a small, quiet, elite school; CREMA-D is a big, diverse school. Your research question is basically: *does a student trained in one school still do well in the other?*

### What you did with the data (Phase 3)
1. **Harmonised the labels.** RAVDESS has 8 emotions, CREMA-D 6. You kept only the 6 shared ones (dropped RAVDESS *calm* and *surprised*).
2. **Found class imbalance.** RAVDESS has only 96 neutral clips vs 192 for the others. Imbalance means plain accuracy can mislead (Chapter 7).
3. **Checked label quality.** CREMA-D has votes from human listeners. Listeners agreed with the actor's intended emotion only **41.6%** of the time, and for *sad* only **16%**. So some "labels" are doubtful even for humans.
4. **Made speaker-independent folds** (explained in Chapter 5).

**Acted vs real:** both datasets use actors performing emotions. Real-life emotion is subtler. This is a limitation you always mention.

---

## Chapter 4 — Preparing the audio and turning it into numbers

### 4.1 Preprocessing (Phase 5): make all clips comparable
Like making every student write the exam with the same pen and paper:
1. **Mono:** 5 RAVDESS files were stereo → averaged into one channel.
2. **Resample to 16 kHz:** RAVDESS was 48 kHz, CREMA-D 16 kHz. If you didn't fix this, the model could tell the datasets apart just from the format. WavLM also requires 16 kHz.
3. **Trim silence** at the start and end: RAVDESS clips had ~1.8 s of extra silence.
4. **No volume normalisation:** loudness *is* an emotion clue (angry = loud), so you kept it.

### 4.2 Features: two ways to describe a voice

**Way 1: hand-made features (eGeMAPS).** Experts decided which voice measurements matter for emotion. A tool called **openSMILE** computes **88 named numbers** for each clip: average pitch, pitch variation, loudness, jitter (pitch shakiness), shimmer (loudness shakiness), harmonics-to-noise ratio (clear vs breathy voice), formants, MFCCs, speaking rate…
- **Pro:** each number has a meaning, so explanations are readable ("the voice was quiet").
- **Con:** limited; it only measures what experts thought of.
- Grouped into **Energy (15), Frequency (24), Spectral (43), Temporal (6)**.

**Way 2: learned features (WavLM).** WavLM is a big neural network made by Microsoft and trained on huge amounts of *unlabelled* speech with a fill-in-the-blank game: parts of the audio are hidden and it must guess them. This is **self-supervised learning**: it learns from the data itself, with no human labels. While learning to fill blanks, it learns a lot about speech, including tone.
- You **froze** it (didn't change it) and used it as a **feature extractor**: the clip goes in, and out come **768 numbers** describing it.
- WavLM has **13 layers** (like floors in a building). Lower floors know raw sound, top floors know words. **Middle layers (5–7) held the most emotion**, which you found by testing every layer. Your final model uses **layer 7**.
- **Mean-pooling:** WavLM gives numbers for every 20 ms; you averaged them over time to get one 768-number summary per clip.
- **Pro:** much more accurate. **Con:** 768 numbers with no names, so explanations are not human-readable.

**Fusion** = putting both together: 88 + 768 = **856 numbers**.

---

## Chapter 5 — The golden rule: never test on what you trained on

This is the most important chapter for the viva.

### 5.1 Train vs test, and overfitting
If a student memorises last year's exact exam answers, they score 100% on that paper but fail a new one. That's **overfitting**: learning the training examples too specifically instead of the general pattern. So we **always test on examples the model has never seen**.

### 5.2 Leakage: the hidden cheating
**Leakage** = information from the test set sneaking into training, which makes results look better than they really are.

In speech, the classic leak is the **same speaker in training and test**. The model can learn "this is Actor 7's voice, and Actor 7's angry clips sound like this", recognising *voices* instead of *emotions*.

### 5.3 Speaker-independent evaluation (your fix)
You split by **speaker**: every actor's clips go entirely into training *or* entirely into test, never both. You used **StratifiedGroupKFold**:
- **Group** = speaker (keeps each speaker together),
- **Stratified** = each part has a similar mix of emotions **and genders**.

### 5.4 Cross-validation (CV) and folds
One test split could be lucky or unlucky. So you split the speakers into **5 folds** (5 groups). You train on 4 and test on the 5th, and repeat 5 times so every fold is tested once. Then you report the **average ± spread**. This is **5-fold cross-validation**.

### 5.5 Hyperparameters and nested CV
Models have settings you choose before training, called **hyperparameters** (for example, how strongly to keep the model simple). Choosing them by looking at test results is also cheating. So inside each training part you did a **second, smaller CV (3 folds, also by speaker)** just to choose settings and the WavLM layer. This is **nested cross-validation**. The outer test fold is touched only once, at the end.

### 5.6 Your proof that this matters (Experiment E0, hypothesis H0)
Same model, two ways of splitting:

| | Random split (leaky) | Speaker split (honest) | Inflation |
|---|---|---|---|
| RAVDESS eGeMAPS | 63.2% | 53.9% | **+9.3** |
| RAVDESS WavLM | 91.8% | 84.0% | **+7.8** |
| CREMA-D eGeMAPS | 52.3% | 50.8% | +1.5 |
| CREMA-D WavLM | 74.2% | 72.5% | +1.7 |

**Story:** with only 24 actors, each test actor's voice is "familiar" if leaked, so cheating helps a lot (+9). With 91 actors it helps less (+1.5). **H0 supported.** *Memory hook: "Small school, big cheating."*

---

## Chapter 6 — The models (the "brains")

### 6.0 Two preparation steps inside every model
- **Standardisation (StandardScaler):** features have different units (pitch in semitones, loudness in sones…). Standardising rescales each feature to mean 0 and standard deviation 1, so no feature dominates just because of its units. It is learned **from training data only**.
- **Clipping to ±5 SD:** any value more than 5 standard deviations from the mean is capped. Chapter 9 tells why you added this.
- Both live inside a **Pipeline**, so they are always fitted on training data only. This is another anti-leakage measure.

### 6.1 Logistic Regression (your main model)
Imagine a judge with a scorecard. Each feature has a **weight** (how much it matters and in which direction). For each emotion the judge adds up `weight × feature value` to get a **score**. The scores are turned into **probabilities** that add up to 100% (the *softmax* step), and the highest one wins.
- Simple, fast, and its explanation is **exact** (Chapter 8).
- **C** = how much freedom the model has. A small C keeps weights small, which prevents overfitting. This is called **regularisation**.
- Despite the name "regression", it is a **classifier**.

### 6.2 SVM (Support Vector Machine, RBF kernel)
It draws the **boundary between classes with the widest possible safety gap** (margin). The RBF kernel lets the boundary curve. It works by measuring **distances** between examples.

### 6.3 LightGBM (gradient-boosted trees)
A **decision tree** is a flowchart of yes/no questions ("loudness > 0.5? then…"). **Boosting** builds hundreds of small trees one after another, each fixing the previous ones' mistakes. It is strong on tables of features and can capture **interactions** between features.

### 6.4 MLP (Multi-Layer Perceptron)
A small **neural network**: layers of "neurons" (256 of them), each combining inputs with weights and a non-linear function. It is flexible but needs care to avoid overfitting (you used early stopping).

### 6.5 Why no KNN?
KNN (k-nearest neighbours) predicts by finding the most similar stored training clips. It was **excluded by your project rule**, and it is also a bad fit:
1. It relies on distances, which break when corpora differ (your SVM, also distance-based, collapsed to 20.6% cross-corpus).
2. Distances lose meaning in 768 dimensions.
3. It would match a clip to the *same speaker's* clips, rewarding voice identity.
4. It has no global pattern to explain.

### 6.6 Results inside each corpus (Experiment E1)
Average UAR over the 5 speaker-independent folds (UAR is explained in Chapter 7):

| | eGeMAPS | WavLM | Fusion |
|---|---|---|---|
| RAVDESS (LogReg) | 55.4% | 81.0% | **82.2%** |
| CREMA-D (LogReg) | 51.4% | 73.8% | **74.1%** |

- WavLM beats eGeMAPS by **22–26 points**, in **every one of the 5 folds** in both corpora.
- Fusion adds only a little.
- **Logistic Regression was best or tied-best.** The fancy non-linear models didn't help on WavLM features, because WavLM's numbers are already arranged so that a straight-line boundary works well.

---

## Chapter 7 — How to measure "good": the metrics

Use a medical test as the example: 100 patients, 10 are sick.

| Metric | Question it answers |
|---|---|
| **Accuracy** | out of all predictions, how many were right? (A test that always says "healthy" gets 90%, which shows why accuracy can mislead with imbalance) |
| **Recall** (per class) | of the truly sick, how many did we catch? |
| **Precision** (per class) | of those we called sick, how many really were? |
| **F1** | balance of precision and recall |
| **UAR** (Unweighted Average Recall) | **average the recall of each emotion equally**, your main metric; fair even when classes are imbalanced (= *balanced accuracy*) |
| **Macro-F1** | average F1 over emotions |
| **Confusion matrix** | table of true emotion vs predicted emotion; shows *which* emotions get mixed up |
| **ROC-AUC** | how well the model *ranks*: if you pick one angry and one non-angry clip at random, how often is the angry one scored higher? 0.5 = random, 1.0 = perfect. Yours: **0.97** (RAVDESS), **0.94** (CREMA-D) |
| **Calibration / ECE** | when the model says "70% sure", is it right 70% of the time? ECE = average gap. Yours: **0.025 / 0.021**, very honest within the training corpora |

**Your main confusion:** *fearful → sad* (13.5% RAVDESS, 17.3% CREMA-D). Quiet emotions get pulled toward *sad*.

---

## Chapter 8 — Explainability (XAI): asking the model "why?"

### 8.1 The basic idea: SHAP
Four friends win a ₹100 prize as a team. How do you split it fairly? You look at how much the prize changes when each friend joins, averaged over every order in which they could have joined. That is the **Shapley value**, from game theory.

**SHAP** applies this to models: the "prize" is the model's score for an emotion, and the "players" are the features. Each feature gets a **contribution**: positive = pushed *toward* that emotion, negative = pushed *away*. The contributions add up exactly to (this clip's score − the average score).

### 8.2 SHAP for your Logistic Regression is exact
For a linear model, SHAP has a simple formula:

> contribution of a feature = **weight × (this clip's value − average training value)**

No guessing or sampling, and you verified it matches the official SHAP library exactly. For LightGBM you used **TreeSHAP**, also exact.

### 8.3 Global vs local
- **Local explanation** = why *this one clip* got its prediction (used in your app).
- **Global explanation** = average absolute contribution over all clips: which features matter most overall.

### 8.4 Out-of-fold explanation
Every clip was explained by a model that had **never heard that speaker**, so the explanations describe a model that generalises, not one that memorised.

### 8.5 A second opinion: permutation importance
Shuffle one feature's values (which destroys its information) and see how much accuracy drops. A big drop means the feature was important. This is a completely different method, and it agreed with SHAP: Spearman **0.79** (RAVDESS), **0.96** (CREMA-D).

### 8.6 What SHAP revealed (Phase 7)
- **CREMA-D model listens mostly to loudness and sound level**; **RAVDESS model to spectral shape (MFCCs) and formants**.
- **Per feature, the Energy group is the most informative** in both corpora, which matches Paper 5.
- Directions make sense: *angry* is pushed by steeper loudness rises; *sad* by low sound level. This also explains the main error: a **quiet fearful** clip gets pushed to **sad**.

### 8.7 The bug story (great for the viva!)
The first SHAP run showed a single feature contributing **±170** to one clip, when normal values are below 1. Investigating it, you found that eGeMAPS "stddevNorm" features = standard deviation ÷ mean, and when the mean is near zero the value **explodes** (up to 86 standard deviations). One crazy number could decide the prediction.
**Fix:** clip values to ±5 SD in every pipeline and re-run everything. Accuracy barely changed (54.6 → 55.4%), which is exactly why **accuracy alone would never have revealed the bug**. *Memory hook: "SHAP was the X-ray; accuracy was just the thermometer."*

### 8.8 Shortcut check
39 CREMA-D clips had *no detected pitch* (very soft or breathy, mostly sad). Was the model cheating with "no pitch → sad"? You removed them and retrained: UAR 51.41% vs 51.40%. No shortcut.

**Remember:** SHAP shows what the **model uses**. It does not prove that a voice property **causes** an emotion.

---

## Chapter 9 — The big test: changing schools (cross-corpus, Experiment E2, H1)

### 9.1 The basic idea: domain shift
A model trained in one setting (microphone, room, actors, accents) meets a different setting. Even if the emotions are the same, the *numbers* shift. This is called **domain shift**. Your PCA plot (a way to squash many features into a 2-D picture) showed the two corpora sitting in **different places**, for both feature types.

### 9.2 Two ways to scale the features
- **Source normalisation (strict):** scale everything using only the training corpus's statistics.
- **Per-corpus normalisation:** scale each corpus with **its own** mean and SD. This uses the target's *unlabelled* numbers, never its labels, so it is not cheating. It is a known trick from cross-corpus research (Schuller et al., 2010).

### 9.3 Results (Logistic Regression, UAR on the *other* corpus)
| Direction | eGeMAPS | WavLM | Fusion |
|---|---|---|---|
| RAVDESS → CREMA-D (source) | 30.1 | 38.7 | 42.1 |
| RAVDESS → CREMA-D (per-corpus) | 35.6 | 45.0 | 48.6 |
| CREMA-D → RAVDESS (source) | 30.0 | 58.8 | 66.3 |
| CREMA-D → RAVDESS (per-corpus) | 43.2 | 67.8 | 68.3 |

### 9.4 What it means
1. **Everyone drops 20–35 points.** Inside-one-dataset numbers overstate real-world performance.
2. **H1 depends on direction.**
   - Trained on small RAVDESS (24 actors): WavLM loses **relatively more** than eGeMAPS (+6.1 points). H1 holds.
   - Trained on big, diverse CREMA-D (91 actors): WavLM loses **much less** (−18.4). H1 fails.
   - *Story:* a powerful brain trained in a tiny school memorises that school's style; trained in a big, diverse school, it learns what really matters. **Diversity of training data matters more than the type of features.**
3. WavLM is still better in **absolute** terms in every case.
4. Per-corpus normalisation recovered **+2 to +13 points**, so part of the gap is just a shift in levels (e.g. recording volume).

### 9.5 Confidence intervals: how sure are we?
**Bootstrap:** pretend the test speakers are a sample, then resample them with replacement 1,000 times and recompute UAR each time. The middle 95% of those results is the **95% confidence interval (CI)**. You resampled **speakers, not clips**, because one speaker's clips are similar to each other. If a CI for a difference doesn't include 0, the difference is **significant** (unlikely to be luck).

---

## Chapter 10 — Are the explanations stable? (Experiment E3, H2)

### 10.1 The basic idea: comparing two rankings
Two judges rank 88 features by importance. **Spearman correlation** measures how similar their rankings are: 1 = identical, 0 = unrelated, −1 = opposite.

### 10.2 What you compared
Pairs of models trained on:
- **two resamples** of the same data,
- **two different groups of speakers** from the same corpus,
- **two different corpora**,
- and **two random rankings** as the "no agreement" baseline.

### 10.3 Results (median Spearman)
| | Resample | Speakers | Corpora | Random |
|---|---|---|---|---|
| LogReg · eGeMAPS | 0.78 | 0.63 | 0.50 | 0.00 |
| LightGBM · eGeMAPS | 0.76 | 0.44 | 0.20 | 0.00 |
| LogReg · WavLM | 0.68 | 0.34 | 0.10 | 0.00 |

- **H2 supported:** stability drops in the predicted order, but always stays above random (statistically tested with Mann–Whitney, p < 10⁻¹⁰).
- **Explanations are never fully stable**, even on the same data (as Paper 10 warned).
- **Trade-off:** the simple interpretable model has the most stable explanations; the most accurate one (WavLM) has the least stable.
- **Feature *groups* are stable** even when single features aren't, so report "Energy matters most" rather than "feature #37 matters most".

*Memory hook: "Same class > same school > different school > random."*

---

## Chapter 11 — Can stable features transfer better? (Experiment E4, H3)

### The idea
If some features are **consistently** important (across 20 models trained on different halves of the speakers), maybe they are the "true" emotion cues and will transfer better. You picked this **stable core** (k = 10, 20, 30 features) and compared it with the usual **top-k most important** features.

### Result: H3 inconclusive
Across 12 settings: **1 significant gain** (+5.0), **4 significant losses**, **7 no difference**. The two lists shared 70–87% of features anyway.

**Lesson:** stability is useful for **diagnosing** models, not as a recipe for choosing features. A failed hypothesis is still a result, and reporting it honestly is good science.

---

## Chapter 12 — Where and for whom does it fail? (Experiment E5)

- **Neutral collapses** from RAVDESS to CREMA-D (recall **0.09**): RAVDESS "neutral" is calm and controlled, CREMA-D's isn't. *Same word, different acting.*
- **Sad and happy** break in the other direction.
- **Gender:** WavLM was 4–7 points better for female speakers within a corpus; the gap vanished across corpora. With few actors and balanced folds, you call it an **observation**, not proven bias.
- **Label quality:** on CREMA-D clips where listeners agreed with the label, UAR was **84%** vs **67%** on the rest. Per emotion, human difficulty and model difficulty correlate (0.60). **What humans find hard, the model finds hard.**

---

## Chapter 13 — Standard health checks (Experiment E6)

- **Overfitting gap:** training UAR vs new-speaker UAR. LogReg fusion: 99.7 → 82.0 (RAVDESS), 86.6 → 74.1 (CREMA-D). **LightGBM memorised** (100% on training) without doing better on new speakers, another reason LogReg is the main model.
- **Learning curve:** train on 9, 18, 36, 54, 72 speakers. WavLM keeps improving (67 → 75%); eGeMAPS levels off around 51%. **More speakers would help WavLM further.**
- **ROC-AUC and calibration:** see Chapter 7.

---

## Chapter 14 — The final model and the app (Phase 9)

### Final model
- Trained on **all 8,498 clips of 115 speakers** (both corpora), because diversity helps (Chapter 9).
- **Predictor:** LogReg on fusion (WavLM layer 7, C = 0.01).
- **Explainer:** LogReg on eGeMAPS, for readable SHAP.
- **Honest check before trusting it:** the same recipe with fold-0 speakers held out scored 72.5% (RAVDESS) and 72.3% (CREMA-D). Pooling cost RAVDESS ~2.5 points but gives one model that knows both styles.
- Saved with **joblib** to `models/`, which is never uploaded to GitHub.

### The app (Streamlit)
Streamlit turns Python into a web app. Yours:
1. Takes a **dataset clip, an uploaded file or your microphone**.
2. Runs the **same preprocessing and features** as training (verified identical).
3. Shows the **prediction and all 6 scores** in a console panel, three **trust notes**, and a **SHAP explanation** ("Why" tab) for any emotion.
4. Adds interactive tools: an **emotion compass**, a **what-if studio** (move one voice feature, watch the answer change), a **two-dataset lens** (models trained on RAVDESS, CREMA-D or both judge the same clip), a **timeline** and a **voice profile**.
5. Shows **waveform, spectrogram, Mel-spectrogram, pitch** ("Signal" tab).
6. Runs a **stress test**: quieter, louder, pitch-shifted and noisy copies. Does the prediction survive? (Volume tests the loudness reliance SHAP found.)

**Honesty features:**
- Dataset demos use a model trained **without those speakers**, so the test is honest.
- Uploads and microphone recordings get an **out-of-domain warning**.
- A **typicality check** ranks how unusual the clip is versus all training clips.
- **Disagreement** between the two models is shown.
- Probabilities are called **"model score"**, not certainty.

---

## Chapter 15 — Your tools

| Tool | What it's for |
|---|---|
| **Python** | the programming language for everything |
| **VS Code** | editor where you open and run code |
| **Jupyter notebooks** | step-by-step code + outputs (notebooks 01–07, one per stage) |
| **venv (.venv)** | a private box of Python libraries for this project, so versions don't clash |
| **Git** | saves versions of your code (commits) |
| **GitHub** | online copy of your repository, for sharing with sir and teammates |
| **.gitignore** | list of things never uploaded: audio, features, models, secrets |
| **scikit-learn** | models, pipelines, cross-validation, metrics |
| **LightGBM** | boosted trees |
| **openSMILE** | eGeMAPS features |
| **transformers (Hugging Face)** | loads WavLM |
| **SHAP** | explanations |
| **Streamlit + Plotly** | the app and its interactive charts |

---

## Chapter 16 — The whole story in 12 lines (memorise this)

1. **Goal:** recognise 6 acted emotions from voice, explain them, and test if both survive a new dataset.
2. **Data:** RAVDESS (24 actors) and CREMA-D (91 actors), harmonised to 6 emotions.
3. **Prep:** mono, 16 kHz, trim silence, keep loudness.
4. **Features:** eGeMAPS (88 named) vs WavLM (768 learned, middle layers best) vs fusion.
5. **Golden rule:** speaker-independent 5-fold CV with nested tuning. Leakage inflated RAVDESS by up to 9 points (H0 ✔).
6. **Models:** LogReg (main), SVM, LightGBM, MLP; no KNN.
7. **Inside one corpus:** fusion LogReg 82.2% / 74.1% UAR; WavLM ≫ eGeMAPS in every fold; well calibrated.
8. **XAI:** exact SHAP + permutation check. CREMA-D model = loudness, RAVDESS = spectral shape; Energy best per feature. **SHAP found a bug** that accuracy hid.
9. **Across corpora:** everyone drops 20–35 points. H1 depends on direction: **training diversity matters most**. Per-corpus normalisation helps.
10. **Explanation stability:** 0.78 > 0.63 > 0.50 > 0 (H2 ✔); WavLM least stable; groups stable.
11. **Stable-core selection:** didn't reliably help (H3 inconclusive). Errors: neutral collapses, quiet → sad, human-hard = model-hard.
12. **App:** real model, explanation, robustness test, honest warnings.

---

## Chapter 17 — Test yourself

Answer without looking, then check the chapter in brackets.

1. Why is 16.7% important? *(Ch 1)*
2. What is the difference between eGeMAPS and WavLM? *(Ch 4)*
3. What is leakage, and how did you prevent it? *(Ch 5)*
4. Why did leakage hurt RAVDESS more than CREMA-D? *(Ch 5.6)*
5. What is nested cross-validation for? *(Ch 5.5)*
6. Why is UAR better than accuracy here? *(Ch 7)*
7. Explain SHAP with the prize-sharing example. *(Ch 8.1)*
8. What is the linear SHAP formula? *(Ch 8.2)*
9. Tell the clipping-bug story in 30 seconds. *(Ch 8.7)*
10. Why did WavLM transfer well from CREMA-D but not from RAVDESS? *(Ch 9.4)*
11. Is per-corpus normalisation cheating? Why not? *(Ch 9.2)*
12. What does a Spearman of 0.50 between corpora mean? *(Ch 10)*
13. Why was H3 still worth reporting? *(Ch 11)*
14. Why does *fearful* get confused with *sad*? *(Ch 7, 8.6)*
15. Why no KNN? Give two reasons. *(Ch 6.5)*
16. How does the app avoid misleading users? *(Ch 14)*

Next step after this guide: `report/10_viva_preparation.md` (47 examiner-style questions).
