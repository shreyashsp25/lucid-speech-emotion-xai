# Paper 02 — CREMA-D

**Citation:** Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., & Verma, R. (2014). CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset. *IEEE Transactions on Affective Computing*, 5(4), 377–390. https://doi.org/10.1109/TAFFC.2014.2336244

**Free PDF:** https://pmc.ncbi.nlm.nih.gov/articles/PMC4313618 (PubMed Central author manuscript)
**Dataset download:** https://github.com/CheyneyComputerScience/CREMA-D (requires Git LFS) — Open Database License (ODbL)
**Type:** Dataset paper

---

## What they did (plain language)
They recorded 91 actors of different ages and ethnicities saying 12 sentences in 6 emotions, then used crowd-sourcing to have about 10 people rate each clip — separately for audio-only, video-only and audio+video.

## Dataset design
| Item | Value |
|---|---|
| Actors | 91 (48 M, 43 F), aged 20–74 (mean 36); 61 Caucasian, 22 African American, 7 Asian, 1 unspecified; 10 Hispanic |
| Sentences | 12 |
| Emotions | 6: anger, disgust, fear, happy, neutral, sad (no surprise) |
| Intensity | Only sentence 1 has low/medium/high; other 11 "unspecified" |
| Clips | **7,442** (20 of 7,462 lost) |
| Recording | Sound-attenuated room, green screen, far-field directional microphone |
| Source audio | AAC 48 kHz inside the video files (the repo also provides WAV for processing — sample rate to verify in Phase 3) |

## Rating procedure
- 2,443 crowd raters; each clip rated in 3 modalities; mean **9.8 ratings per clip** (range 4–12).
- **Human accuracy: audio-only 40.9%**, visual-only 58.2%, audio-visual 63.6%.
- Krippendorff's α = 0.42 overall (low agreement); 0.79 on clips where ≥80% agreed.
- Recognition order: neutral > happy > anger > disgust > fear > sad. Main confusions: anger→disgust, disgust→sad/fear, fear↔sad.
- Audio-only: only **41% of clips "matching"** (majority agrees with intended label), 46% non-matching, 13% ambiguous.

## Preprocessing / Features / Models / XAI
Not Reported — no ML experiments in the sections read.

## Evaluation protocol
Perceptual validation only.

## Limitations (evident from the paper)
- Low inter-rater agreement for audio-only → **many intended labels are not what listeners perceive**.
- Intensity only controlled for 1 of 12 sentences.
- Recording via far-field mic → more variable quality than RAVDESS.
- Acted speech.

## Relevance to Team Lucid
- Our second core dataset; **91 speakers** makes speaker-independent evaluation much more stable than RAVDESS.
- Human audio-only accuracy of **~41%** means a model at 55–65% is *not* "bad" — must be framed against this.
- Ships both *intended* labels and *perceived* (crowd-vote) labels → a design choice for us: train on intended or on agreed-upon clips?
- Big demographic diversity → useful for checking whether our explanations depend on speaker group.

## Viva-ready points
- "Why is CREMA-D accuracy lower than RAVDESS?" → Harder data: more speakers, far-field audio, and humans only reach ~41% audio-only.
- "Which label do you use?" → Must decide (Phase 3) and justify.
