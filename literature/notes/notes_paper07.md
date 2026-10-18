# Paper 07 — EMO-SUPERB

**Citation:** Wu, H., Chou, H.-C., Chang, K.-W., Goncalves, L., Du, J., Jang, J.-S. R., Lee, C.-C., & Lee, H.-Y. (2024). EMO-SUPERB: An In-depth Look at Speech Emotion Recognition. *arXiv preprint* arXiv:2402.13018. (Venue: TO VERIFY.)

**Free PDF:** https://arxiv.org/pdf/2402.13018
**Type:** Benchmark paper

---

## What they did (plain language)
They built a common test-bench for speech emotion recognition: 15 pretrained speech models tested the same way on 6 datasets, with fixed speaker-independent splits and a public leaderboard. They also used ChatGPT to turn annotators' free-text comments into extra soft labels.

## Datasets
IEMOCAP, **CREMA-D**, MSP-IMPROV, MSP-PODCAST, BIIC-NNIME, BIIC-PODCAST (English + Mandarin). All original emotion labels kept (no merging).

## Preprocessing / partition
**Speaker-independent splits**; in dialogues, both speakers go to the same partition to avoid leakage. Splits published in their appendix.

## Features
15 self-supervised models as **frozen** feature extractors (e.g., wav2vec 2.0, HuBERT, WavLM, XLS-R-1B, DeCoAR 2, APC, TERA); FBANK baseline. Only layer weights + small head trained.

## Models
Lightweight downstream classifier; class-balanced cross-entropy, label smoothing 0.05, AdamW. Soft (multi-label) targets.

## XAI method
Not Reported (only layer-weight analysis).

## Evaluation protocol
Speaker-independent fixed splits; **macro-F1** main metric; single runs.

## Key findings
- All SSL models clearly beat FBANK; XLS-R-1B best on average, WavLM best in 3/9 conditions.
- Smaller DeCoAR 2 (~90M) beats wav2vec 2.0 (~317M) → bigger isn't always better.
- **Shallow/middle layers** carry more emotion information.
- ChatGPT relabelling of free-text annotations → ~3.08% average relative gain (MSP-PODCAST only).

## Limitations (stated)
- Only English and Mandarin; no child/elderly speech.
- Gender bias not analysed.
- No full fine-tuning (compute); single runs, no variance.
- RAVDESS not included.

## Relevance to Team Lucid
- Provides a **reference speaker-independent setup for CREMA-D** with SSL features.
- Confirms that CREMA-D is a standard benchmark and that frozen-feature + small-head is an accepted, compute-friendly design (good for our laptops).
- Shows the field's focus is accuracy; **explainability is absent** from the benchmark.

## Viva-ready points
- "Is freezing a pretrained model and training a small head acceptable?" → Yes, it's the standard SUPERB/EMO-SUPERB protocol.
