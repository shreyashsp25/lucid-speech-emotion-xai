"""
Dataset metadata for RAVDESS and CREMA-D.

Builds one tidy table per corpus (one row per audio file) with:
path, corpus, speaker_id, gender, emotion, label (harmonised 6-class),
intensity, sentence, sample_rate, channels, duration_s, read_error.

All later phases read these tables instead of re-parsing filenames, so every
experiment uses the same labels, speakers and folds.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd
import soundfile as sf
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
META_DIR = PROJECT_ROOT / "data" / "metadata"

# The 6 emotions shared by both corpora (Phase 2 decision).
HARMONISED = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]

# --- RAVDESS filename codes (from the official Zenodo description) ---------
# Example: 03-01-06-01-02-01-12.wav
#   modality-channel-emotion-intensity-statement-repetition-actor
RAVDESS_EMOTION = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
RAVDESS_INTENSITY = {"01": "normal", "02": "strong"}
RAVDESS_STATEMENT = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}

# --- CREMA-D filename codes (from the official GitHub README) --------------
# Example: 1001_DFA_ANG_XX.wav   actor_sentence_emotion_level
CREMAD_EMOTION = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fearful",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}
CREMAD_LEVEL = {"LO": "low", "MD": "medium", "HI": "high", "XX": "unspecified"}
CREMAD_SENTENCE = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}

LFS_POINTER_PREFIX = b"version https://git-lfs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def audio_info(path: Path) -> dict:
    """Read the file header only (fast). Never loads the full waveform."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(64)
        if head.startswith(LFS_POINTER_PREFIX):
            return _bad("git-lfs pointer (audio not downloaded)")
        info = sf.info(str(path))
        dur = info.frames / info.samplerate if info.samplerate else 0.0
        return {
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "duration_s": round(dur, 4),
            "subtype": info.subtype,
            "read_error": "",
        }
    except Exception as exc:  # corrupt / unreadable file
        return _bad(str(exc))


def _bad(msg: str) -> dict:
    return {"sample_rate": None, "channels": None, "duration_s": None,
            "subtype": None, "read_error": msg}


def _rel(path: Path) -> str:
    """Store paths relative to the project so the table works on every laptop."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# RAVDESS
# ---------------------------------------------------------------------------
def parse_ravdess_filename(name: str) -> dict:
    parts = Path(name).stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Unexpected RAVDESS filename: {name}")
    modality, channel, emo, inten, stmt, rep, actor = parts
    actor_num = int(actor)
    emotion = RAVDESS_EMOTION[emo]
    return {
        "modality": modality,          # 03 = audio-only
        "vocal_channel": channel,      # 01 = speech, 02 = song
        "emotion": emotion,
        "label": emotion if emotion in HARMONISED else None,
        "intensity": RAVDESS_INTENSITY[inten],
        "sentence": RAVDESS_STATEMENT[stmt],
        "repetition": int(rep),
        "speaker_id": f"ravdess_{actor_num:02d}",
        "gender": "male" if actor_num % 2 == 1 else "female",
    }


def build_ravdess_metadata(root: Path | None = None) -> pd.DataFrame:
    root = Path(root) if root else RAW_DIR / "ravdess"
    files = sorted(root.rglob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No .wav files under {root}. See data/README.md.")
    rows = []
    for f in files:
        meta = parse_ravdess_filename(f.name)
        if meta["modality"] != "03" or meta["vocal_channel"] != "01":
            continue  # keep audio-only speech
        rows.append({"path": _rel(f), "file": f.name, "corpus": "ravdess",
                     **meta, **audio_info(f)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CREMA-D
# ---------------------------------------------------------------------------
def parse_cremad_filename(name: str) -> dict:
    parts = Path(name).stem.split("_")
    if len(parts) != 4:
        raise ValueError(f"Unexpected CREMA-D filename: {name}")
    actor, sent, emo, level = parts
    emotion = CREMAD_EMOTION[emo]
    return {
        "actor_id": int(actor),
        "speaker_id": f"cremad_{int(actor)}",
        "sentence_code": sent,
        "sentence": CREMAD_SENTENCE.get(sent, sent),
        "emotion_code": emo,
        "emotion": emotion,
        "label": emotion,              # all 6 CREMA-D emotions are harmonised
        "intensity": CREMAD_LEVEL.get(level, level),
    }


def build_cremad_metadata(root: Path | None = None) -> pd.DataFrame:
    root = Path(root) if root else RAW_DIR / "cremad"
    audio_dir = root / "AudioWAV"
    files = sorted(audio_dir.glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No .wav files under {audio_dir}. See data/README.md.")
    rows = [{"path": _rel(f), "file": f.name, "corpus": "cremad",
             **parse_cremad_filename(f.name), **audio_info(f)} for f in files]
    df = pd.DataFrame(rows)

    # Demographics (official CSV in the CREMA-D repo)
    demo_path = root / "VideoDemographics.csv"
    if demo_path.exists():
        demo = pd.read_csv(demo_path)
        demo = demo.rename(columns={"ActorID": "actor_id", "Age": "age",
                                    "Sex": "gender", "Race": "race",
                                    "Ethnicity": "ethnicity"})
        demo["gender"] = demo["gender"].str.lower()
        df = df.merge(demo, on="actor_id", how="left")
    else:
        warnings.warn(f"{demo_path} not found - demographics skipped")
        df["gender"] = None

    # Crowd ratings: does the audio-only majority vote match the intended label?
    summary_path = root / "processedResults" / "summaryTable.csv"
    if summary_path.exists():
        summ = pd.read_csv(summary_path)
        if {"FileName", "VoiceVote"}.issubset(summ.columns):
            summ = summ[["FileName", "VoiceVote"]].rename(
                columns={"FileName": "stem", "VoiceVote": "voice_vote"})
            df["stem"] = df["file"].str.replace(".wav", "", regex=False)
            df = df.merge(summ, on="stem", how="left").drop(columns="stem")
            intended = df["emotion_code"].str[0]          # ANG -> A, DIS -> D ...
            df["voice_agrees"] = df["voice_vote"].astype(str).str.upper() == intended
        else:
            warnings.warn(f"Unexpected columns in {summary_path}: {list(summ.columns)}")
    else:
        warnings.warn(f"{summary_path} not found - agreement labels skipped")
    return df


# ---------------------------------------------------------------------------
# Speaker-independent folds
# ---------------------------------------------------------------------------
def add_speaker_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42,
                      label_col: str = "label") -> pd.DataFrame:
    """
    Assign every row to a fold so that ALL clips of a speaker share one fold
    (no speaker in both train and test) while keeping class balance and
    gender balance similar across folds. Rows without a harmonised label get fold = -1.
    """
    out = df.copy()
    out["fold"] = -1
    mask = out[label_col].notna() & (out["read_error"] == "")
    sub = out[mask]
    # Stratify on emotion AND speaker gender, so each fold has a similar
    # male/female mix (otherwise a fold can end up e.g. 4 female / 1 male).
    strat = sub[label_col].astype(str) + "_" + sub["gender"].astype(str)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, test_idx) in enumerate(sgkf.split(sub, strat, sub["speaker_id"])):
        out.loc[sub.index[test_idx], "fold"] = k
    return out


def check_no_speaker_leakage(df: pd.DataFrame) -> bool:
    """True if every speaker appears in exactly one fold."""
    used = df[df["fold"] >= 0]
    return bool((used.groupby("speaker_id")["fold"].nunique() == 1).all())


def save_metadata(df: pd.DataFrame, name: str) -> Path:
    META_DIR.mkdir(parents=True, exist_ok=True)
    out = META_DIR / f"{name}.csv"
    df.to_csv(out, index=False)
    return out
