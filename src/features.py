"""
Feature extraction: eGeMAPS (interpretable) and WavLM (pretrained embeddings).

Outputs (git-ignored, regenerate with notebooks/02_feature_extraction.ipynb):
  data/processed/egemaps_<corpus>.csv   one row per clip, 88 named columns
  data/processed/wavlm_<corpus>.npy     shape (n_clips, 13 layers, 768), float16
  data/processed/wavlm_<corpus>_files.csv   row order of the .npy file
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.audio import load_audio, TARGET_SR
from src.data import PROJECT_ROOT

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
WAVLM_NAME = "microsoft/wavlm-base-plus"


# ---------------------------------------------------------------------------
# eGeMAPS (88 functionals, openSMILE)
# ---------------------------------------------------------------------------
def egemaps_extractor():
    import opensmile
    return opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                           feature_level=opensmile.FeatureLevel.Functionals)


def extract_egemaps(meta: pd.DataFrame, corpus: str) -> pd.DataFrame:
    smile = egemaps_extractor()
    rows = []
    for path, file in tqdm(zip(meta["path"], meta["file"]), total=len(meta),
                           desc=f"eGeMAPS {corpus}"):
        y, sr = load_audio(path)
        feats = smile.process_signal(y, sr).iloc[0]
        feats.name = file
        rows.append(feats)
    df = pd.DataFrame(rows)
    df.index.name = "file"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / f"egemaps_{corpus}.csv")
    return df


def load_egemaps(corpus: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / f"egemaps_{corpus}.csv", index_col="file")


# ---------------------------------------------------------------------------
# WavLM (frozen; mean-pooled hidden states of every layer)
# ---------------------------------------------------------------------------
def pick_device() -> str:
    import torch
    return "cpu" if not torch.backends.mps.is_available() else "mps"


def load_wavlm(name: str = WAVLM_NAME, device: str = "cpu"):
    from transformers import AutoFeatureExtractor, WavLMModel
    fe = AutoFeatureExtractor.from_pretrained(name)
    model = WavLMModel.from_pretrained(name).eval().to(device)
    return fe, model


def wavlm_embed(y: np.ndarray, fe, model, device: str = "cpu") -> np.ndarray:
    """Return (n_layers + 1, hidden) = mean over time of every hidden state."""
    import torch
    inputs = fe(y, sampling_rate=TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        out = model(inputs.input_values.to(device), output_hidden_states=True)
    hs = torch.stack(out.hidden_states)            # (L+1, 1, T, H)
    return hs.mean(dim=2).squeeze(1).float().cpu().numpy()


def extract_wavlm(meta: pd.DataFrame, corpus: str, device: str = "cpu",
                  name: str = WAVLM_NAME, fe=None, model=None) -> np.ndarray:
    if fe is None or model is None:
        fe, model = load_wavlm(name, device)
    embs = []
    for path in tqdm(meta["path"], desc=f"WavLM {corpus} ({device})"):
        y, _ = load_audio(path)
        embs.append(wavlm_embed(y, fe, model, device).astype(np.float16))
    arr = np.stack(embs)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / f"wavlm_{corpus}.npy", arr)
    meta[["file"]].to_csv(PROCESSED_DIR / f"wavlm_{corpus}_files.csv", index=False)
    return arr


def load_wavlm_features(corpus: str) -> tuple[np.ndarray, list[str]]:
    arr = np.load(PROCESSED_DIR / f"wavlm_{corpus}.npy")
    files = pd.read_csv(PROCESSED_DIR / f"wavlm_{corpus}_files.csv")["file"].tolist()
    return arr, files
