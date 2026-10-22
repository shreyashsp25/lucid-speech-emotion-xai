"""
Extract eGeMAPS + WavLM features of simulated recording conditions (src/augment.py).

    python -m src.robust_features train_aug          # one random channel per clip, every clip
    python -m src.robust_features laptop phone far_room --fold 0   # fixed test conditions, fold-0 clips

Writes data/processed/robust/<condition>_<corpus>.npz with arrays egemaps (n, 88), wavlm (n, 13, 768)
float16 and files (n,). Resumable: work is saved every 200 clips.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src import augment as A
from src.audio import load_audio
from src.data import PROJECT_ROOT
from src.models import META_DIR

OUT = PROJECT_ROOT / "data" / "processed" / "robust"


def clips(corpus: str, fold: int | None) -> pd.DataFrame:
    meta = pd.read_csv(META_DIR / f"{corpus}.csv")
    meta = meta[meta["label"].notna() & (meta["fold"] >= 0)].reset_index(drop=True)
    return meta if fold is None else meta[meta["fold"] == fold].reset_index(drop=True)


def run(condition: str, corpus: str, fold: int | None, chunk: int = 200):
    from src.predict import extract_one
    meta = clips(corpus, fold)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{condition}_{corpus}.npz"
    done = dict(np.load(path, allow_pickle=True)) if path.exists() else {"files": np.array([], dtype=object),
                                                                       "egemaps": np.zeros((0, 88), np.float32),
                                                                       "wavlm": np.zeros((0, 13, 768), np.float16)}
    have = set(done["files"].tolist())
    todo = meta[~meta["file"].isin(have)]
    print(f"{condition} {corpus}: {len(have)} done, {len(todo)} to do", flush=True)
    F, E, W = list(done["files"]), list(done["egemaps"]), list(done["wavlm"])
    t0 = time.time()
    for i, r in enumerate(todo.itertuples(), 1):
        y, _ = load_audio(r.path)
        z = A.random_channel(y, r.file) if condition == "train_aug" else A.test_condition(y, condition, r.file)
        f = extract_one(z, sr=A.SR)
        F.append(r.file); E.append(f["egemaps"].astype(np.float32)); W.append(f["wavlm"].astype(np.float16))
        if i % chunk == 0 or i == len(todo):
            np.savez(path, files=np.array(F, dtype=object), egemaps=np.stack(E), wavlm=np.stack(W))
            rate = (time.time() - t0) / i
            print(f"  {condition} {corpus} {len(F)}/{len(meta)}  {rate:.2f} s/clip  "
                  f"about {rate * (len(todo) - i) / 60:.0f} min left", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("conditions", nargs="+")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--corpora", nargs="+", default=["ravdess", "cremad"])
    a = ap.parse_args()
    for cond in a.conditions:
        for c in a.corpora:
            run(cond, c, a.fold)
