"""
Build the extra model the app needs for honest live demos.

    python -m src.app_assets

The final model (models/final_models.joblib) was trained on EVERY clip, so a dataset
clip played in the app would be one the model has already heard. This script trains
the SAME recipe (src.predict.train_final) with all speakers of fold 0 held out and saves
it as models/heldout_fold0.joblib. In the app, "Dataset example" mode uses this model on
fold-0 clips only, so the true label is known AND the speaker is new to the model.
"""
from __future__ import annotations

import numpy as np

from src.models import load_corpus
from src.predict import train_final, save_final
from src.app_utils import HELDOUT_PATH

ARRAYS = ["y", "groups", "folds", "egemaps", "wavlm"]


def build_heldout(fold: int = 0):
    corpora = {c: load_corpus(c) for c in ["ravdess", "cremad"]}
    train = {}
    for c, d in corpora.items():
        keep = np.asarray(d["folds"]) != fold
        train[c] = {k: np.asarray(d[k])[keep] for k in ARRAYS}
        train[c]["egemaps_names"] = d["egemaps_names"]
    bundle = train_final(train)
    bundle["info"]["trained_on"] = f"RAVDESS + CREMA-D, all speakers EXCEPT fold {fold}"
    bundle["heldout_fold"] = fold
    held = {c: sorted(set(np.asarray(d["groups"])[np.asarray(d["folds"]) == fold])) for c, d in corpora.items()}
    bundle["info"]["heldout_speakers"] = held
    save_final(bundle, HELDOUT_PATH)
    print("saved", HELDOUT_PATH, "| held-out speakers:", {c: len(v) for c, v in held.items()})


if __name__ == "__main__":
    build_heldout()
