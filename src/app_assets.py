"""
Build the extra models the app needs.

    python -m src.app_assets            # held-out demo model (about 3-5 minutes)
    python -m src.app_assets robust     # microphone-robust model for your own recordings
                                        # (needs: python -m src.robust_features train_aug, about 1 hour once)
    python -m src.app_assets anchor     # only the robust model's re-fit file (built by "robust" too)

1. models/heldout_fold0.joblib. The final model (models/final_models.joblib) was trained on EVERY
   clip, so a dataset clip played in the app would be one the model has already heard. This
   trains the SAME recipe (src.predict.train_final) with all speakers of fold 0 held out. In the
   app, "Dataset clip" mode uses this model on fold-0 clips only, so the true label is known AND
   the speaker is new to the model.

2. models/final_models_robust.joblib. The final model's recipe and settings (same WavLM layer and
   C values), trained on every clip TWICE: as recorded, and through a random simulated microphone
   and room (src/augment.py). The app uses it for uploads and recordings. Tested in
   report/09_simulation.md, section 9.

3. models/final_models_robust_anchor.npz. The robust model's training data in preprocessed form
   (float16), so the app can re-fit it to one user's voice ("Teach it your voice": one recording
   of each emotion, src/personal.py) without the simulated-microphone features.
"""
from __future__ import annotations

import sys
import warnings
from datetime import date

import numpy as np

from src.models import load_corpus, preprocess_steps
from src.predict import train_final, save_final, load_final, BUNDLE_PATH
from src.app_utils import HELDOUT_PATH, MODEL_DIR

ARRAYS = ["y", "groups", "folds", "egemaps", "wavlm"]
ROBUST_PATH = MODEL_DIR / "final_models_robust.joblib"
ANCHOR_PATH = MODEL_DIR / "final_models_robust_anchor.npz"


def robust_training_data(L: int):
    """The robust model's training matrix: every clip as recorded and through its simulated channel."""
    from src.voice_experiments import load_condition
    E_clean, E_aug, F, Y = [], [], [], []
    for c in ["ravdess", "cremad"]:
        d = load_corpus(c)
        e_aug, w_aug = load_condition("train_aug", c, d["meta"]["file"])
        E_clean.append(d["egemaps"]); E_aug.append(e_aug)
        F += [np.hstack([d["egemaps"], d["wavlm"][:, L]]), np.hstack([e_aug, w_aug[:, L]])]
        Y += [d["y"], d["y"]]
    return np.vstack(E_clean), np.vstack(E_aug), np.vstack(F), np.concatenate(Y)


def build_anchor():
    from src.personal import save_anchor
    robust = load_final(ROBUST_PATH)
    _, _, F, Y = robust_training_data(int(robust["layer"]))
    save_anchor(ANCHOR_PATH, robust["predictor"], F, Y)
    print("saved", ANCHOR_PATH, "|", len(Y), "training examples")


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


def build_robust():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    final = load_final(BUNDLE_PATH)
    L = int(final["layer"])
    C_pred = final["predictor"].named_steps["clf"].C
    C_expl = final["explainer"].named_steps["clf"].C
    E_clean, E_aug, F, Y = robust_training_data(L)
    E = np.vstack([E_clean, E_aug])
    Y_eg = np.concatenate([np.concatenate([load_corpus(c)["y"] for c in ["ravdess", "cremad"]])] * 2)

    def fit(X, y, C):
        return Pipeline(preprocess_steps() + [("clf", LogisticRegression(C=C, max_iter=3000,
                                                                         class_weight="balanced"))]).fit(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictor, explainer = fit(F, Y, C_pred), fit(E, Y_eg, C_expl)
    lo, hi = np.percentile(E_clean, [1, 99], axis=0)          # typicality stays relative to the real clips
    n_out = ((E_clean < lo) | (E_clean > hi)).sum(axis=1)
    bundle = dict(final)
    bundle.update({
        "predictor": predictor, "explainer": explainer, "layer": L,
        "explainer_train_mean_z": explainer[:-1].transform(E).mean(axis=0),
        "egemaps_train_quantiles": np.stack([lo, hi]), "ood_threshold": float(np.percentile(n_out, 95)),
        "robust": True,
    })
    bundle["info"] = dict(final["info"], trained_on="RAVDESS + CREMA-D, every clip as recorded AND through a "
                                                    "random simulated microphone and room (src/augment.py)",
                          n_clips=int(len(Y)), date=str(date.today()))
    save_final(bundle, ROBUST_PATH)
    print("saved", ROBUST_PATH, "|", len(Y), "training examples")
    build_anchor()


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "heldout"
    {"heldout": build_heldout, "robust": build_robust, "anchor": build_anchor}[what]()
