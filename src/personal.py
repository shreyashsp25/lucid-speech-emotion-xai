"""
"That's wrong, I was ...": teach the model your own voice with a few corrected recordings.

Why. The six training emotions were acted in studios. Some real voices never appear there:
excited shouting, for example, has the high pitch and loudness of the actors' *angry* clips,
and the model has no "excited" label at all. Calibration (src/calibration.py) moves the
reference point to the user's normal voice; it cannot teach the model what the user's
excitement sounds like. A correction can.

How. When the user says a recording was really emotion e, the recording (as a difference from
the user's calibrated normal voice) is added to the relative model's training data with label e
and a large weight (WEIGHT x an average training clip), and the final logistic-regression layer
is re-fitted, starting from the shared model's weights. The preprocessing (scaling, clipping)
stays fixed, so a re-fit takes a few seconds.

Only the calibrated (relative) model learns. When the original model learned too, one
correction pulled ALL of a new speaker's recordings towards that emotion (they sit together, far
from the training data), and accuracy fell from 52% to 42% in the RAVDESS -> CREMA-D test.
Relative features remove that per-speaker offset, so the correction teaches the *difference*.

Corrections stay in the browser session; nothing is written to disk.
Tested by simulated correction on held-out speakers in report/09_simulation.md (section 8).

Enrollment (October 2026, section 10). When the user has taught at least one recording of EVERY
emotion ("say each emotion once"), the original model is re-fitted as well. The failure above came
from single corrections of one emotion; a balanced set of six cannot pull everything towards one
emotion. This is speaker adaptation with a few labelled examples. On held-out speakers, one
recording per emotion raised UAR in every test (studio, simulated laptop, phone and far room, and
across data sets). The original model's training data, needed for the re-fit, is stored once in
preprocessed form (an "anchor" file next to the model), so the re-fit takes a few seconds.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models import CLASSES

WEIGHT = 100         # one correction counts as much as 100 average training clips (chosen on fold-1 speakers)


class Personaliser:
    """Holds a fitted pipeline's training data in its preprocessed form, ready for quick re-fits."""

    def __init__(self, pipeline: Pipeline, X_train: np.ndarray | None, y_train: np.ndarray, Z: np.ndarray | None = None):
        self.pipeline = pipeline
        self.pre = pipeline[:-1]
        self.Z = self.pre.transform(X_train) if Z is None else np.asarray(Z, dtype=np.float64)
        self.y = np.asarray(y_train)
        counts = np.bincount(self.y, minlength=len(CLASSES))
        self.class_w = len(self.y) / (len(CLASSES) * np.maximum(counts, 1))   # same as class_weight="balanced"

    def fit(self, X_personal: np.ndarray, y_personal: np.ndarray, weight: float = WEIGHT) -> Pipeline:
        """A copy of the pipeline whose last layer has also learned from the personal examples."""
        if len(y_personal) == 0:
            return self.pipeline
        base = self.pipeline[-1]
        clf = LogisticRegression(C=base.C, max_iter=300, warm_start=True)
        clf.coef_, clf.intercept_, clf.classes_ = base.coef_.copy(), base.intercept_.copy(), base.classes_
        sw = np.concatenate([self.class_w[self.y], np.full(len(y_personal), weight * self.class_w.mean())])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(np.vstack([self.Z, self.pre.transform(np.asarray(X_personal))]),
                    np.concatenate([self.y, np.asarray(y_personal)]), sample_weight=sw)
        return Pipeline(list(self.pre.steps) + [("clf", clf)])


def save_anchor(path, pipeline: Pipeline, X_train: np.ndarray, y_train: np.ndarray):
    """Store a model's training data in its preprocessed form, so it can be re-fitted without the data."""
    Z = pipeline[:-1].transform(X_train).astype(np.float16)
    np.savez_compressed(path, Z=Z, y=np.asarray(y_train, dtype=np.int8))


def from_anchor(pipeline: Pipeline, path) -> Personaliser:
    d = np.load(path)
    return Personaliser(pipeline, None, d["y"].astype(int), Z=d["Z"])


def covers_all(labels) -> bool:
    """True when every emotion has at least one example (enrollment: the original model may learn too)."""
    return set(CLASSES) <= set(labels)
