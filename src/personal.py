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

    def __init__(self, pipeline: Pipeline, X_train: np.ndarray, y_train: np.ndarray):
        self.pipeline = pipeline
        self.pre = pipeline[:-1]
        self.Z = self.pre.transform(X_train)
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
