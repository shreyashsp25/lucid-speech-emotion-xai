"""
Self-check: is the prediction pipeline on THIS computer working as trained?

    python -m src.selfcheck

Takes 24 held-out fold-0 clips (both datasets, all six emotions), and for each one:
  1. recomputes eGeMAPS and WavLM live, exactly as the app does, and compares them with the
     stored training features (they should be identical up to rounding);
  2. predicts with the held-out demo model (the one "Dataset clip" uses) and with the
     microphone-robust model, from the LIVE features and from the STORED features.

If stored features predict well but live ones do not, feature extraction on this computer is
broken (for example a damaged WavLM download). If both predict badly, a model file is broken.
The report is printed and saved to selfcheck_report.txt in the project folder.
"""
from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT
from src.models import CLASSES, load_corpus

OUT = PROJECT_ROOT / "selfcheck_report.txt"
lines: list[str] = []


def say(s: str = ""):
    print(s, flush=True)
    lines.append(s)


def main(per_class: int = 2):
    import joblib
    import sklearn
    import torch
    import transformers
    from src.predict import extract_one, predict_features, _extractors

    say(f"Lucid self-check  {time.strftime('%Y-%m-%d %H:%M')}")
    say(f"python {platform.python_version()} | {platform.platform()} | numpy {np.__version__} | "
        f"sklearn {sklearn.__version__} | torch {torch.__version__} | transformers {transformers.__version__}")

    models = {}
    for name in ["heldout_fold0", "final_models", "final_models_robust"]:
        p = PROJECT_ROOT / "models" / f"{name}.joblib"
        if p.exists():
            models[name] = joblib.load(p)
            say(f"model {name}: layer {models[name]['layer']}, {p.stat().st_size} bytes")
        else:
            say(f"model {name}: MISSING")

    smile, fe, wavlm, dev = _extractors("cpu")
    say(f"WavLM: {type(wavlm).__name__}, {sum(p.numel() for p in wavlm.parameters())} parameters, "
        f"first weight sum {float(next(wavlm.parameters()).detach().sum()):.6f}")

    rows = []
    for c in ["ravdess", "cremad"]:
        d = load_corpus(c)
        meta = d["meta"]
        for k, emo in enumerate(CLASSES):
            idx = [i for i in np.where((meta["fold"].values == 0) & (d["y"] == k))[0]
                   if (PROJECT_ROOT / meta["path"].iloc[i]).exists()][:per_class]
            for i in idx:
                rows.append((c, emo, meta["path"].iloc[i], d["egemaps"][i], d["wavlm"][i]))
    say(f"clips found: {len(rows)}")
    if not rows:
        say("No fold-0 audio found under data/raw: cannot test live extraction.")
        return

    res = []
    for c, emo, path, eg_s, wl_s in rows:
        f = extract_one(PROJECT_ROOT / path)
        eg_l, wl_l = f["egemaps"], f["wavlm"]
        eg_diff = float(np.max(np.abs(eg_l - eg_s) / (np.abs(eg_s) + 1e-3)))
        wl_corr = float(np.corrcoef(wl_l.ravel(), wl_s.ravel())[0, 1])
        r = {"corpus": c, "true": emo, "file": Path(path).name, "egemaps_maxreldiff": eg_diff, "wavlm_corr": wl_corr}
        for name, b in models.items():
            r[f"{name}_live"] = predict_features(eg_l, wl_l, b)["label"]
            r[f"{name}_stored"] = predict_features(eg_s, wl_s, b)["label"]
        res.append(r)
        say(f"  {c:8s} {emo:8s} eGeMAPS diff {eg_diff:.1e}  WavLM corr {wl_corr:.5f}  "
            + "  ".join(f"{n[:7]}:{r[n + '_live']}/{r[n + '_stored']}" for n in models))
    df = pd.DataFrame(res)

    say()
    say("SUMMARY (live = computed now on this computer, stored = training features)")
    say(f"  eGeMAPS identical to training: {'yes' if df['egemaps_maxreldiff'].max() < 1e-3 else 'NO'} "
        f"(largest relative difference {df['egemaps_maxreldiff'].max():.1e})")
    say(f"  WavLM identical to training:   {'yes' if df['wavlm_corr'].min() > 0.999 else 'NO'} "
        f"(lowest correlation {df['wavlm_corr'].min():.5f})")
    for name in models:
        for kind in ["live", "stored"]:
            col = f"{name}_{kind}"
            acc = (df[col] == df["true"]).mean() * 100
            spread = df[col].value_counts().to_dict()
            say(f"  {name:20s} {kind:6s}: {acc:5.1f}% correct   answers given: {spread}")
    say()
    ok_feat = df["egemaps_maxreldiff"].max() < 1e-3 and df["wavlm_corr"].min() > 0.999
    ho_live = (df["heldout_fold0_live"] == df["true"]).mean() if "heldout_fold0" in models else 0
    if ok_feat and ho_live >= 0.5:
        say("VERDICT: the pipeline works on this computer. Wrong answers on your own voice come from the "
            "recording, not from the model files.")
    elif not ok_feat:
        say("VERDICT: live features differ from the training features. Feature extraction on this computer "
            "is broken (most likely the WavLM download or an audio library).")
    else:
        say("VERDICT: features are fine but the model answers badly. A model file is likely damaged.")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
