"""
Lucid · Explainable Speech Emotion Recognition — interactive research demo (Phase 9).

Run from the project root:
    streamlit run app/streamlit_app.py

Everything shown is computed by the real trained models (models/*.joblib) and the real
preprocessing pipeline (src/), or read from the saved results of Phases 6-9.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import streamlit as st

from src.audio import load_audio, TARGET_SR
from src.models import CLASSES
from src.predict import BUNDLE_PATH, predict_features, extract_one
ROBUST_PATH = BUNDLE_PATH.parent / "final_models_robust.joblib"   # built by: python -m src.app_assets robust
ANCHOR_PATH = BUNDLE_PATH.parent / "final_models_robust_anchor.npz"   # re-fit file for "Teach it your voice"
from src import app_utils as U
from src import studio as S
from src import calibration as K
from src import personal as PZ
from src.recorder import raw_recorder, processing_note

st.set_page_config(page_title="Lucid voice emotion lab", page_icon="〰️", layout="wide",
                   initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# Visual identity
# ---------------------------------------------------------------------------
INK, PAPER, LINE, TEXT, MUTED = "#1E1B3A", "#EEF0F7", "#D5D9E8", "#23263A", "#5E6480"
SIGNAL, SIGNAL_TEXT, CAUTION = "#2EC4B6", "#0E7F7B", "#C98A00"
EMO = U.EMO_COLOUR
GRP = U.GROUP_COLOUR
CHANCE = 100 / 6

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500..800&family=Instrument+Sans:ital,wght@0,400..700;1,400..700&display=swap');
:root {{ --ink:{INK}; --paper:{PAPER}; --line:{LINE}; --text:{TEXT}; --muted:{MUTED};
         --signal:{SIGNAL}; --signal-text:{SIGNAL_TEXT}; --caution:{CAUTION}; }}
.stApp {{ background-color: var(--paper);
          background-image: radial-gradient(rgba(30,27,58,.085) 1px, transparent 1.3px);
          background-size: 18px 18px; }}
.stApp [data-testid="stHeader"] {{ background: rgba(238,240,247,.86); backdrop-filter: blur(8px);
          border-bottom: 1px solid var(--line); }}
.block-container {{ padding-top: 4.2rem; max-width: 1320px; }}
.stApp p, .stApp li, .stApp label, .stApp td, .stApp th, .stApp input, .stApp textarea,
.stApp button, .stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stCaptionContainer"], .stApp .stMarkdown, .stApp [data-testid="stTab"] p
  {{ font-family: 'Instrument Sans', system-ui, sans-serif; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .lucid-display
  {{ font-family: 'Bricolage Grotesque', 'Instrument Sans', sans-serif; color: var(--text);
     letter-spacing: -0.01em; }}
.masthead {{ margin: 0 0 1.3rem 0; max-width: 62rem; }}
.masthead h1 {{ font-size: 3.15rem; font-weight: 780; line-height: 1.02; letter-spacing: -0.025em;
               margin: 0 0 .7rem 0; }}
.spectrum {{ display: flex; gap: 4px; width: 15rem; max-width: 100%; margin: 0 0 .9rem 0; }}
.spectrum span {{ flex: 1; height: 7px; border-radius: 99px; }}
.masthead p {{ color: var(--muted); font-size: 1.06rem; line-height: 1.5; margin: 0; max-width: 46rem; }}
.console {{ --glow: var(--emo, {SIGNAL}); position: relative; overflow: hidden; color: #EDEFFA;
            border-radius: 28px; padding: 1.9rem 2.1rem;
            background: radial-gradient(90% 130% at 0% 0%, color-mix(in srgb, var(--glow) 34%, transparent), transparent 62%),
                        var(--ink);
            box-shadow: 0 28px 50px -30px rgba(30,27,58,.75), inset 0 0 0 1px rgba(255,255,255,.07);
            display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(0, 1fr); gap: 2.2rem;
            align-items: center; margin: .6rem 0 1.1rem 0; }}
.console .heard {{ color: #AEB3D6; font-size: .98rem; margin-bottom: .1rem; }}
.console .verdict {{ display: flex; align-items: center; gap: 1.2rem; }}
.console .word {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 4.3rem;
                  line-height: .95; letter-spacing: -0.03em; margin: .1rem 0 .35rem 0; }}
.console .ring {{ color: #EDEFFA; flex: 0 0 auto; }}
.console .tags {{ display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0 1rem 0; }}
.tag {{ display: inline-block; padding: .18rem .7rem; border-radius: 999px; font-size: .86rem;
        font-weight: 700; border: 1px solid rgba(255,255,255,.22); color: #EDEFFA; }}
.tag.good {{ background: rgba(46,196,182,.16); border-color: rgba(46,196,182,.55); color: #8FF0E6; }}
.tag.bad {{ background: rgba(255,120,120,.14); border-color: rgba(255,140,140,.5); color: #FFB3B3; }}
.bars {{ display: grid; grid-template-columns: 5.6rem 1fr 2.6rem; gap: .32rem .7rem; align-items: center;
         font-size: .9rem; }}
.bars .name {{ color: #C9CDEA; }}
.bars .track {{ height: .55rem; background: rgba(255,255,255,.09); border-radius: 99px; overflow: hidden; }}
.bars .fill {{ height: 100%; border-radius: 99px; }}
.bars .pct {{ text-align: right; color: #C9CDEA; font-variant-numeric: tabular-nums; }}
.console .trace {{ color: var(--signal); }}
.console .trace-cap {{ color: #AEB3D6; font-size: .88rem; margin-top: .5rem; line-height: 1.45; }}
.notes {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .9rem; margin-bottom: 1.2rem; }}
.note {{ background: #fff; border: 1px solid var(--line); border-left: 5px solid var(--signal);
         border-radius: 12px; padding: .8rem 1rem; }}
.note.caution {{ border-left-color: var(--caution); }}
.note h4 {{ font-size: 1rem; margin: 0 0 .25rem 0; }}
.note p {{ color: var(--muted); font-size: .92rem; line-height: 1.45; margin: 0; }}
.foot {{ color: var(--muted); font-size: .84rem; border-top: 1px solid var(--line); margin-top: 2.4rem;
         padding-top: .8rem; }}
.console::after {{ content: ""; position: absolute; inset: 0; pointer-events: none;
            background: repeating-radial-gradient(circle at 100% 100%, transparent 0 26px, rgba(255,255,255,.04) 26px 27px); }}
.console > div {{ position: relative; z-index: 1; }}
.console .word {{ font-size: 5rem; text-shadow: 0 0 46px color-mix(in srgb, var(--glow) 60%, transparent); }}
.console .heard {{ font-size: 1.02rem; }}
.console .trace svg {{ height: 150px; }}
.bars {{ gap: .45rem .8rem; }}
.bars .track {{ height: .7rem; }}
.bars .fill {{ animation: grow .8s cubic-bezier(.2,.8,.2,1) both; }}
@keyframes grow {{ from {{ width: 0; }} }}
.tag {{ padding: .22rem .8rem; background: rgba(255,255,255,.05); }}
.note {{ border-radius: 16px; padding: .95rem 1.1rem; border-left-width: 6px;
         box-shadow: 0 14px 26px -22px rgba(30,27,58,.55); }}
.note h4 {{ font-size: 1.06rem; }}
.stApp [data-testid^="stBaseButton"] {{ border-radius: 999px; font-weight: 600; }}
.stApp [data-testid="stBaseButton-primary"] {{ background: var(--ink); border-color: var(--ink); color: #fff; }}
.stApp [data-testid="stBaseButton-primary"]:hover {{ background: var(--signal-text); border-color: var(--signal-text); }}
.stApp [data-baseweb="tab"] p {{ font-size: 1.04rem; font-weight: 600; }}
.stApp [data-baseweb="tab"][aria-selected="true"] p {{ color: var(--ink); }}
.stApp [data-baseweb="tab-highlight"] {{ background-color: var(--ink); height: 3px; border-radius: 3px; }}
.stApp [data-testid="stExpander"] details {{ background: #fff; border: 1px solid var(--line); border-radius: 16px; }}
.stApp [data-baseweb="select"] > div {{ background: #fff; border-radius: 12px; }}
.stApp [data-testid="stPlotlyChart"] {{ background: #fff; border: 1px solid var(--line); border-radius: 18px;
         padding: .5rem .6rem 0 .6rem; box-shadow: 0 14px 26px -24px rgba(30,27,58,.5); }}
.stApp [data-testid="stMetric"] {{ background: #fff; border: 1px solid var(--line); border-radius: 16px;
         padding: .8rem 1rem; }}
.stApp [data-testid="stAlert"] {{ border-radius: 14px; }}
.stApp audio {{ width: 100%; }}
.stTabs [data-baseweb="tab-list"] {{ gap: .6rem; }}
@media (max-width: 900px) {{ .console {{ grid-template-columns: 1fr; }} .notes {{ grid-template-columns: 1fr; }}
                            .console .word {{ font-size: 3.4rem; }}
                            .masthead h1 {{ font-size: 2.3rem; }} .console {{ padding: 1.4rem 1.3rem; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
""", unsafe_allow_html=True)

pio.templates["lucid"] = go.layout.Template(layout=dict(
    font=dict(family="Instrument Sans, system-ui, sans-serif", size=13, color=TEXT),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor="#E3E7F0", zerolinecolor="#C9CEDD", linecolor="#C9CEDD"),
    yaxis=dict(gridcolor="#E3E7F0", zerolinecolor="#C9CEDD", linecolor="#C9CEDD"),
    colorway=["#2A78D6", "#EB6834", "#1BAF7A", "#7B5BD6", "#8A8F98", "#E9A100"],
    hoverlabel=dict(font_family="Instrument Sans")))
pio.templates.default = "lucid"


def style(fig, h=360, legend=True, title=None, bottom=False):
    """Shared chart look. Title at the very top, legend under it (or below the plot for facet charts)."""
    t = 12 + (34 if title else 0) + (30 if legend and not bottom else 0)
    fig.update_layout(
        height=h, showlegend=legend, margin=dict(l=10, r=10, t=t, b=10 + (70 if legend and bottom else 0)),
        legend=dict(orientation="h", x=0, xanchor="left", title_text="",
                    **({"y": -0.2, "yanchor": "top"} if bottom else {"y": 1.01, "yanchor": "bottom"})))
    if title:
        fig.update_layout(title=dict(text=title, x=0, xanchor="left", y=1, yanchor="top", yref="container",
                                     pad=dict(t=10, l=8), font=dict(size=16, family="Bricolage Grotesque, sans-serif")))
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def masthead(title: str, lede: str):
    strip = "".join(f'<span style="background:{EMO[c]}" title="{c}"></span>' for c in CLASSES)
    st.markdown(f'<div class="masthead"><h1>{title}</h1><div class="spectrum" role="img" '
                f'aria-label="The six emotions the model knows">{strip}</div><p>{lede}</p></div>',
                unsafe_allow_html=True)


def bars_html(proba: dict, highlight: str | None = None) -> str:
    rows = []
    for c in CLASSES:
        v = float(proba[c])
        colour = EMO[c] if (highlight is None or c == highlight) else "rgba(160,166,200,.55)"
        rows.append(f'<div class="name">{c}</div><div class="track"><div class="fill" '
                    f'style="width:{v * 100:.1f}%;background:{colour}"></div></div><div class="pct">{v:.0%}</div>')
    return '<div class="bars">' + "".join(rows) + "</div>"


def footer():
    st.markdown('<div class="foot">Team Lucid: Pranav Sajith, Piriya Shreyash Shanmukh and Praveen Raj. '
                'Research demo trained on acted English speech (RAVDESS and CREMA-D). Model scores are not a '
                'measurement of how anyone feels.</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the trained model…")
def get_bundle(path: str):
    return joblib.load(path)


@st.cache_resource(show_spinner="Loading openSMILE and WavLM (first time only, about 20 s)…")
def warm_extractors():
    from src.predict import _extractors
    return _extractors("cpu")


@st.cache_data(show_spinner=False)
def train_egemaps():
    return U.training_egemaps()


@st.cache_data(show_spinner=False)
def ood_reference():
    """Per-clip count of features outside the 1-99% training range, for every training clip."""
    T = train_egemaps().values
    lo, hi = np.percentile(T, [1, 99], axis=0)
    return lo, hi, ((T < lo) | (T > hi)).sum(axis=1)


@st.cache_data(show_spinner=False)
def features_for(y: np.ndarray) -> dict:
    f = extract_one(y, sr=TARGET_SR)
    return {"egemaps": f["egemaps"], "wavlm": f["wavlm"]}


@st.cache_data(show_spinner=False)
def views_for(y: np.ndarray) -> dict:
    return U.signal_views(y)


@st.cache_data(show_spinner=False)
def examples():
    return U.heldout_examples(0)


@st.cache_resource(show_spinner="Preparing the voice-calibration model (first time only, about 10 seconds)…")
def calib_model(layer: int, heldout_fold):
    return K.load_or_train(layer, heldout_fold)


@st.cache_resource(show_spinner="Getting ready to learn your voice (first time only, about 10 seconds)…")
def personaliser_rel(layer: int, heldout_fold=None):
    X, y, _ = K.relative_training_data(layer, heldout_fold)
    return PZ.Personaliser(calib_model(layer, heldout_fold)["predictor"], X, y)


@st.cache_resource(show_spinner="Getting ready to adapt the main model (first time only, about 20 seconds)…")
def personaliser_abs(bundle_path: str):
    """Re-fitter for the original model. The robust model uses its stored anchor file; the studio models
    re-use the stored training features (all clips, or all clips except the held-out fold)."""
    b = get_bundle(bundle_path)
    if b.get("robust"):
        return PZ.from_anchor(b["predictor"], ANCHOR_PATH) if ANCHOR_PATH.exists() else None
    from src.models import load_corpus
    L, fold = int(b["layer"]), b.get("heldout_fold")
    X, y = [], []
    for c in ["ravdess", "cremad"]:
        d = load_corpus(c)
        keep = np.ones(len(d["y"]), bool) if fold is None else np.asarray(d["folds"]) != fold
        X.append(np.hstack([np.asarray(d["egemaps"])[keep], np.asarray(d["wavlm"])[keep][:, L]]))
        y.append(np.asarray(d["y"])[keep])
    return PZ.Personaliser(b["predictor"], np.vstack(X), np.concatenate(y))


TEACH_OPTIONS = ["neutral", "happy / excited", "sad", "angry", "fearful", "disgust"]


def personal_model(layer: int, base, items: list, bundle_path: str, heldout_fold=None):
    """Models re-fitted with labelled examples of one speaker's voice: (original, relative); None = unchanged.

    The calibrated (relative) model learns from any examples. The original model learns only when every
    emotion has an example (enrollment): single corrections of one emotion pulled ALL of a new speaker's
    recordings towards it (report/09_simulation.md, section 8); a balanced set of six did not (section 10).
    """
    if not items or base is None:
        return None, None
    key = (bundle_path, heldout_fold, tuple(sorted((k, v["label"], str(v.get("nonce", ""))) for k, v in items)),
           float(np.sum(base["egemaps"])))
    fits = st.session_state.setdefault("teach_fits", {})
    if key in fits:
        return fits[key]
    E = [v for _, v in items]
    labels = [e["label"] for e in E]
    with st.spinner(f"Learning from {len(E)} example{'s' if len(E) != 1 else ''} of this voice…"):
        yp = np.array([CLASSES.index(l) for l in labels])
        Xr = np.array([np.hstack([e["egemaps"] - base["egemaps"], e["wavlm"][layer] - base["wavlm"][layer]]) for e in E])
        pR = personaliser_rel(layer, heldout_fold).fit(Xr, yp)
        pA = None
        if PZ.covers_all(labels):
            P = personaliser_abs(bundle_path)
            if P is not None:
                pA = P.fit(np.array([np.hstack([e["egemaps"], e["wavlm"][layer]]) for e in E]), yp)
    if len(fits) > 8:
        fits.clear()
    fits[key] = (pA, pR)
    return pA, pR


def teach_panel(clip_id: str, feats: dict, label: str, calibrated: bool):
    """'That's wrong, I was ...': save this recording as an example of the user's own voice."""
    ex = st.session_state.setdefault("teach", {})
    with st.container(border=True):
        if not calibrated:
            st.markdown("**Wrong answer?** Calibrate to your voice first (Step 1 above). Then you can tell it what "
                        "you were really feeling whenever it gets you wrong, and it learns your voice.")
            return
        if clip_id in ex:
            st.markdown(f"**Saved: this recording is {ex[clip_id]['label']}.** The answer above may not change yet, "
                        "because the original model still has a say. Corrections work over your next recordings, "
                        "and two or three of the same feeling work much better than one.")
        else:
            st.markdown(f"**Wrong answer?** If **{label}** is not what you were feeling, tell it what you were "
                        "really feeling. It learns from this recording and uses that for your next recordings.")
        c1, c2 = st.columns([3.2, 1])
        pick = c1.segmented_control("I was actually", TEACH_OPTIONS, key=f"teach_pick_{clip_id}",
                                    label_visibility="collapsed")
        if c2.button("Teach it", disabled=pick is None, width="stretch", type="primary"):
            ex[clip_id] = {"egemaps": feats["egemaps"], "wavlm": feats["wavlm"], "label": pick.split(" /")[0]}
            st.rerun()
        if ex:
            counts = pd.Series([v["label"] for v in ex.values()]).value_counts()
            c1.caption(f"{len(ex)} correction{'s' if len(ex) != 1 else ''} saved for this session: "
                       + ", ".join(f"{k} × {v}" for k, v in counts.items())
                       + ". The model has no 'excited' label, so excitement is learned as happy.")
            if c2.button("Forget corrections", width="stretch"):
                st.session_state["teach"] = {}
                st.session_state.pop("teach_fits", None)
                st.session_state["enrol_gen"] = st.session_state.get("enrol_gen", 0) + 1
                st.rerun()


def wave_from_bytes(raw: bytes, suffix: str):
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(raw); tmp.flush()
        try:
            y, _ = load_audio(tmp.name)
        except Exception:  # noqa: BLE001
            return None
    return y if len(y) >= 0.3 * TARGET_SR else None


def user_model_path() -> Path:
    """Own recordings go to the microphone-robust model when it has been built (src/app_assets.py)."""
    return ROBUST_PATH if ROBUST_PATH.exists() else BUNDLE_PATH


def keep_copy(wav: bytes, kind: str):
    """Save a recording to recordings/ when the user has switched that on."""
    if not st.session_state.get("keep_copies"):
        return
    import time
    folder = ROOT / "recordings"
    folder.mkdir(exist_ok=True)
    (folder / f"{time.strftime('%Y%m%d-%H%M%S')}_{kind}.wav").write_bytes(wav)


CALIB_SENTENCES = ["Kids are talking by the door.", "Dogs are sitting by the door.", "It's eleven o'clock."]
CLOSE_MARGIN = 0.15   # top two scores closer than this: show both ("close call")


def calibration_for_speaker(row):
    """Dataset clip: baseline from up to three OTHER neutral clips of the same held-out speaker, and
    (optionally) one OTHER clip of each emotion of that speaker to adapt both models (enrollment)."""
    st.session_state["speaker_enrol"] = []
    c1, c2 = st.columns(2)
    on = c1.toggle("Calibrate to this speaker's normal voice", value=True,
                   help="Uses up to three other neutral clips of the same held-out speaker as the reference point. "
                        "On the held-out speakers this raised UAR from 72.5% to 77.1% (RAVDESS) and from 72.3% to "
                        "75.8% (CREMA-D). Switch it off to see the plain model.")
    teach = c2.toggle("Teach it this speaker's voice", value=True, disabled=not on,
                      help="Also uses one OTHER clip of each emotion from the same speaker, as a user would in "
                           "'Teach it your voice'. Both models adapt to this speaker; the clip you are analysing is "
                           "never one of them (report/09_simulation.md, section 10).")
    if not on:
        return None, None
    ex = examples()
    spk = ex[(ex["speaker_id"] == row["speaker_id"]) & (ex["file"] != row["file"])]
    same = spk[spk["label"] == "neutral"]
    if same.empty:
        st.info("This speaker has no other neutral clips on this computer, so calibration is not available.")
        return None, None
    pick = same.sample(n=min(3, len(same)), random_state=0)
    feats = [features_for(load_audio(ROOT / p)[0]) for p in pick["path"]]
    desc = f"{len(feats)} other neutral clips of speaker {row['speaker_id']}"
    if teach:
        chosen = []
        for emo in CLASSES:
            cand = spk[spk["label"] == emo]
            if emo == "neutral" and len(cand) > len(pick):          # prefer a neutral clip not used as baseline
                cand = cand[~cand["file"].isin(pick["file"])]
            if cand.empty:
                chosen = []
                break
            chosen.append(cand.sample(n=1, random_state=0).iloc[0])
        if chosen:
            items = []
            for r in chosen:
                f = features_for(load_audio(ROOT / r["path"])[0])
                items.append((r["file"], {"egemaps": f["egemaps"], "wavlm": f["wavlm"], "label": r["label"]}))
            st.session_state["speaker_enrol"] = items
            desc += f", adapted with one other clip of each emotion"
        else:
            st.info("This speaker has no other clip of every emotion on this computer, so only calibration is used.")
    return K.baseline_from(feats), desc


def enrollment_for_user(calibrated: bool):
    """Own voice: one recording of each emotion. With all six, both models adapt to this voice."""
    teach = st.session_state.setdefault("teach", {})
    gen = st.session_state.setdefault("enrol_gen", 0)
    have = [e for e in CLASSES if f"enrol:{e}" in teach]
    with st.expander(f"Step 2: teach it your voice ({len(have)} of 6 emotions)", expanded=calibrated and len(have) < 6):
        if not calibrated:
            st.info("Calibrate first (Step 1). Teaching works on top of your calibration.")
            return
        st.markdown("Say **one sentence in each emotion**, for example “Kids are talking by the door”. Act it a "
                    "little, as the actors do. When all six are in, the app adapts **its own two models** to your "
                    "voice (speaker adaptation). On speakers it had never heard, this raised accuracy in every test "
                    "we ran, for example from 78% to 85% UAR on new RAVDESS speakers (report, section 10).")
        for emo in CLASSES:
            rec = raw_recorder(f"{emo.capitalize()}", key=f"enrol_rec_{gen}_{emo}")
            if rec is not None and teach.get(f"enrol:{emo}", {}).get("nonce") != rec["nonce"]:
                y = wave_from_bytes(rec["wav"], ".wav")
                if y is None:
                    st.error(f"{emo}: too short or unreadable. Record the full sentence again.")
                else:
                    f = features_for(y)
                    teach[f"enrol:{emo}"] = {"egemaps": f["egemaps"], "wavlm": f["wavlm"], "label": emo,
                                             "nonce": rec["nonce"]}
                    keep_copy(rec["wav"], f"teach_{emo}")
        ups = st.file_uploader("Or upload one recording per emotion, each named after its emotion (for example "
                               "happy.wav)", type=["wav", "flac", "ogg", "mp3"], accept_multiple_files=True,
                               key=f"enrol_up_{gen}")
        for up in ups or []:
            emo = next((e for e in CLASSES if e in up.name.lower()), None)
            if emo is None:
                st.warning(f"{up.name}: name the file after one of the six emotions.")
                continue
            sig = f"up:{up.name}:{up.size}"
            if teach.get(f"enrol:{emo}", {}).get("nonce") != sig:
                y = wave_from_bytes(up.getvalue(), Path(up.name).suffix.lower())
                if y is not None:
                    f = features_for(y)
                    teach[f"enrol:{emo}"] = {"egemaps": f["egemaps"], "wavlm": f["wavlm"], "label": emo, "nonce": sig}
        have = [e for e in CLASSES if f"enrol:{e}" in teach]
        missing = [e for e in CLASSES if e not in have]
        st.caption(("Done: " + ", ".join(have) + ". " if have else "")
                   + ("Still to record: " + ", ".join(missing) + "." if missing else
                      "All six are in: the app is now adapted to your voice. Recordings stay in this browser session."))
        if have and st.button("Clear my voice examples"):
            for e in CLASSES:
                teach.pop(f"enrol:{e}", None)
            st.session_state.pop("teach_fits", None)
            st.session_state["enrol_gen"] = gen + 1
            st.rerun()


def calibration_for_user():
    """Own recordings: up to three recordings of the user's normal voice, kept only in this browser session."""
    store = st.session_state.setdefault("calib_store", {})
    gen = st.session_state.setdefault("calib_gen", 0)
    ready = len(store)
    with st.expander(f"Step 1: calibrate to my voice ({ready} of 3 recordings ready)", expanded=ready < 3):
        st.markdown("Say each sentence in your **normal, calm voice**, the way you would talk to a friend. The app "
                    "then judges every recording by how it differs from *your* normal voice and *your* microphone, "
                    "instead of the recording studios it was trained in.")
        for i in range(3):
            rec = raw_recorder(f"Normal voice {i + 1}: “{CALIB_SENTENCES[i]}”", key=f"calib_rec_{gen}_{i}")
            if rec is not None:
                sig = rec["nonce"]
                if store.get(f"slot{i}", (None,))[0] != sig:
                    y = wave_from_bytes(rec["wav"], ".wav")
                    if y is None:
                        st.error("Too short or unreadable. Record the full sentence again.")
                    else:
                        store[f"slot{i}"] = (sig, features_for(y))
                        keep_copy(rec["wav"], f"calibration{i + 1}")
        ups = st.file_uploader("Or upload recordings of your normal voice (WAV, FLAC, OGG or MP3)",
                               type=["wav", "flac", "ogg", "mp3"], accept_multiple_files=True, key=f"calib_up_{gen}")
        for up in ups or []:
            key = f"up:{up.name}:{up.size}"
            if key not in store:
                y = wave_from_bytes(up.getvalue(), Path(up.name).suffix.lower())
                if y is not None:
                    store[key] = (key, features_for(y))
        if store and st.button("Clear my calibration"):
            store.clear()
            st.session_state["calib_gen"] = gen + 1
            st.rerun()
        st.caption("Recordings stay on this computer and are forgotten when you close the page. Three recordings "
                   "work best; one or two also work, but less reliably. Record in the same room, with the same "
                   "microphone and at the same distance as your later recordings: in our tests a normal sentence "
                   "only 2 dB quieter than the calibration was still read as sad.")
    if not store:
        return None, None
    if not st.toggle("Use my calibration", value=True):
        return None, None
    return K.baseline_from([v[1] for v in store.values()]), f"{len(store)} recording{'s' if len(store) != 1 else ''} of your normal voice"


# ===========================================================================
# PAGE 1 — Analyse a voice
# ===========================================================================
def get_input():
    """Returns (raw_bytes, waveform, true label, metadata row, model path, out-of-domain flag, suffix) or None."""
    src_choice = st.segmented_control("Source", ["Dataset clip", "Upload a file", "Record"],
                                      default="Dataset clip", label_visibility="collapsed")
    if src_choice is None:
        src_choice = "Dataset clip"
    if src_choice == "Dataset clip":
        if not U.HELDOUT_PATH.exists():
            st.warning("The held-out demo model is missing. Run `python -m src.app_assets` once (about 3 "
                       "minutes) so dataset clips come from speakers the model has never heard.")
            return None
        ex = examples()
        if ex.empty:
            st.error("No fold-0 audio files found under data/raw/. Copy the datasets into data/raw first.")
            return None
        c1, c2, c3 = st.columns([1, 1, 1.2])
        corpus = c1.selectbox("Dataset", ["RAVDESS", "CREMA-D"])
        emo = c2.selectbox("Emotion the actor performed", CLASSES)
        sub = ex[(ex["corpus_name"] == corpus) & (ex["label"] == emo)].reset_index(drop=True)
        if sub.empty:
            st.info("No clips of this kind are available on this computer.")
            return None
        # A random clip for every dataset and emotion (the first clip is always the same speaker, actor 9,
        # whom the held-out model hears as sad for most emotions).
        pick_key = f"pick_{corpus}_{emo}"
        if pick_key not in st.session_state:
            st.session_state[pick_key] = int(np.random.randint(len(sub)))
        c3.markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
        if c3.button("Pick another clip", width="stretch"):
            step = int(np.random.randint(1, len(sub))) if len(sub) > 1 else 0
            st.session_state[pick_key] = (st.session_state[pick_key] + step) % len(sub)
        row = sub.iloc[st.session_state[pick_key] % len(sub)]
        path = ROOT / row["path"]
        y, _ = load_audio(path)
        base, base_desc = calibration_for_speaker(row)
        return path.read_bytes(), y, row["label"], row, U.HELDOUT_PATH, False, ".wav", base, base_desc
    base, base_desc = calibration_for_user()
    enrollment_for_user(base is not None)
    st.caption("The model knows six emotions: neutral, happy, sad, angry, fearful and disgust. Excitement is not one "
               "of them, so excited shouting usually lands on angry or happy.")
    if src_choice == "Upload a file":
        up = st.file_uploader("A WAV, FLAC, OGG or MP3 file with 2 to 5 seconds of one person speaking",
                              type=["wav", "flac", "ogg", "mp3"])
        if up is None:
            st.info("Upload a short recording to analyse it.")
            return None
        return up.getvalue(), None, None, None, user_model_path(), True, Path(up.name).suffix.lower(), base, base_desc
    rec = raw_recorder("Step 3: record one sentence with an emotion, for example “Kids are talking by the door”",
                       key="main_rec")
    st.toggle("Keep a copy of my recordings in the project's recordings/ folder", key="keep_copies",
              help="Off by default. Copies stay on this computer (the folder is never uploaded to GitHub) and let "
                   "you check later what the microphone really captured.")
    if rec is None:
        st.info("Press Record, say one sentence with an emotion, then press Stop.")
        return None
    note = processing_note(rec["processing"])
    if note:
        st.caption(note)
    if st.session_state.get("kept_nonce") != rec["nonce"]:
        keep_copy(rec["wav"], "recording")
        st.session_state["kept_nonce"] = rec["nonce"]
    return rec["wav"], None, None, None, user_model_path(), True, ".wav", base, base_desc


def page_analyse():
    masthead("Analyse a voice",
             "Play a clip and the model predicts one of six emotions, then shows which parts of the voice "
             "drove that answer.")
    if not BUNDLE_PATH.exists():
        st.error("No trained model found. Run `notebooks/07_final_model.ipynb` first.")
        return
    got = get_input()
    if got is None:
        st.markdown(f'<div class="console"><div><div class="heard">Nothing playing yet</div>'
                    f'<div class="word" style="color:#5E6496">silence</div></div>'
                    f'<div class="trace">{S.waveform_svg(None, SIGNAL)}</div></div>', unsafe_allow_html=True)
        footer()
        return
    raw_bytes, y, truth, row, bundle_path, out_of_domain, suffix, base, base_desc = got
    if y is None:
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(raw_bytes); tmp.flush()
            try:
                y, _ = load_audio(tmp.name)
            except Exception as e:  # noqa: BLE001
                st.error(f"This file could not be read ({e}). Convert it to WAV and try again.")
                return
    if len(y) < 0.3 * TARGET_SR:
        st.error("The recording is shorter than 0.3 seconds after removing silence. Record a full sentence.")
        return

    st.audio(raw_bytes)
    bundle = get_bundle(str(bundle_path))
    names = bundle["egemaps_names"]
    warm_extractors()
    with st.spinner("Measuring the voice and predicting…"):
        feats = features_for(y)
        res = predict_features(feats["egemaps"], feats["wavlm"], bundle)
    duration = len(y) / TARGET_SR
    proba = pd.Series(res["proba"]).reindex(CLASSES).astype(float)
    orig_label, orig_top = res["label"], float(proba.max())
    clip_id = f"{len(y)}:{float(np.abs(y).sum()):.4f}"
    L = int(bundle["layer"])
    items = (list(st.session_state.get("teach", {}).items()) if row is None
             else list(st.session_state.get("speaker_enrol") or []))
    pA, pR = personal_model(L, base, items, str(bundle_path), None if row is None else 0)
    n_teach = len(items) if pR is not None else 0
    adapted = pA is not None
    if adapted:                                   # enrollment: the original model has learned this voice too
        proba = pd.Series(pA.predict_proba(np.hstack([feats["egemaps"], feats["wavlm"][L]])[None, :])[0],
                          index=CLASSES).astype(float)
    cal, relm = None, None
    if base is not None:
        relm = calib_model(L, 0 if row is not None else None)
        rel_use = relm if pR is None else {**relm, "predictor": pR}
        cal = K.calibrated_predict(feats["egemaps"], feats["wavlm"], base, rel_use, proba.values)
        proba = pd.Series(cal["proba"]).reindex(CLASSES).astype(float)
        rel_z = relm["explainer"].predict_proba(cal["relative_egemaps"][None, :])[0]
        second_label = CLASSES[int(np.argmax(rel_z))]
    else:
        second_label = res["explainer_label"]
    label, top = CLASSES[int(np.argmax(proba.values))], float(proba.max())
    ranked = proba.sort_values(ascending=False)
    runner, runner_p = str(ranked.index[1]), float(ranked.iloc[1])
    # Close call: on held-out speakers, when the top two scores were within 15 points (about 1 clip in 8) the
    # top answer was right only 33-40% of the time, but the true emotion was one of the two 73-88% of the time
    # (report/09_simulation.md, section 3). So the app names both instead of pretending to be sure.
    close = (top - runner_p) < CLOSE_MARGIN

    # ---------------- console ----------------
    tags = []
    if close:
        tags.append(f'<span class="tag" style="border-color:#E0A526;color:#F2C14E">close call: {label} or {runner}</span>')
    if truth is not None:
        if truth == label:
            tags.append('<span class="tag good">matches the performed emotion</span>')
        elif close and truth == runner:
            tags.append(f'<span class="tag good">actor performed {truth}, the close second</span>')
        else:
            tags.append(f'<span class="tag bad">actor performed {truth}</span>')
    if cal is not None:
        tags.append('<span class="tag good">calibrated to ' + ("your voice" if row is None else "this speaker") + '</span>')
    if adapted:
        tags.append('<span class="tag good">adapted to ' + ("your voice" if row is None else "this speaker") + '</span>')
    elif n_teach:
        tags.append(f'<span class="tag good">learned from {n_teach} correction{"s" if n_teach != 1 else ""}</span>')
    agree = second_label == label
    who = (f"held-out speaker {row['speaker_id']} ({row['gender']}) from {row['corpus_name']}, never heard by this model"
           if row is not None else "your recording, analysed on this computer and not stored")
    st.markdown(
        f'<div class="console" style="--emo:{EMO[label]}"><div><div class="heard">The model hears</div>'
        f'<div class="verdict"><div class="word" style="color:{EMO[label]}">{label}</div>'
        + (f'<div style="align-self:flex-end;color:#C9CCE0;font-size:1.15rem;margin:0 .6rem .9rem 0">'
           f'or <b style="color:{EMO[runner]}">{runner}</b> {runner_p:.0%}</div>' if close else '') +
        f'<div class="ring">{S.ring_svg(top, EMO[label], "rgba(255,255,255,.12)")}</div></div>'
        f'<div class="tags">{"".join(tags)}</div>{bars_html(proba.to_dict(), label)}</div>'
        f'<div><div class="trace">{S.waveform_svg(y, EMO[label])}</div>'
        f'<div class="trace-cap">{duration:.2f} seconds of speech after trimming silence. Source: {who}.</div></div></div>',
        unsafe_allow_html=True)

    if row is None:
        teach_panel(clip_id, feats, label, base is not None)

    # ---------------- trust notes ----------------
    lo, hi, ref = ood_reference()
    n_out = int(((feats["egemaps"] < lo) | (feats["egemaps"] > hi)).sum())
    rank = float((ref < n_out).mean() * 100)
    whose = "your voice" if row is None else "this speaker"
    if row is None and adapted:
        taught = " The model has also adapted to your voice from your recordings of all six emotions."
    elif row is None and n_teach:
        taught = f" It has also learned from {n_teach} correction{'s' if n_teach != 1 else ''}."
    else:
        taught = ""
    if cal is not None:
        n1 = ('', f"Calibrated to {whose}", f"This clip is compared with {base_desc}." + taught
              + (" Still read the score as approximate." if out_of_domain else ""))
    elif out_of_domain:
        n1 = ('caution', "Different recording conditions",
              "Your microphone and room differ from the studios the model was trained in, so normal speech can "
              "read as sad. Calibrate to your voice (step 1) for a better answer.")
    else:
        n1 = ('', "New speaker, same recording style",
              "On clips like this the scores are reliable: a 70% score was right about 70% of the time.")
    n2 = ('caution' if rank >= 95 else '', "Unusual voice" if rank >= 95 else "Typical voice",
          f"{n_out} of 88 voice features are outside the usual training range."
          + (" Treat the result with caution." if rank >= 95 else ""))
    n3 = ('' if agree else 'caution', "Second opinion agrees" if agree else "Second opinion differs",
          f"A simpler model that only uses the 88 named features says {second_label}. "
          + ("Both models point the same way." if agree else "The evidence is mixed."))
    st.markdown('<div class="notes">' + "".join(
        f'<div class="note {c}"><h4>{h}</h4><p>{p}</p></div>' for c, h, p in (n1, n2, n3)) + '</div>',
        unsafe_allow_html=True)
    for w in res["warnings"]:
        if "outside the range" in w:
            continue
        if w.startswith("Low confidence") and top >= 0.4:      # the plain model was unsure, the final answer is not
            continue
        st.warning(w)
    if top < 0.4 and not any(w.startswith("Low confidence") for w in res["warnings"]):
        st.warning("Low confidence: no emotion clearly dominates.")

    tabs = st.tabs(["Why", "Signal"])

    # ---------------- why ----------------
    with tabs[0]:
        train = train_egemaps()
        cw1, cw2 = st.columns(2)
        k_name = cw1.selectbox("Explain the evidence for", CLASSES, index=CLASSES.index(label),
                               help="Why this emotion, or why not another one?")
        k = CLASSES.index(k_name)
        others = [c for c in CLASSES if c != k_name]
        vs_name = cw2.selectbox("Compared with", ["all other emotions"] + others, key=f"vs_{k_name}",
                                help="Optional: pick one emotion to ask 'why this emotion and not that one?'.")
        j = CLASSES.index(vs_name) if vs_name in CLASSES else None
        if cal is not None:
            phi = K.relative_shap(cal["relative_egemaps"], relm, k)
            if j is not None:
                phi = phi - K.relative_shap(cal["relative_egemaps"], relm, j)
            expl = pd.DataFrame({"feature": names, "readable": [U.readable(n) for n in names],
                                 "group": [U.feature_group(n) for n in names],
                                 "value": cal["relative_egemaps"], "contribution": phi})
            expl = expl.reindex(expl["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
            expl["percentile"] = np.nan
        else:
            expl = U.local_shap(feats["egemaps"], bundle, k)
            if j is not None:
                other = U.local_shap(feats["egemaps"], bundle, j).set_index("feature")["contribution"]
                expl["contribution"] = expl["contribution"] - expl["feature"].map(other).values
                expl = expl.reindex(expl["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
            expl["percentile"] = U.percentiles(expl["value"].values, train[list(expl["feature"])])
        st.markdown(U.explanation_sentence(expl, k_name, versus=None if j is None else vs_name,
                                           relative_to=None if cal is None else
                                           ("your" if row is None else "this speaker’s") + " normal voice"))
        top_n = expl.head(10).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=top_n["contribution"], y=top_n["readable"], orientation="h",
            marker_color=["#2A78D6" if v > 0 else "#EB6834" for v in top_n["contribution"]],
            customdata=np.c_[top_n["value"], top_n["percentile"], top_n["group"]],
            hovertemplate=("<b>%{y}</b><br>contribution %{x:+.2f} log-odds<br>difference from the normal voice "
                           "%{customdata[0]:+.3f}<br>group %{customdata[2]}<extra></extra>") if cal is not None else
                          ("<b>%{y}</b><br>contribution %{x:+.2f} log-odds<br>value %{customdata[0]:.3f}"
                           "<br>training percentile %{customdata[1]:.0f}<br>group %{customdata[2]}<extra></extra>")))
        fig.add_vline(x=0, line_color="gray")
        fig.update_xaxes(title=(f"blue pushes toward {k_name}, orange pushes away" if j is None else
                                f"blue favours {k_name}, orange favours {vs_name}"))
        show(style(fig, 430, legend=False, title=(f"The 10 voice features that mattered most for {k_name}" if j is None
                                                  else f"What tips it to {k_name} rather than {vs_name}")))
        if cal is not None:
            rel_p = pd.Series(cal["relative_proba"]).reindex(CLASSES)
            st.markdown(f"**How calibration changed the answer.** The original model said **{orig_label}** "
                        f"({orig_top:.0%}); the model that compares with the normal voice says "
                        f"**{rel_p.idxmax()}** ({rel_p.max():.0%}). The result above combines the two.")
        st.caption("Each bar is one measured property of the voice. Longer bars mattered more. This shows what the "
                   "model uses; it does not prove that a voice property causes an emotion.")

    # ---------------- signal ----------------
    with tabs[1]:
        v = views_for(y)
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            row_heights=[0.2, 0.3, 0.3, 0.2],
                            subplot_titles=("Waveform and energy", "Spectrogram (dB)",
                                            "Mel-spectrogram (64 bands)", "Pitch contour (pYIN)"))
        wav = v["wave"] / (np.abs(v["wave"]).max() + 1e-9)
        fig.add_trace(go.Scatter(x=v["t_wave"], y=wav, line=dict(color="#8A8F98", width=1), name="waveform"), 1, 1)
        rms = (v["rms_db"] - v["rms_db"].max()) / 80 + 1
        fig.add_trace(go.Scatter(x=v["t"], y=np.clip(rms, 0, 1), line=dict(color=SIGNAL_TEXT, width=2),
                                 name="energy (relative)"), 1, 1)
        fig.add_trace(go.Heatmap(x=v["t"], y=v["freqs"], z=v["spec_db"], colorscale="Magma", showscale=False,
                                 name="spectrogram"), 2, 1)
        fig.add_trace(go.Heatmap(x=v["t"], y=np.arange(64), z=v["mel_db"], colorscale="Viridis", showscale=False,
                                 name="mel"), 3, 1)
        fig.add_trace(go.Scatter(x=v["t"], y=v["f0"], mode="lines", line=dict(color="#2A78D6", width=3),
                                 name="pitch (F0)", hovertemplate="%{x:.2f} s · %{y:.0f} Hz<extra></extra>"), 4, 1)
        fig.update_yaxes(title_text="Hz", row=2, col=1, range=[0, 8000])
        fig.update_yaxes(title_text="mel band", row=3, col=1)
        fig.update_yaxes(title_text="Hz", row=4, col=1)
        fig.update_yaxes(range=[-1.05, 1.05], row=1, col=1)
        fig.update_xaxes(title_text="time (s)", row=4, col=1)
        show(style(fig, 820, bottom=True))
        f0 = v["f0"][~np.isnan(v["f0"])]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Speech length", f"{duration:.2f} s")
        m2.metric("Median pitch", f"{np.median(f0):.0f} Hz" if len(f0) else "none found")
        m3.metric("Pitch range (10th to 90th)", f"{np.percentile(f0, 90) - np.percentile(f0, 10):.0f} Hz" if len(f0) else "none found")
        m4.metric("Voiced frames", f"{len(f0) / max(1, len(v['f0'])):.0%}")
        st.caption("This is exactly the processed 16 kHz signal the models receive, after silence trimming. The "
                   "pitch contour is drawn with pYIN; the model's own pitch features come from openSMILE.")

    footer()


# ===========================================================================
# PAGE 2 — How the model decides
# ===========================================================================
def page_global():
    masthead("How the model decides",
             "Global explanations from Phase 7. Every clip was explained by a model that had never heard its speaker, "
             "then the explanations were averaged.")
    c1, c2, _ = st.columns([1, 1, 2])
    corpus = c1.selectbox("Dataset", ["ravdess", "cremad"], format_func=U.NAME.get)
    model = c2.selectbox("Model", ["logreg", "lgbm"],
                         format_func={"logreg": "Logistic Regression", "lgbm": "LightGBM"}.get)
    gi = U.load_csv(f"phase7/global_importance_{corpus}_{model}.csv")
    if gi is None:
        st.error("Phase 7 results not found in results/phase7/."); return
    gi["readable"] = gi["feature"].map(U.readable)
    top = gi.head(15).iloc[::-1]
    ca, cb = st.columns([1.6, 1])
    with ca:
        fig = go.Figure(go.Bar(x=top["importance"], y=top["readable"], orientation="h", showlegend=False,
                               marker_color=[GRP[g] for g in top["group"]], customdata=top[CLASSES].values,
                               hovertemplate="<b>%{y}</b><br>mean |SHAP| %{x:.3f}<br>" +
                               "<br>".join(f"{c}: %{{customdata[{i}]:.2f}}" for i, c in enumerate(CLASSES)) +
                               "<extra></extra>"))
        for g, col in GRP.items():
            fig.add_trace(go.Bar(x=[None], y=[None], marker_color=col, name=g))
        fig.update_xaxes(title="mean |SHAP| (log-odds)")
        show(style(fig, 520, title=f"Top 15 features for {U.NAME[corpus]}"))
    with cb:
        p7 = U.load_json("phase7_summary.json")
        gs = p7.get("group_share", {}).get(f"{U.NAME[corpus]} · {model}", {})
        if gs:
            sizes = {"Energy": 15, "Frequency": 24, "Spectral": 43, "Temporal": 6}
            df = pd.DataFrame({"group": list(gs), "share": [gs[g] * 100 for g in gs],
                               "per feature": [gs[g] * 100 / sizes[g] for g in gs]})
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Share of total |SHAP| (%)", "Per feature (%)"))
            fig.add_trace(go.Bar(x=df["group"], y=df["share"], marker_color=[GRP[g] for g in df["group"]]), 1, 1)
            fig.add_trace(go.Bar(x=df["group"], y=df["per feature"], marker_color=[GRP[g] for g in df["group"]]), 1, 2)
            fig.update_annotations(font=dict(size=13, color=MUTED))
            show(style(fig, 320, legend=False).update_layout(margin=dict(t=46)))
            st.caption("Spectral leads in total only because it has 43 of the 88 features. Per feature, Energy is "
                       "the most informative group, in line with Paper 5.")
        sp = p7.get("shap_vs_permutation", {}).get(corpus)
        if sp:
            st.metric("SHAP versus permutation importance (Spearman)", f"{sp['spearman']:.2f}",
                      help="A second, independent explanation method agrees with SHAP.")
            st.caption(f"Top-10 overlap between the two methods: {sp['top10_overlap']} of 10")

    st.subheader("Do the two datasets rely on the same features?")
    a = U.load_csv("phase7/global_importance_ravdess_logreg.csv")
    b = U.load_csv("phase7/global_importance_cremad_logreg.csv")
    if a is not None and b is not None:
        m = a[["feature", "group", "importance"]].merge(b[["feature", "importance"]], on="feature", suffixes=("_r", "_c"))
        m["readable"] = m["feature"].map(U.readable)
        fig = px.scatter(m, x="importance_r", y="importance_c", color="group", color_discrete_map=GRP,
                         hover_name="readable", labels={"importance_r": "importance on RAVDESS",
                                                        "importance_c": "importance on CREMA-D"})
        mx = max(m["importance_r"].max(), m["importance_c"].max())
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(dash="dot", color="gray"),
                                 name="equal importance"))
        fig.update_traces(marker=dict(size=9, opacity=.8), selector=dict(mode="markers"))
        c1, c2 = st.columns([1.6, 1])
        with c1:
            show(style(fig, 430))
        with c2:
            cc = U.load_json("phase7_summary.json").get("cross_corpus_preview", {}).get("logreg", {})
            st.metric("Rank correlation (Spearman)", f"{cc.get('spearman', float('nan')):.2f}")
            st.metric("Top-10 overlap (Jaccard)", f"{cc.get('top10_jaccard', float('nan')):.2f}")
            st.caption("Each dot is one eGeMAPS feature. The CREMA-D model leans on loudness, the RAVDESS model on "
                       "MFCCs and formants. The overall profiles correlate moderately, but the top features differ.")

    st.subheader("What drives each emotion")
    d = U.load_csv("phase7/per_emotion_drivers.csv")
    if d is not None:
        c1, c2 = st.columns([1, 3])
        cn = c1.selectbox("Dataset ", ["RAVDESS", "CREMA-D"])
        en = c1.selectbox("Emotion", CLASSES)
        sub = d[(d["corpus"] == cn) & (d["emotion"] == en)].copy()
        sub.insert(2, "Feature (readable)", sub["feature"].map(U.readable))
        c2.dataframe(sub.drop(columns=["corpus", "emotion"]), hide_index=True, width="stretch")

    np_ = U.load_json("phase7_summary.json").get("no_pitch_shortcut_check")
    if np_:
        with st.expander("Shortcut check: does the model treat 'no pitch found' as sad?"):
            c1, c2 = st.columns(2)
            c1.metric("UAR with the 39 no-pitch clips", f"{np_['uar_with']:.1%}")
            c2.metric("UAR after removing them", f"{np_['uar_without']:.1%}")
            st.caption("No measurable difference, so no harmful shortcut (Phase 7).")
    footer()


# ===========================================================================
# PAGE 3 — Model performance
# ===========================================================================
def page_performance():
    masthead("Model performance",
             "Speaker-independent 5-fold cross-validation: no speaker is ever in both training and test. "
             "UAR is the average recall over the six emotions; guessing scores 16.7%.")
    e1 = U.load_csv("phase6/e1_summary.csv")
    if e1 is not None:
        e1 = e1.copy()
        e1["Model"] = e1["model"].map({"logreg": "LogReg", "svm": "SVM (RBF)", "lgbm": "LightGBM", "mlp": "MLP"})
        e1["Features"] = e1["features"].map({"egemaps": "eGeMAPS", "wavlm": "WavLM", "fusion": "Fusion"})
        e1["Corpus"] = e1["corpus"].map(U.NAME)
        fig = px.bar(e1, x="Model", y=e1["uar_mean"] * 100, color="Features", barmode="group",
                     facet_col="Corpus", error_y=e1["uar_std"] * 100,
                     color_discrete_map={"eGeMAPS": "#EB6834", "WavLM": "#7B5BD6", "Fusion": "#2A78D6"},
                     category_orders={"Model": ["LogReg", "SVM (RBF)", "LightGBM", "MLP"], "Corpus": ["RAVDESS", "CREMA-D"],
                                      "Features": ["eGeMAPS", "WavLM", "Fusion"]})
        fig.add_hline(y=CHANCE, line_dash="dot", line_color="gray")
        fig.update_yaxes(title="UAR (%), mean and SD over folds", range=[0, 100])
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_yaxes(title_text="", col=2)
        show(style(fig, 450, bottom=True, title="Within-dataset results (Phase 6). Dotted line = chance. KNN excluded by design"))
        best = e1.sort_values("uar_mean", ascending=False).groupby("Corpus").head(1)
        cols = st.columns(len(best))
        for col, r in zip(cols, best.itertuples()):
            col.metric(f"Best on {r.Corpus}", f"{r.uar_mean:.1%} UAR", f"{r.Model} with {r.Features}", delta_color="off")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Why speaker-independent splits matter")
        e0 = U.load_csv("phase6/e0_leakage.csv")
        if e0 is not None:
            e0 = e0.copy(); e0["label"] = e0["corpus"].map(U.NAME) + ", " + e0["features"]
            fig = go.Figure([go.Bar(name="random split (speaker leakage)", x=e0["label"],
                                    y=e0["random_split_uar"] * 100, marker_color="#D64545"),
                             go.Bar(name="speaker-independent split", x=e0["label"],
                                    y=e0["speaker_split_uar"] * 100, marker_color="#2A78D6")])
            fig.update_yaxes(title="UAR (%)")
            show(style(fig, 340).update_layout(barmode="group"))
            st.caption("Random splits put the same actor in training and test, inflating UAR by up to "
                       f"{e0['inflation_pts'].max():.1f} points on RAVDESS (24 actors).")
    with c2:
        st.subheader("Confusion matrix")
        a, b, c = st.columns(3)
        corpus = a.selectbox("Dataset", ["ravdess", "cremad"], format_func=U.NAME.get, key="cm_c")
        fs = b.selectbox("Features", ["fusion", "wavlm", "egemaps"], key="cm_f")
        mdl = c.selectbox("Model", ["logreg", "svm", "lgbm", "mlp"], key="cm_m")
        p = U.RESULTS / "phase6" / f"e1_{corpus}_{fs}_{mdl}_oof.npy"
        if p.exists():
            from sklearn.metrics import confusion_matrix
            yt, yp = U.labels_for(corpus), np.load(p)
            cm = confusion_matrix(yt, yp, labels=range(6), normalize="true") * 100
            fig = go.Figure(go.Heatmap(z=cm, x=CLASSES, y=CLASSES, colorscale=[[0, "#FFFFFF"], [1, INK]], zmin=0, zmax=100,
                                       text=cm.round(0), texttemplate="%{text:.0f}", showscale=False))
            fig.update_yaxes(autorange="reversed", title="performed")
            fig.update_xaxes(title="predicted")
            show(style(fig, 360, legend=False))
            st.caption("Row percentages from out-of-fold predictions: every clip predicted by a model that never "
                       "heard its speaker.")
        else:
            st.info("Predictions for this combination were not saved.")

    e6 = U.load_json("phase8_e6_summary.json")
    if e6:
        st.subheader("Standard evaluation (Phase 8, E6)")
        t1, t2, t3, t4 = st.tabs(["ROC-AUC", "Calibration", "Overfitting check", "Learning curve"])
        with t1:
            r = pd.DataFrame(e6["roc_auc"])
            r["config"] = r["model"] + ", " + r["features"]
            fig = px.bar(r, x="config", y="macro_auc", color="corpus", barmode="group",
                         color_discrete_sequence=["#2A78D6", "#EB6834"])
            fig.update_yaxes(range=[0.5, 1], title="macro ROC-AUC (one-vs-rest)")
            show(style(fig, 360))
        with t2:
            cols = st.columns(4)
            for col, (k, v) in zip(cols, e6["calibration_ece"].items()):
                name, fs_ = k.split("_")
                col.metric(f"ECE, {U.NAME[name]}, {fs_}", f"{v:.3f}")
            img = U.RESULTS / "figures" / "p8_e6_calibration.png"
            if img.exists():
                st.image(str(img))
            st.caption("ECE is the average gap between stated confidence and actual accuracy, measured on new "
                       "speakers from the same datasets. On a new microphone the gap grows.")
        with t3:
            o = pd.DataFrame(e6["overfitting"])
            o["config"] = o["corpus"] + ", " + o["model"] + ", " + o["features"]
            o = o.sort_values("gap_pts")
            fig = go.Figure()
            for r_ in o.itertuples():
                fig.add_trace(go.Scatter(x=[r_.val_uar * 100, r_.train_uar * 100], y=[r_.config] * 2, mode="lines",
                                         line=dict(color="rgba(140,140,140,.6)", width=3), showlegend=False))
            fig.add_trace(go.Scatter(x=o["val_uar"] * 100, y=o["config"], mode="markers", name="new speakers",
                                     marker=dict(size=11, color="#2A78D6")))
            fig.add_trace(go.Scatter(x=o["train_uar"] * 100, y=o["config"], mode="markers", name="training speakers",
                                     marker=dict(size=11, color="#EB6834")))
            fig.update_xaxes(title="UAR (%)")
            show(style(fig, 470))
            st.caption("LightGBM memorises its training speakers (about 100%) without beating LogReg on new speakers.")
        with t4:
            fig = go.Figure()
            for fs_, col in [("egemaps", "#EB6834"), ("wavlm", "#7B5BD6")]:
                d = e6["learning_curve"][fs_]
                xs = [int(k) for k in d["val_uar"]]
                fig.add_trace(go.Scatter(x=xs, y=[v * 100 for v in d["val_uar"].values()], name=f"{fs_}, new speakers",
                                         line=dict(color=col, width=3)))
                fig.add_trace(go.Scatter(x=xs, y=[v * 100 for v in d["train_uar"].values()], name=f"{fs_}, training",
                                         line=dict(color=col, dash="dash")))
            fig.update_xaxes(title="number of training speakers (CREMA-D)")
            fig.update_yaxes(title="UAR (%)")
            show(style(fig, 380))
            st.caption("WavLM keeps improving with more speakers; eGeMAPS levels off near 51%.")

    p9 = U.load_json("phase9_final_model_summary.json")
    if p9.get("heldout_check"):
        st.subheader("Final model recipe on held-out speakers")
        st.dataframe(pd.DataFrame(p9["heldout_check"]), hide_index=True, width="stretch")
        st.caption("Pooled training (both datasets) versus the single-dataset Phase 6 model, on the same fold-0 speakers.")
    footer()


# ===========================================================================
# PAGE 4 — New-dataset test
# ===========================================================================
def page_cross():
    masthead("The new-dataset test",
             "Train on one dataset, test on the other (Phase 8). The 95% intervals come from resampling speakers.")
    h1 = U.load_csv("phase8/e2_h1_test.csv")
    e2 = U.load_csv("phase8/e2_all.csv")
    if h1 is not None:
        rows = []
        for r in h1.itertuples():
            for fs in ["egemaps", "wavlm", "fusion"]:
                m, lo, hi = U.parse_ci(getattr(r, f"{fs}_cross_uar"))
                rows.append({"direction": r.direction, "norm": "source scaler" if r.norm == "source" else
                             "per-corpus normalisation", "features": fs, "uar": m, "lo": lo, "hi": hi})
        df = pd.DataFrame(rows)
        df["err_hi"], df["err_lo"] = df["hi"] - df["uar"], df["uar"] - df["lo"]
        fig = px.bar(df, x="features", y="uar", color="norm", barmode="group", facet_col="direction",
                     error_y="err_hi", error_y_minus="err_lo", color_discrete_sequence=["#8A8F98", "#2A78D6"],
                     category_orders={"direction": ["RAVDESS→CREMA-D", "CREMA-D→RAVDESS"],
                                      "features": ["egemaps", "wavlm", "fusion"]})
        if e2 is not None:
            ind = e2[e2.model == "logreg"].drop_duplicates(["source", "features"])
            for i, d in enumerate(["RAVDESS→CREMA-D", "CREMA-D→RAVDESS"]):
                src = "ravdess" if d.startswith("RAVDESS") else "cremad"
                s = ind[ind.source == src].set_index("features")["indomain_uar"] * 100
                fig.add_trace(go.Scatter(x=list(s.index), y=list(s.values), mode="markers",
                                         name="in-domain (target)" if i == 0 else None, showlegend=i == 0,
                                         marker=dict(symbol="line-ew-open", size=28, color="#D64545", line=dict(width=3))),
                              row=1, col=i + 1)
        fig.add_hline(y=CHANCE, line_dash="dot", line_color="gray")
        fig.update_yaxes(title="UAR on the other dataset (%)", range=[0, 90])
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_yaxes(title_text="", col=2)
        show(style(fig, 450, bottom=True, title="Cross-dataset UAR, Logistic Regression. Red line = score at home"))
        st.markdown("**H1: does WavLM lose relatively more than eGeMAPS?**")
        t = h1[["direction", "norm", "wavlm_minus_egemaps_rel_drop", "H1_supported"]].rename(columns={
            "wavlm_minus_egemaps_rel_drop": "Difference in relative drop (WavLM − eGeMAPS)", "H1_supported": "H1 supported"})
        st.dataframe(t, hide_index=True, width="stretch")
        st.info("H1 depends on direction: trained on the small RAVDESS, WavLM loses relatively more; trained on the "
                "varied CREMA-D, it transfers much better. Per-corpus normalisation helps every configuration.")
    if e2 is not None:
        with st.expander("All models (SVM, LightGBM, MLP)"):
            m = st.selectbox("Model", ["logreg", "svm", "lgbm", "mlp"])
            sub = e2[e2.model == m].copy()
            sub["cross UAR %"] = (sub["cross_uar"] * 100).round(1)
            sub["in-domain UAR %"] = (sub["indomain_uar"] * 100).round(1)
            st.dataframe(sub[["source", "target", "features", "norm", "cross UAR %", "in-domain UAR %"]],
                         hide_index=True, width="stretch")

    st.subheader("Are explanations stable? (H2)")
    pairs = U.load_csv("phase8/e3_pairs.csv")
    if pairs is not None:
        fig = px.box(pairs, x="level", y="spearman", color="rep", category_orders={"level": ["resample", "speaker", "corpus", "random"]},
                     color_discrete_sequence=["#2A78D6", "#1BAF7A", "#7B5BD6"], points=False)
        fig.update_yaxes(title="Spearman between two importance rankings")
        fig.update_xaxes(title="the two models were trained on")
        c1, c2 = st.columns([1.6, 1])
        with c1:
            show(style(fig, 400))
        with c2:
            st.markdown("- **resample**: two bootstrap samples of the same data\n"
                        "- **speaker**: two separate groups of speakers, same dataset\n"
                        "- **corpus**: one dataset each\n"
                        "- baseline, two random rankings: median about 0.00 (not drawn)")
            st.success("H2 supported for all three models (Mann–Whitney p < 10⁻¹⁰). The interpretable LogReg "
                       "eGeMAPS explanations are the most stable; WavLM dimensions the least.")

    st.subheader("Stable-core feature selection (H3)")
    h3 = U.load_csv("phase8/e4_h3_test.csv")
    if h3 is not None:
        h3 = h3.copy()
        h3["setting"] = h3["direction"] + ", " + h3["norm"] + ", k=" + h3["k"].astype(str)
        lo = [U.parse_ci(s) for s in h3["ci95"]]
        h3["lo"] = [a[0] for a in lo]; h3["hi"] = [a[1] for a in lo]
        fig = go.Figure(go.Scatter(x=h3["stable_minus_top_pts"], y=h3["setting"], mode="markers",
                                   marker=dict(size=11, color=np.where(h3["significant"],
                                                                       np.where(h3["stable_minus_top_pts"] > 0, "#1BAF7A", "#D64545"),
                                                                       "#8A8F98")),
                                   error_x=dict(type="data", symmetric=False, array=h3["hi"] - h3["stable_minus_top_pts"],
                                                arrayminus=h3["stable_minus_top_pts"] - h3["lo"])))
        fig.add_vline(x=0, line_color="gray")
        fig.update_xaxes(title="UAR difference, stable core minus usual top-k (points, 95% CI)")
        show(style(fig, 440, legend=False))
        st.caption("Green = significant gain, red = significant loss, grey = not distinguishable from 0. H3 is "
                   "inconclusive: stability helps diagnose models but is not a reliable selection recipe.")

    p8 = U.load_json("phase8_summary.json")
    st.subheader("Where it fails (E5)")
    if p8.get("E5_per_class"):
        pc = pd.DataFrame(p8["E5_per_class"])
        long = pc.melt(id_vars=["direction", "emotion"], value_vars=["in_domain_recall", "cross_recall"],
                       var_name="setting", value_name="recall")
        long["setting"] = long["setting"].map({"in_domain_recall": "at home", "cross_recall": "on the other dataset"})
        fig = px.bar(long, x="emotion", y="recall", color="setting", barmode="group", facet_col="direction",
                     color_discrete_sequence=["#8A8F98", "#D64545"], category_orders={"emotion": CLASSES})
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1] + " (tested on target)"))
        fig.update_yaxes(title_text="", col=2)
        show(style(fig, 400, bottom=True, title="Which emotions break across datasets (fusion LogReg, per-corpus norm)"))
        st.caption("RAVDESS-trained models almost never recognise CREMA-D neutral (recall 0.09): the acted neutral "
                   "style differs between datasets. Sad and happy break in the other direction.")
    c1, c2 = st.columns(2)
    if p8.get("E5_groups"):
        with c1:
            st.markdown("**Speaker gender and age (UAR)**")
            st.dataframe(pd.DataFrame(p8["E5_groups"]), hide_index=True, width="stretch", height=320)
    if p8.get("E5_label_quality"):
        with c2:
            lq = p8["E5_label_quality"]
            st.markdown("**Label quality: did CREMA-D listeners agree with the actor?**")
            st.dataframe(pd.DataFrame(lq["table"]), hide_index=True, width="stretch")
            st.metric("Human recognition versus model recall per emotion (Spearman)", f"{lq['human_vs_model_spearman']:.2f}")
            st.caption("What humans find hard, the model finds hard: CREMA-D scores are partly limited by label noise.")
    footer()


# ===========================================================================
# PAGE 5 — About
# ===========================================================================
def page_about():
    masthead("About this system", "Emotion Recognition Using Explainable Machine Learning, by Team Lucid.")
    layer = get_bundle(str(BUNDLE_PATH))["layer"] if BUNDLE_PATH.exists() else "?"
    st.graphviz_chart(f"""
digraph {{
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#FFFFFF", color="{INK}", fontname="Helvetica", fontcolor="{TEXT}"];
  edge [color="{MUTED}"];
  audio [label="Audio\\n(any rate, mono or stereo)"]; pre [label="Clean\\nmono, 16 kHz, trim silence"];
  eg [label="eGeMAPS\\n88 named voice features"]; wl [label="WavLM base-plus\\nlayer {layer}, averaged (768)"];
  pred [label="Fusion LogReg\\n(predicts)", fillcolor="{INK}", fontcolor="white"]; expm [label="eGeMAPS LogReg\\n(explains)"];
  out [label="Emotion and scores", color="{SIGNAL_TEXT}"];
  shap [label="Exact SHAP\\nexplanation", color="#EB6834"];
  audio -> pre -> eg; pre -> wl; eg -> pred; wl -> pred; eg -> expm; pred -> out; expm -> shap;
}}""")
    info = U.load_json("phase9_final_model_summary.json").get("final_info", {})
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
**Research question.** How well do emotion models, and their explanations, transfer from one dataset to another?

**Data.** RAVDESS (24 actors, 1,056 clips) and CREMA-D (91 actors, 7,442 clips), six shared emotions.

**Final model.** Logistic Regression on 88 eGeMAPS features plus WavLM layer {layer}, trained on
{info.get('n_clips', 8498):,} clips from {info.get('n_speakers', 115)} speakers. Dataset clips in the demo use the
same recipe trained without the fold-0 speakers, so you always hear a voice the model has never heard.

**Design choices**
- Speaker-independent evaluation everywhere (leakage inflated UAR by up to 9 points).
- Two representations: interpretable eGeMAPS and self-supervised WavLM.
- KNN excluded by project rule; LogReg, SVM, LightGBM and MLP compared.
- SHAP is exact for the models explained, cross-checked with permutation importance.
""")
    with c2:
        st.markdown("""
**What the Analyse page shows**
- **The answer**: the emotion the model hears, with all six scores.
- **Why**: the voice features that pushed the answer, computed exactly with SHAP.
- **Signal**: the waveform, spectrogram and pitch of the clip.

**Limitations and ethics**
- Acted, exaggerated English speech; real emotions are subtler.
- Two datasets only; scores are over-confident on new microphones and rooms.
- Explanations show what the model uses, not what causes emotions.
- Not for decisions about people (hiring, policing, health). Recordings stay on this computer.
""")
    st.caption("Datasets: RAVDESS (CC BY-NC-SA 4.0), CREMA-D (ODbL). Models: microsoft/wavlm-base-plus, openSMILE eGeMAPSv02.")
    footer()


nav = st.navigation([
    st.Page(page_analyse, title="Analyse a voice", url_path="analyse", default=True),
    st.Page(page_global, title="How it decides", url_path="explanations"),
    st.Page(page_performance, title="Performance", url_path="performance"),
    st.Page(page_cross, title="New-dataset test", url_path="new-dataset"),
    st.Page(page_about, title="About", url_path="about"),
], position="top")
nav.run()
