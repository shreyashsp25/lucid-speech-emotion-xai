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
from plotly.subplots import make_subplots
import streamlit as st

from src.audio import load_audio, TARGET_SR
from src.models import CLASSES
from src.predict import BUNDLE_PATH, predict_features, extract_one
from src import app_utils as U

st.set_page_config(page_title="Lucid · Speech Emotion XAI", page_icon="🎙️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.hero {padding: 1.1rem 1.4rem; border-radius: 14px; margin-bottom: 1rem;
       background: linear-gradient(120deg, rgba(42,120,214,.12), rgba(123,91,214,.10));
       border: 1px solid rgba(128,128,128,.18);}
.hero h1 {font-size: 1.75rem; margin: 0 0 .25rem 0;}
.hero p {margin: 0; opacity: .8;}
.card {padding: 1rem 1.2rem; border-radius: 12px; border: 1px solid rgba(128,128,128,.22);
       background: rgba(128,128,128,.05); height: 100%;}
.big-label {font-size: 2.3rem; font-weight: 700; line-height: 1.1; margin: .2rem 0 .3rem 0;}
.muted {opacity: .72; font-size: .9rem;}
.pill {display: inline-block; padding: .12rem .6rem; border-radius: 999px; font-size: .8rem;
       font-weight: 600; margin-right: .3rem;}
</style>
""", unsafe_allow_html=True)

EMO = U.EMO_COLOUR
GRP = U.GROUP_COLOUR
CHANCE = 100 / 6


def style(fig, h=360, legend=True, title=None, bottom=False):
    """Shared chart look. Title at the very top, legend under it (or below the plot for facet charts)."""
    t = 12 + (34 if title else 0) + (30 if legend and not bottom else 0)
    fig.update_layout(
        height=h, showlegend=legend, font=dict(size=13),
        margin=dict(l=10, r=10, t=t, b=10 + (70 if legend and bottom else 0)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", x=0, xanchor="left", title_text="",
                    **({"y": -0.2, "yanchor": "top"} if bottom else {"y": 1.01, "yanchor": "bottom"})))
    if title:
        fig.update_layout(title=dict(text=title, x=0, xanchor="left", y=1, yanchor="top",
                                     yref="container", pad=dict(t=10, l=8), font=dict(size=15)))
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def pill(text, colour):
    return f'<span class="pill" style="background:{colour}22;color:{colour};border:1px solid {colour}55">{text}</span>'


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained model…")
def get_bundle(path: str):
    return joblib.load(path)


@st.cache_resource(show_spinner="Loading openSMILE + WavLM (first time only, ~20 s)…")
def warm_extractors():
    from src.predict import _extractors
    _extractors("cpu")
    return True


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


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("### 🎙️ Lucid · Speech Emotion XAI")
st.sidebar.caption("Emotion recognition from voice, with explanations and honest limits.")
page = st.sidebar.radio("Navigate", ["Analyse a recording", "How the model decides",
                                     "Model performance", "Cross-corpus & errors",
                                     "About & limitations"], label_visibility="collapsed")
st.sidebar.divider()
info = U.load_json("phase9_final_model_summary.json").get("final_info", {})
if info:
    st.sidebar.markdown(f"**Final model** · LogReg on eGeMAPS + WavLM layer "
                        f"{get_bundle(str(BUNDLE_PATH))['layer'] if BUNDLE_PATH.exists() else '?'}")
    st.sidebar.caption(f"Trained on {info['n_clips']:,} clips · {info['n_speakers']} speakers "
                       f"(RAVDESS + CREMA-D) · 6 emotions · chance = 16.7%")
st.sidebar.caption("Team Lucid: Pranav Sajith · Piriya Shreyash Shanmukh · Praveen Raj")


# ===========================================================================
# PAGE 1 — Analyse a recording
# ===========================================================================
def page_analyse():
    st.markdown('<div class="hero"><h1>Analyse a recording</h1>'
                '<p>Real model, real preprocessing: audio → mono 16 kHz → silence trimmed → '
                'eGeMAPS (88 named voice features) + WavLM (self-supervised speech model) → '
                'prediction → SHAP explanation.</p></div>', unsafe_allow_html=True)
    if not BUNDLE_PATH.exists():
        st.error("No trained model found. Run `notebooks/07_final_model.ipynb` first.")
        return

    mode = st.radio("Input", ["Dataset example (speaker the model never heard)",
                              "Upload a file", "Record with microphone"], horizontal=True)
    raw_bytes, y, truth, row, suffix = None, None, None, None, ".wav"
    bundle_path, out_of_domain = BUNDLE_PATH, True

    if mode.startswith("Dataset"):
        if not U.HELDOUT_PATH.exists():
            st.warning("The held-out demo model is missing. Run `python -m src.app_assets` once "
                       "(about 3 minutes) so dataset examples come from speakers the model has never "
                       "heard. Until then, dataset clips would be **training** clips, so they are disabled.")
            return
        ex = examples()
        if ex.empty:
            st.error("No fold-0 audio files found under data/raw/.")
            return
        c1, c2, c3 = st.columns([1, 1, 2])
        corpus = c1.selectbox("Corpus", ["RAVDESS", "CREMA-D"])
        emo = c2.selectbox("True emotion", CLASSES)
        sub = ex[(ex["corpus_name"] == corpus) & (ex["label"] == emo)].reset_index(drop=True)
        if "pick" not in st.session_state:
            st.session_state.pick = 0
        if c3.button("🎲 Random clip", width="stretch"):
            st.session_state.pick = int(np.random.randint(len(sub)))
        idx = st.session_state.pick % len(sub)
        row = sub.iloc[idx]
        st.caption(f"Clip {idx + 1} of {len(sub)}: `{row['file']}` · speaker {row['speaker_id']} "
                   f"({row['gender']}) · this speaker is in the held-out fold 0.")
        path = ROOT / row["path"]
        raw_bytes = path.read_bytes()
        y, _ = load_audio(path)
        truth, bundle_path, out_of_domain = row["label"], U.HELDOUT_PATH, False
    elif mode == "Upload a file":
        up = st.file_uploader("WAV, FLAC, OGG or MP3 · 2–5 s of one person speaking one sentence",
                              type=["wav", "flac", "ogg", "mp3"])
        if up is None:
            st.info("Upload a short recording to begin, or switch to a dataset example.")
            return
        raw_bytes, suffix = up.getvalue(), Path(up.name).suffix.lower()
    else:
        rec = st.audio_input("Record one sentence (2–5 s), e.g. *“Kids are talking by the door.”*")
        if rec is None:
            st.info("Press the microphone, speak one sentence with an emotion, then stop.")
            return
        raw_bytes = rec.getvalue()

    if y is None:
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(raw_bytes); tmp.flush()
            try:
                y, _ = load_audio(tmp.name)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read this audio file ({e}). Try WAV.")
                return
    if len(y) < 0.3 * TARGET_SR:
        st.error("The recording is too short after removing silence (< 0.3 s).")
        return

    st.audio(raw_bytes)
    clip_key = (len(y), float(np.abs(y).sum()))
    bundle = get_bundle(str(bundle_path))
    warm_extractors()
    with st.spinner("Extracting eGeMAPS + WavLM features and predicting…"):
        feats = features_for(y)
        res = predict_features(feats["egemaps"], feats["wavlm"], bundle)
    duration = len(y) / TARGET_SR
    if duration < 1.0:
        res["warnings"].append("Very short clip (< 1 s) after trimming silence.")
    proba = pd.Series(res["proba"]).reindex(CLASSES).astype(float)
    label, top = res["label"], float(proba.max())

    # ---------------- prediction row ----------------
    c1, c2, c3 = st.columns([1.05, 1.5, 1.35])
    with c1:
        strength = "low" if top < 0.4 else ("moderate" if top < 0.7 else "high")
        verdict = ""
        if truth is not None:
            ok = truth == label
            verdict = (pill("✓ correct", "#1BAF7A") if ok else pill(f"✗ true: {truth}", "#D64545"))
        agree = res["explainer_label"] == label
        st.markdown(
            f'<div class="card"><div class="muted">Predicted emotion</div>'
            f'<div class="big-label" style="color:{EMO[label]}">{label.capitalize()}</div>'
            f'{pill(f"model score {top:.0%} · {strength}", EMO[label])}{verdict}'
            f'<div class="muted" style="margin-top:.7rem">Interpretable eGeMAPS model says: '
            f'<b>{res["explainer_label"]}</b> {"(agrees)" if agree else "(disagrees)"}</div>'
            f'<div class="muted">Speech analysed: {duration:.2f} s after trimming silence</div>'
            + (f'<div class="muted">Listeners in CREMA-D {"agreed" if row.get("voice_agrees") else "did NOT agree"} '
               f'with the actor\'s intended emotion for this clip.</div>'
               if row is not None and "voice_agrees" in row and pd.notna(row.get("voice_agrees")) else "")
            + '</div>', unsafe_allow_html=True)
    with c2:
        fig = go.Figure(go.Bar(x=proba.values * 100, y=CLASSES, orientation="h",
                               marker_color=[EMO[c] if c == label else "rgba(140,140,140,.45)" for c in CLASSES],
                               text=[f"{p:.0%}" for p in proba.values], textposition="outside"))
        fig.update_xaxes(range=[0, 112], title="model score (%)")
        fig.update_traces(cliponaxis=False)
        fig.update_yaxes(autorange="reversed")
        fig.add_vline(x=CHANCE, line_dash="dot", line_color="gray",
                      annotation_text="chance", annotation_position="top")
        show(style(fig, 290, legend=False, title="Class scores (fusion model)"))
    with c3:
        e6 = U.load_json("phase8_e6_summary.json").get("calibration_ece", {})
        st.markdown("**How much to trust this**")
        if out_of_domain:
            st.warning("**Your recording is outside the training data** (different microphone, room "
                       "and speaker). In our cross-corpus tests, accuracy fell from 74–82% to 39–68% UAR "
                       "under such shifts, and scores become over-confident. Treat these as rough.")
        else:
            st.success(f"New speaker, same recording style as training. On such data the scores are well "
                       f"calibrated (ECE {e6.get('ravdess_fusion', 0.025):.3f} RAVDESS, "
                       f"{e6.get('cremad_fusion', 0.021):.3f} CREMA-D): a score of 70% was right about 70% "
                       f"of the time.")
        if not agree:
            st.info("The two models disagree, so the evidence is mixed.")
        lo, hi, ref = ood_reference()
        n_out = int(((feats["egemaps"] < lo) | (feats["egemaps"] > hi)).sum())
        rank = float((ref < n_out).mean() * 100)
        if rank >= 95:
            st.warning(f"**Unusual recording**: {n_out} of 88 voice features fall outside the range of "
                       f"training clips, more atypical than {rank:.0f}% of them. Treat the result with caution.")
        else:
            st.caption(f"Typicality check: {n_out} of 88 features outside the 1–99% training range "
                       f"(normal: more atypical than only {rank:.0f}% of training clips).")
        for w in res["warnings"]:
            if "outside the range" not in w:
                st.warning(w)
        st.caption("Scores are model outputs, not a measurement of how the speaker feels. "
                   "Trained on acted English speech only.")

    tabs = st.tabs(["🧠 Explanation", "🔊 Signal", "📋 Voice features", "🧪 Robustness simulation"])

    # ---------------- explanation ----------------
    with tabs[0]:
        train = train_egemaps()
        k_name = st.selectbox("Explain the evidence for", CLASSES, index=CLASSES.index(label),
                              help="Why this emotion, or why not another one?")
        k = CLASSES.index(k_name)
        expl = U.local_shap(feats["egemaps"], bundle, k)
        expl["percentile"] = U.percentiles(expl["value"].values, train[list(expl["feature"])])
        st.markdown(U.explanation_sentence(expl, k_name))
        ca, cb = st.columns([1.7, 1])
        with ca:
            top_n = expl.head(12).iloc[::-1]
            fig = go.Figure(go.Bar(
                x=top_n["contribution"], y=top_n["readable"], orientation="h",
                marker_color=["#2A78D6" if v > 0 else "#EB6834" for v in top_n["contribution"]],
                customdata=np.c_[top_n["value"], top_n["percentile"], top_n["group"]],
                hovertemplate="<b>%{y}</b><br>contribution %{x:+.2f} log-odds<br>value %{customdata[0]:.3f}"
                              "<br>training percentile %{customdata[1]:.0f}<br>group %{customdata[2]}<extra></extra>"))
            fig.add_vline(x=0, line_color="gray")
            fig.update_xaxes(title="contribution (log-odds) · blue = toward · orange = away")
            show(style(fig, 470, legend=False, title=f"Top 12 voice features for '{k_name}' (eGeMAPS model, exact linear SHAP)"))
        with cb:
            g = expl.groupby("group")["contribution"].sum().reindex(list(GRP))
            fig = go.Figure(go.Bar(x=g.index, y=g.values, marker_color=[GRP[i] for i in g.index],
                                   text=[f"{v:+.2f}" for v in g.values], textposition="outside"))
            fig.add_hline(y=0, line_color="gray")
            show(style(fig, 250, legend=False, title="Net evidence by feature group").update_layout(yaxis_title="log-odds"))
            split = U.fusion_evidence_split(feats["egemaps"], feats["wavlm"][bundle["layer"]], bundle, k)
            fig = go.Figure([go.Bar(y=[""], x=[v * 100], name=k, orientation="h", marker_color=c,
                                    text=f"{v:.0%}", textposition="inside")
                             for (k, v), c in zip(split.items(), ["#EB6834", "#7B5BD6"])])
            fig.update_layout(barmode="stack")
            fig.update_xaxes(range=[0, 100], title="% of the main model's evidence for this emotion")
            show(style(fig, 200, title="eGeMAPS vs WavLM evidence (fusion model)"))
        st.caption("Contributions are relative to the average training clip. Correlated features (e.g. loudness "
                   "and overall sound level) can receive opposite signs that largely cancel: read them together. "
                   "Group-level evidence is more "
                   "trustworthy than single features: in Phase 8 (E3), feature rankings changed across "
                   "datasets but group shares stayed stable. SHAP shows what the model uses; it does not "
                   "prove that a voice property causes an emotion.")

    # ---------------- signal ----------------
    with tabs[1]:
        v = views_for(y)
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            row_heights=[0.2, 0.3, 0.3, 0.2],
                            subplot_titles=("Waveform (normalised) and energy", "Spectrogram (STFT, dB)",
                                            "Mel-spectrogram (64 bands)", "Pitch contour (F0, pYIN)"))
        wav = v["wave"] / (np.abs(v["wave"]).max() + 1e-9)
        fig.add_trace(go.Scatter(x=v["t_wave"], y=wav, line=dict(color="#8A8F98", width=1),
                                 name="waveform"), 1, 1)
        rms = (v["rms_db"] - v["rms_db"].max()) / 80 + 1
        fig.add_trace(go.Scatter(x=v["t"], y=np.clip(rms, 0, 1), line=dict(color="#EB6834", width=2),
                                 name="energy (relative)"), 1, 1)
        fig.add_trace(go.Heatmap(x=v["t"], y=v["freqs"], z=v["spec_db"], colorscale="Magma",
                                 showscale=False, name="spectrogram"), 2, 1)
        fig.add_trace(go.Heatmap(x=v["t"], y=np.arange(64), z=v["mel_db"], colorscale="Viridis",
                                 showscale=False, name="mel"), 3, 1)
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
        m1.metric("Duration (trimmed)", f"{duration:.2f} s")
        m2.metric("Median pitch", f"{np.median(f0):.0f} Hz" if len(f0) else "—")
        m3.metric("Pitch range (10–90%)", f"{np.percentile(f0, 90) - np.percentile(f0, 10):.0f} Hz" if len(f0) else "—")
        m4.metric("Voiced frames", f"{len(f0) / max(1, len(v['f0'])):.0%}")
        st.caption("These views show what the models receive: the processed 16 kHz signal after silence "
                   "trimming. The pitch contour is estimated with pYIN for display; the model's own pitch "
                   "features come from openSMILE.")

    # ---------------- feature summary ----------------
    with tabs[2]:
        train = train_egemaps()
        names = bundle["egemaps_names"]
        key = ["F0semitoneFrom27.5Hz_sma3nz_amean", "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
               "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2", "loudness_sma3_amean",
               "loudness_sma3_meanRisingSlope", "equivalentSoundLevel_dBp", "HNRdBACF_sma3nz_amean",
               "jitterLocal_sma3nz_amean", "shimmerLocaldB_sma3nz_amean", "alphaRatioV_sma3nz_amean",
               "hammarbergIndexV_sma3nz_amean", "spectralFlux_sma3_amean", "F1frequency_sma3nz_amean",
               "loudnessPeaksPerSec", "VoicedSegmentsPerSec", "MeanUnvoicedSegmentLength"]
        vals = pd.Series(feats["egemaps"], index=names)
        T = train[names]
        rows = []
        for f in key:
            p = float((T[f] < vals[f]).mean() * 100)
            rows.append({"Feature": U.readable(f), "Group": U.feature_group(f), "Value": round(float(vals[f]), 3),
                         "Training median": round(float(T[f].median()), 3),
                         "Percentile": round(p), "Level": U.describe_level(p)})
        tab = pd.DataFrame(rows)
        st.dataframe(tab, width="stretch", hide_index=True, height=35 * (len(tab) + 1) + 3,
                     column_config={"Percentile": st.column_config.ProgressColumn(
                         "Percentile vs training clips", min_value=0, max_value=100, format="%d")})
        st.caption("16 of the 88 eGeMAPS features, chosen because they have a clear acoustic meaning. "
                   "Percentile = share of the 8,498 training clips with a lower value.")
        with st.expander("All 88 eGeMAPS features"):
            allp = U.percentiles(vals.values, T)
            st.dataframe(pd.DataFrame({"Feature": [U.readable(n) for n in names], "openSMILE name": names,
                                       "Group": [U.feature_group(n) for n in names],
                                       "Value": vals.values.round(4), "Percentile": allp.round(0)}),
                         width="stretch", hide_index=True, height=400)

    # ---------------- robustness ----------------
    with tabs[3]:
        st.markdown("Does the prediction survive changes that should **not** change the emotion? "
                    "Each row re-runs the full pipeline on a modified copy of this recording.")
        if st.button("Run robustness simulation (≈10–20 s)", type="primary"):
            out = []
            prog = st.progress(0.0)
            for i, (_, kind, amt, lab) in enumerate(U.PERTURBATIONS):
                yy = y if kind is None else U.perturb(y, kind, amt)
                ff = features_for(yy)
                rr = predict_features(ff["egemaps"], ff["wavlm"], bundle)
                out.append({"Condition": lab, **{c: float(rr["proba"][c]) for c in CLASSES},
                            "Prediction": rr["label"], "eGeMAPS model": rr["explainer_label"]})
                prog.progress((i + 1) / len(U.PERTURBATIONS))
            st.session_state["robust"] = (clip_key, pd.DataFrame(out))
        if "robust" in st.session_state and st.session_state["robust"][0] == clip_key:
            df = st.session_state["robust"][1]
            fig = go.Figure(go.Heatmap(z=df[CLASSES].values * 100, x=CLASSES, y=df["Condition"],
                                       colorscale="Blues", zmin=0, zmax=100,
                                       text=(df[CLASSES].values * 100).round(0), texttemplate="%{text:.0f}",
                                       colorbar=dict(title="%")))
            fig.update_yaxes(autorange="reversed")
            show(style(fig, 380, legend=False, title="Model scores under each modification"))
            stable = (df["Prediction"] == df["Prediction"].iloc[0]).mean()
            st.dataframe(df[["Condition", "Prediction", "eGeMAPS model"]], hide_index=True, width="stretch")
            st.markdown(f"Prediction unchanged in **{stable:.0%}** of conditions. Volume changes are a direct "
                        "test of the loudness reliance found in Phase 7 (quiet → *sad*): if a 12 dB quieter copy "
                        "flips to *sad*, the model is using recording level, which a microphone also changes.")


# ===========================================================================
# PAGE 2 — How the model decides
# ===========================================================================
def page_global():
    st.markdown('<div class="hero"><h1>How the model decides</h1><p>Global SHAP explanations from Phase 7: '
                'each clip explained by a model that never heard its speaker, then averaged.</p></div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 2])
    corpus = c1.selectbox("Corpus", ["ravdess", "cremad"], format_func=U.NAME.get)
    model = c2.selectbox("Model", ["logreg", "lgbm"],
                         format_func={"logreg": "Logistic Regression", "lgbm": "LightGBM"}.get)
    gi = U.load_csv(f"phase7/global_importance_{corpus}_{model}.csv")
    if gi is None:
        st.error("Phase 7 results not found."); return
    gi["readable"] = gi["feature"].map(U.readable)
    top = gi.head(15).iloc[::-1]
    ca, cb = st.columns([1.6, 1])
    with ca:
        fig = go.Figure(go.Bar(x=top["importance"], y=top["readable"], orientation="h", showlegend=False,
                               marker_color=[GRP[g] for g in top["group"]],
                               customdata=top[CLASSES].values,
                               hovertemplate="<b>%{y}</b><br>mean |SHAP| %{x:.3f}<br>" +
                               "<br>".join(f"{c}: %{{customdata[{i}]:.2f}}" for i, c in enumerate(CLASSES)) +
                               "<extra></extra>"))
        for g, col in GRP.items():
            fig.add_trace(go.Bar(x=[None], y=[None], marker_color=col, name=g))
        fig.update_xaxes(title="mean |SHAP| (log-odds)")
        show(style(fig, 520, title=f"Top 15 features · {U.NAME[corpus]} · {model}"))
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
            show(style(fig, 300, legend=False))
            st.caption("Spectral leads in total only because it has 43 of 88 features. **Per feature, Energy "
                       "is the most informative group**, in line with Paper 5.")
        sp = p7.get("shap_vs_permutation", {}).get(corpus)
        if sp:
            st.metric("SHAP vs permutation importance (Spearman)", f"{sp['spearman']:.2f}",
                      help="A second, independent explanation method agrees with SHAP.")
            st.caption(f"Top-10 overlap between the two methods: {sp['top10_overlap']}/10")

    st.subheader("Do the two corpora rely on the same features?")
    a = U.load_csv("phase7/global_importance_ravdess_logreg.csv")
    b = U.load_csv("phase7/global_importance_cremad_logreg.csv")
    if a is not None and b is not None:
        m = a[["feature", "group", "importance"]].merge(b[["feature", "importance"]], on="feature",
                                                        suffixes=("_r", "_c"))
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
            st.caption("Each dot is one eGeMAPS feature (LogReg). The CREMA-D model leans on **loudness**, "
                       "the RAVDESS model on **MFCCs and formants**. Overall profiles correlate moderately, "
                       "but the top features differ: explanations learned on one corpus do not simply carry "
                       "over (tested formally in Phase 8, E3).")

    st.subheader("What drives each emotion")
    d = U.load_csv("phase7/per_emotion_drivers.csv")
    if d is not None:
        c1, c2 = st.columns([1, 3])
        cn = c1.selectbox("Corpus ", ["RAVDESS", "CREMA-D"])
        en = c1.selectbox("Emotion", CLASSES)
        sub = d[(d["corpus"] == cn) & (d["emotion"] == en)].copy()
        sub.insert(2, "Feature (readable)", sub["feature"].map(U.readable))
        c2.dataframe(sub.drop(columns=["corpus", "emotion"]), hide_index=True, width="stretch")

    p7 = U.load_json("phase7_summary.json")
    np_ = p7.get("no_pitch_shortcut_check")
    if np_:
        with st.expander("Shortcut check: does the model treat 'no pitch detected' as sad?"):
            c1, c2 = st.columns(2)
            c1.metric("UAR with the 39 no-pitch clips", f"{np_['uar_with']:.1%}")
            c2.metric("UAR after removing them", f"{np_['uar_without']:.1%}")
            st.caption("No measurable difference: no harmful shortcut (Phase 7, §8).")


# ===========================================================================
# PAGE 3 — Model performance
# ===========================================================================
def page_performance():
    st.markdown('<div class="hero"><h1>Model performance</h1><p>Speaker-independent 5-fold cross-validation '
                '(no speaker in both training and test). Metric: UAR = average recall over the 6 emotions.'
                '</p></div>', unsafe_allow_html=True)
    e1 = U.load_csv("phase6/e1_summary.csv")
    if e1 is not None:
        e1 = e1.copy()
        e1["Model"] = e1["model"].map({"logreg": "LogReg", "svm": "SVM (RBF)", "lgbm": "LightGBM", "mlp": "MLP"})
        e1["Features"] = e1["features"].map({"egemaps": "eGeMAPS", "wavlm": "WavLM", "fusion": "Fusion"})
        e1["Corpus"] = e1["corpus"].map(U.NAME)
        fig = px.bar(e1, x="Model", y=e1["uar_mean"] * 100, color="Features", barmode="group",
                     facet_col="Corpus", error_y=e1["uar_std"] * 100,
                     color_discrete_map={"eGeMAPS": "#EB6834", "WavLM": "#7B5BD6", "Fusion": "#2A78D6"},
                     category_orders={"Model": ["LogReg", "SVM (RBF)", "LightGBM", "MLP"],
                                      "Corpus": ["RAVDESS", "CREMA-D"]})
        fig.add_hline(y=CHANCE, line_dash="dot", line_color="gray")
        fig.update_yaxes(title="UAR (%) · mean ± SD over folds", range=[0, 100])
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_yaxes(title_text="", col=2)
        show(style(fig, 450, bottom=True, title="Within-corpus results (Phase 6) · dotted line = chance (16.7%) · KNN excluded by design"))
        best = e1.sort_values("uar_mean", ascending=False).groupby("Corpus").head(1)
        cols = st.columns(len(best))
        for col, r in zip(cols, best.itertuples()):
            col.metric(f"Best on {r.Corpus}", f"{r.uar_mean:.1%} UAR", f"{r.Model} · {r.Features}", delta_color="off")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Why speaker-independent splits matter")
        e0 = U.load_csv("phase6/e0_leakage.csv")
        if e0 is not None:
            e0 = e0.copy(); e0["label"] = e0["corpus"].map(U.NAME) + " · " + e0["features"]
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
        corpus = a.selectbox("Corpus", ["ravdess", "cremad"], format_func=U.NAME.get, key="cm_c")
        fs = b.selectbox("Features", ["fusion", "wavlm", "egemaps"], key="cm_f")
        mdl = c.selectbox("Model", ["logreg", "svm", "lgbm", "mlp"], key="cm_m")
        p = U.RESULTS / "phase6" / f"e1_{corpus}_{fs}_{mdl}_oof.npy"
        if p.exists():
            from sklearn.metrics import confusion_matrix
            yt, yp = U.labels_for(corpus), np.load(p)
            cm = confusion_matrix(yt, yp, labels=range(6), normalize="true") * 100
            fig = go.Figure(go.Heatmap(z=cm, x=CLASSES, y=CLASSES, colorscale="Blues", zmin=0, zmax=100,
                                       text=cm.round(0), texttemplate="%{text:.0f}", showscale=False))
            fig.update_yaxes(autorange="reversed", title="true")
            fig.update_xaxes(title="predicted")
            show(style(fig, 360, legend=False))
            st.caption("Row-normalised (%), out-of-fold predictions: every clip predicted by a model that "
                       "never heard its speaker.")
        else:
            st.info("Predictions for this combination were not saved.")

    e6 = U.load_json("phase8_e6_summary.json")
    if e6:
        st.subheader("Standard evaluation (Phase 8, E6)")
        t1, t2, t3, t4 = st.tabs(["ROC-AUC", "Calibration", "Overfitting check", "Learning curve"])
        with t1:
            r = pd.DataFrame(e6["roc_auc"])
            r["config"] = r["model"] + " · " + r["features"]
            fig = px.bar(r, x="config", y="macro_auc", color="corpus", barmode="group",
                         color_discrete_sequence=["#2A78D6", "#EB6834"])
            fig.update_yaxes(range=[0.5, 1], title="macro ROC-AUC (one-vs-rest)")
            show(style(fig, 360))
            img = U.RESULTS / "figures" / "p8_e6_roc.png"
            if img.exists():
                with st.expander("ROC curves per emotion"):
                    st.image(str(img))
        with t2:
            ece = e6["calibration_ece"]
            cols = st.columns(4)
            for col, (k, v) in zip(cols, ece.items()):
                name, fs = k.split("_")
                col.metric(f"ECE · {U.NAME[name]} · {fs}", f"{v:.3f}")
            img = U.RESULTS / "figures" / "p8_e6_calibration.png"
            if img.exists():
                st.image(str(img))
            st.caption("ECE = average gap between stated confidence and actual accuracy. Measured on new "
                       "speakers from the same corpora; on a new microphone the gap grows.")
        with t3:
            o = pd.DataFrame(e6["overfitting"])
            o["config"] = o["corpus"] + " · " + o["model"] + " · " + o["features"]
            o = o.sort_values("gap_pts")
            fig = go.Figure()
            for r_ in o.itertuples():
                fig.add_trace(go.Scatter(x=[r_.val_uar * 100, r_.train_uar * 100], y=[r_.config] * 2,
                                         mode="lines", line=dict(color="rgba(140,140,140,.6)", width=3),
                                         showlegend=False))
            fig.add_trace(go.Scatter(x=o["val_uar"] * 100, y=o["config"], mode="markers", name="validation (new speakers)",
                                     marker=dict(size=11, color="#2A78D6")))
            fig.add_trace(go.Scatter(x=o["train_uar"] * 100, y=o["config"], mode="markers", name="training",
                                     marker=dict(size=11, color="#EB6834")))
            fig.update_xaxes(title="UAR (%)")
            show(style(fig, 470))
            st.caption("LightGBM memorises its training speakers (≈100%) without beating LogReg on new "
                       "speakers, one reason LogReg is the main model.")
        with t4:
            lc = e6["learning_curve"]
            fig = go.Figure()
            for fs_, col in [("egemaps", "#EB6834"), ("wavlm", "#7B5BD6")]:
                d = lc[fs_]
                xs = [int(k) for k in d["val_uar"]]
                fig.add_trace(go.Scatter(x=xs, y=[v * 100 for v in d["val_uar"].values()], name=f"{fs_} · new speakers",
                                         line=dict(color=col, width=3)))
                fig.add_trace(go.Scatter(x=xs, y=[v * 100 for v in d["train_uar"].values()], name=f"{fs_} · training",
                                         line=dict(color=col, dash="dash")))
            fig.update_xaxes(title="number of training speakers (CREMA-D)")
            fig.update_yaxes(title="UAR (%)")
            show(style(fig, 380))
            st.caption("WavLM keeps improving with more speakers; eGeMAPS plateaus near 51%.")

    p9 = U.load_json("phase9_final_model_summary.json")
    if p9.get("heldout_check"):
        st.subheader("Final model recipe on held-out speakers")
        st.dataframe(pd.DataFrame(p9["heldout_check"]), hide_index=True, width="stretch")
        st.caption("Pooled training (both corpora) vs the single-corpus Phase 6 model, on the same fold-0 speakers.")


# ===========================================================================
# PAGE 4 — Cross-corpus & errors
# ===========================================================================
def page_cross():
    st.markdown('<div class="hero"><h1>Cross-corpus generalisation & error analysis</h1><p>Train on one corpus, '
                'test on the other (Phase 8). 95% confidence intervals from a speaker-level bootstrap.</p></div>',
                unsafe_allow_html=True)
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
                fig.add_trace(go.Scatter(x=list(s.index), y=list(s.values), mode="markers", name="in-domain (target)" if i == 0 else None,
                                         showlegend=i == 0, marker=dict(symbol="line-ew-open", size=28, color="#D64545",
                                                                         line=dict(width=3))), row=1, col=i + 1)
        fig.add_hline(y=CHANCE, line_dash="dot", line_color="gray")
        fig.update_yaxes(title="UAR on the other corpus (%)", range=[0, 90])
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_yaxes(title_text="", col=2)
        show(style(fig, 450, bottom=True, title="Cross-corpus UAR, Logistic Regression · red line = in-domain score · dotted = chance"))
        st.markdown("**H1: does WavLM lose relatively more than eGeMAPS?**")
        t = h1[["direction", "norm", "wavlm_minus_egemaps_rel_drop", "H1_supported"]].rename(columns={
            "wavlm_minus_egemaps_rel_drop": "Δ relative drop (WavLM − eGeMAPS)", "H1_supported": "H1 supported"})
        st.dataframe(t, hide_index=True, width="stretch")
        st.info("H1 depends on the direction: trained on the small RAVDESS, WavLM loses relatively more; "
                "trained on the diverse CREMA-D, it transfers much better. Per-corpus normalisation helps every "
                "configuration.")
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
        order = ["resample", "speaker", "corpus", "random"]
        fig = px.box(pairs, x="level", y="spearman", color="rep", category_orders={"level": order},
                     color_discrete_sequence=["#2A78D6", "#1BAF7A", "#7B5BD6"], points=False)
        fig.update_yaxes(title="Spearman between two importance rankings")
        fig.update_xaxes(title="the two models were trained on…")
        c1, c2 = st.columns([1.6, 1])
        with c1:
            show(style(fig, 400))
        with c2:
            st.markdown("- **resample**: two bootstrap samples of the same data\n"
                        "- **speaker**: two disjoint speaker groups, same corpus\n"
                        "- **corpus**: one corpus each\n"
                        "- baseline, two random rankings: median ≈ 0.00 (not drawn)")
            st.success("H2 supported for all three models (Mann–Whitney p < 10⁻¹⁰). The interpretable "
                       "LogReg·eGeMAPS explanations are the most stable; WavLM dimensions the least.")

    st.subheader("Stable-core feature selection (H3)")
    h3 = U.load_csv("phase8/e4_h3_test.csv")
    if h3 is not None:
        fig = go.Figure()
        h3 = h3.copy()
        h3["setting"] = h3["direction"] + " · " + h3["norm"] + " · k=" + h3["k"].astype(str)
        lo = [U.parse_ci(s) for s in h3["ci95"]]
        h3["lo"] = [a[0] for a in lo]; h3["hi"] = [a[1] for a in lo]
        fig.add_trace(go.Scatter(x=h3["stable_minus_top_pts"], y=h3["setting"], mode="markers",
                                 marker=dict(size=11, color=np.where(h3["significant"],
                                                                     np.where(h3["stable_minus_top_pts"] > 0, "#1BAF7A", "#D64545"),
                                                                     "#8A8F98")),
                                 error_x=dict(type="data", symmetric=False, array=h3["hi"] - h3["stable_minus_top_pts"],
                                              arrayminus=h3["stable_minus_top_pts"] - h3["lo"])))
        fig.add_vline(x=0, line_color="gray")
        fig.update_xaxes(title="UAR difference, stable core − usual top-k (points, 95% CI)")
        show(style(fig, 440, legend=False))
        st.caption("Green = significant gain, red = significant loss, grey = not distinguishable from 0. "
                   "H3 is inconclusive: stability helps diagnose models but is not a reliable selection recipe.")

    p8 = U.load_json("phase8_summary.json")
    st.subheader("Error analysis (E5)")
    if p8.get("E5_per_class"):
        pc = pd.DataFrame(p8["E5_per_class"])
        long = pc.melt(id_vars=["direction", "emotion"], value_vars=["in_domain_recall", "cross_recall"],
                       var_name="setting", value_name="recall")
        long["setting"] = long["setting"].map({"in_domain_recall": "in-domain", "cross_recall": "cross-corpus"})
        fig = px.bar(long, x="emotion", y="recall", color="setting", barmode="group", facet_col="direction",
                     color_discrete_sequence=["#8A8F98", "#D64545"], category_orders={"emotion": CLASSES})
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1] + " (tested on target)"))
        fig.update_yaxes(title_text="", col=2)
        show(style(fig, 400, bottom=True, title="Which emotions break across corpora (fusion LogReg, per-corpus norm)"))
        st.caption("RAVDESS-trained models almost never recognise CREMA-D neutral (recall 0.09): the acted "
                   "'neutral' style differs between corpora. Sad and happy break in the other direction.")
    c1, c2 = st.columns(2)
    if p8.get("E5_groups"):
        with c1:
            st.markdown("**Speaker gender and age (UAR)**")
            st.dataframe(pd.DataFrame(p8["E5_groups"]), hide_index=True, width="stretch", height=320)
    if p8.get("E5_label_quality"):
        with c2:
            lq = p8["E5_label_quality"]
            st.markdown("**Label quality: did CREMA-D listeners agree with the actor's intent?**")
            st.dataframe(pd.DataFrame(lq["table"]), hide_index=True, width="stretch")
            st.metric("Human recognition vs model recall per emotion (Spearman)", f"{lq['human_vs_model_spearman']:.2f}")
            st.caption("What humans find hard, the model finds hard: scores on CREMA-D are partly limited by label noise.")


# ===========================================================================
# PAGE 5 — About
# ===========================================================================
def page_about():
    st.markdown('<div class="hero"><h1>About this system</h1><p>Emotion Recognition Using Explainable Machine '
                'Learning · Team Lucid</p></div>', unsafe_allow_html=True)
    st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#2A78D622", color="#2A78D6", fontname="Helvetica"];
  audio [label="Audio\\n(any rate, mono/stereo)"]; pre [label="Preprocess\\nmono · 16 kHz · trim silence"];
  eg [label="eGeMAPS\\n88 named voice features"]; wl [label="WavLM base-plus\\nlayer 6, mean-pooled (768)"];
  pred [label="Fusion LogReg\\n(predictor)"]; expm [label="eGeMAPS LogReg\\n(explainer)"];
  out [label="Emotion + scores", fillcolor="#1BAF7A22", color="#1BAF7A"];
  shap [label="Exact linear SHAP\\n+ percentiles", fillcolor="#EB683422", color="#EB6834"];
  audio -> pre -> eg; pre -> wl; eg -> pred; wl -> pred; eg -> expm; pred -> out; expm -> shap;
}""")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Research question.** How well do emotion models *and their explanations* transfer from one corpus to another?

**Data.** RAVDESS (24 actors, 1,056 clips used) and CREMA-D (91 actors, 7,442 clips), six shared emotions.

**Key design choices**
- Speaker-independent evaluation everywhere (leakage inflated UAR by up to 9 points).
- Two representations: interpretable eGeMAPS and self-supervised WavLM.
- KNN excluded by project rule; LogReg, SVM, LightGBM and MLP compared.
- SHAP is exact for the models explained (linear and tree), cross-checked with permutation importance.
""")
    with c2:
        st.markdown("""
**Limitations**
- Acted, exaggerated English speech; real emotions are subtler.
- Two corpora only; cross-corpus findings are about one pair.
- Scores are over-confident on new microphones and rooms.
- Explanations show what the model uses, not what causes emotions.

**Ethics**
- Not for decisions about people (hiring, policing, health).
- Accuracy differs slightly by gender (Phase 8, E5).
- Recordings are processed locally and not stored by this app.
""")
    st.caption("Datasets: RAVDESS (CC BY-NC-SA 4.0), CREMA-D (ODbL). Models: microsoft/wavlm-base-plus, "
               "openSMILE eGeMAPSv02.")


{"Analyse a recording": page_analyse, "How the model decides": page_global,
 "Model performance": page_performance, "Cross-corpus & errors": page_cross,
 "About & limitations": page_about}[page]()
