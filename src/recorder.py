"""
Raw microphone recorder for the app (Streamlit custom component, v2 API).

Why. Streamlit's built-in st.audio_input asks the browser for audio with its default "voice call"
processing: automatic gain control (quiet speech turned up, shouting turned down), noise
suppression and echo cancellation. The model's strongest cues are loudness and voice quality
(Phase 7), and none of its training or testing included that processing. This recorder asks for
the microphone with all three switched OFF, records mono audio, resamples it to 16 kHz in the
browser and hands Python a WAV file. It also reports which processing the browser really applied.

    from src.recorder import raw_recorder
    rec = raw_recorder("Record one sentence", key="main")   # None, or dict(wav=bytes, seconds, processing)
"""
from __future__ import annotations

import base64

import streamlit as st

HTML = """
<div class="lr">
  <button class="lr-btn" type="button"><span class="lr-dot"></span><span class="lr-txt">Record</span></button>
  <div class="lr-mid">
    <div class="lr-label"></div>
    <div class="lr-meter"><div class="lr-fill"></div></div>
    <div class="lr-status">Press Record, say one sentence, then press Stop.</div>
  </div>
  <div class="lr-time">0.0 s</div>
</div>
"""

CSS = """
.lr { display:flex; align-items:center; gap:14px; padding:12px 14px; border-radius:12px;
      background: var(--st-secondary-background-color); font-family: var(--st-font); }
.lr-btn { display:flex; align-items:center; gap:8px; border:none; border-radius:999px; padding:9px 16px;
          background: var(--st-primary-color); color:#fff; font-weight:600; cursor:pointer; font-size:14px;
          font-family: inherit; min-width:104px; justify-content:center; }
.lr-btn.on { background:#C93A3A; }
.lr-dot { width:10px; height:10px; border-radius:50%; background:#fff; }
.lr-btn.on .lr-dot { border-radius:2px; }
.lr-mid { flex:1; min-width:0; }
.lr-label { font-size:14px; color: var(--st-text-color); margin-bottom:6px; }
.lr-meter { height:6px; border-radius:3px; background: rgba(128,128,128,.25); overflow:hidden; }
.lr-fill { height:100%; width:0%; background: var(--st-primary-color); transition: width .08s linear; }
.lr-status { font-size:12px; margin-top:6px; color: var(--st-text-color); opacity:.7; }
.lr-time { font-variant-numeric: tabular-nums; font-size:14px; color: var(--st-text-color); opacity:.8; width:54px;
           text-align:right; }
"""

JS = r"""
function encodeWav(samples, sr) {
  const buf = new ArrayBuffer(44 + samples.length * 2), v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); w(8, 'WAVE'); w(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36, 'data'); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Uint8Array(buf);
}
function toBase64(bytes) {
  let s = ''; const step = 0x8000;
  for (let i = 0; i < bytes.length; i += step) s += String.fromCharCode.apply(null, bytes.subarray(i, i + step));
  return btoa(s);
}
async function to16k(samples, sr) {
  if (sr === 16000) return samples;
  const n = Math.ceil(samples.length * 16000 / sr);
  const off = new OfflineAudioContext(1, n, 16000);
  const b = off.createBuffer(1, samples.length, sr); b.copyToChannel(samples, 0);
  const s = off.createBufferSource(); s.buffer = b; s.connect(off.destination); s.start();
  return (await off.startRendering()).getChannelData(0);
}

export default function (component) {
  const { parentElement, data, setStateValue } = component;
  const root = parentElement.querySelector('.lr');
  const btn = root.querySelector('.lr-btn'), txt = root.querySelector('.lr-txt');
  const fill = root.querySelector('.lr-fill'), status = root.querySelector('.lr-status');
  const time = root.querySelector('.lr-time');
  root.querySelector('.lr-label').textContent = (data && data.label) || '';
  if (data && data.done) status.textContent = data.done;
  const maxSec = (data && data.max_seconds) || 10;
  const R = root.__rec || (root.__rec = { on: false });
  if (root.__wired) return () => {};
  root.__wired = true;

  async function start() {
    try {
      R.stream = await navigator.mediaDevices.getUserMedia({ audio: {
        echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 } });
    } catch (e) {
      status.textContent = 'The microphone is blocked: ' + e.message + '. Allow it in the browser and try again.';
      return;
    }
    const track = R.stream.getAudioTracks()[0];
    R.settings = track.getSettings ? track.getSettings() : {};
    R.ctx = new AudioContext();
    R.sr = R.ctx.sampleRate;
    const src = R.ctx.createMediaStreamSource(R.stream);
    R.proc = R.ctx.createScriptProcessor(4096, 1, 1);
    R.chunks = [];
    R.proc.onaudioprocess = (e) => {
      const d = e.inputBuffer.getChannelData(0);
      R.chunks.push(new Float32Array(d));
      let peak = 0; for (let i = 0; i < d.length; i++) peak = Math.max(peak, Math.abs(d[i]));
      fill.style.width = Math.min(100, 100 * Math.sqrt(peak)) + '%';
    };
    src.connect(R.proc); R.proc.connect(R.ctx.destination);
    R.t0 = performance.now(); R.on = true;
    btn.classList.add('on'); txt.textContent = 'Stop';
    status.textContent = 'Recording from the raw microphone... press Stop when you finish the sentence.';
    R.timer = setInterval(() => {
      const t = (performance.now() - R.t0) / 1000;
      time.textContent = t.toFixed(1) + ' s';
      if (t >= maxSec) stop();
    }, 100);
  }

  async function stop() {
    if (!R.on) return;
    R.on = false; clearInterval(R.timer);
    btn.classList.remove('on'); txt.textContent = 'Record again'; fill.style.width = '0%';
    try { R.proc.disconnect(); } catch (e) {}
    R.stream.getTracks().forEach((t) => t.stop());
    await R.ctx.close();
    let n = 0; R.chunks.forEach((c) => (n += c.length));
    const all = new Float32Array(n); let o = 0;
    R.chunks.forEach((c) => { all.set(c, o); o += c.length; });
    const seconds = n / R.sr;
    if (seconds < 0.5) { status.textContent = 'Too short. Record the whole sentence.'; return; }
    status.textContent = 'Processing...';
    const y16 = await to16k(all, R.sr);
    const s = R.settings || {};
    setStateValue('audio', {
      wav: toBase64(encodeWav(y16, 16000)), seconds: seconds, nonce: Date.now(),
      processing: { autoGainControl: s.autoGainControl ?? null, noiseSuppression: s.noiseSuppression ?? null,
                    echoCancellation: s.echoCancellation ?? null, deviceSampleRate: R.sr },
    });
    status.textContent = 'Recorded ' + seconds.toFixed(1) + ' s.';
  }

  btn.onclick = () => (R.on ? stop() : start());
  return () => { if (R.on) { try { R.stream.getTracks().forEach((t) => t.stop()); } catch (e) {} } };
}
"""

_component = st.components.v2.component("lucid_raw_recorder", html=HTML, css=CSS, js=JS)


def raw_recorder(label: str, key: str, max_seconds: int = 10) -> dict | None:
    """Show the recorder; return None or {"wav": bytes, "seconds": float, "processing": dict, "nonce": int}."""
    res = _component(key=key, data={"label": label, "max_seconds": max_seconds},
                     on_audio_change=lambda: None, height="content")
    val = getattr(res, "audio", None)
    if not val or not val.get("wav"):
        return None
    return {"wav": base64.b64decode(val["wav"]), "seconds": float(val.get("seconds", 0)),
            "processing": val.get("processing") or {}, "nonce": val.get("nonce")}


def processing_note(processing: dict) -> str:
    """Plain-language note on what the browser actually did to the microphone signal."""
    on = [name for name, k in [("automatic gain control", "autoGainControl"), ("noise suppression", "noiseSuppression"),
                               ("echo cancellation", "echoCancellation")] if processing.get(k) is True]
    if not processing:
        return ""
    if not on:
        return "Raw microphone: automatic gain control, noise suppression and echo cancellation are off."
    return ("Your browser kept " + ", ".join(on) + " on even though the app asked for them off; "
            "results may be less reliable. Chrome normally honours the request.")
