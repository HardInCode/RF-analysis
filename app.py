import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os, tempfile, time, base64
from pathlib import Path
from sca_utils import (
    generate_synthetic_traces, export_to_wav, run_cpa, plot_cpa_results,
    align_traces, visualize_alignment, compare_keys, plot_full_key_recovery,
    align_max_peak,
)

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RF-SCA Lab — Side-Channel Analysis Pipeline",
    page_icon="📡", layout="wide", initial_sidebar_state="expanded",
)
css = Path(__file__).parent / ".streamlit" / "style.css"
st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
_defaults = dict(
    data_loaded=False, data_source=None, traces=None,
    plaintexts=None, aligned_traces=None, processing_done=False,
    full_key_recovery=False, recovered_key=None, fs=1_000_000,
)
for k, v in _defaults.items():
    st.session_state.setdefault(k, v)
ss = st.session_state  # shorthand

if ss.traces is not None and ss.plaintexts is not None:
    ss.processing_done = True

# ── Helpers ───────────────────────────────────────────────────────────────────
def dark_fig(fig):
    """Apply dark theme to a matplotlib figure in-place."""
    bg, panel, grid, txt = "#0a0e1a", "#0f1629", "#1e2a45", "#94a3b8"
    fig.patch.set_facecolor(bg)
    for ax in fig.get_axes():
        ax.set_facecolor(panel)
        ax.tick_params(colors=txt, labelsize=8)
        ax.xaxis.label.set_color(txt); ax.yaxis.label.set_color(txt)
        ax.title.set_color("#e2e8f0"); ax.title.set_fontsize(10); ax.title.set_fontweight("semibold")
        for s in ax.spines.values(): s.set_color("#1e293b")
        ax.grid(True, color=grid, lw=0.5, alpha=0.7, ls="--"); ax.set_axisbelow(True)
    fig.tight_layout(pad=1.5)
    return fig

def sec(icon, title, desc=""):
    sub = f'<div class="sec-sub">{desc}</div>' if desc else ""
    st.markdown(
        f'<div class="sec-hdr"><span style="font-size:1.1rem">{icon}</span>'
        f'<div><div class="sec-title">{title}</div>{sub}</div></div>',
        unsafe_allow_html=True,
    )

def key_html(key_arr, true_key=None):
    """Render key bytes as styled chips, green/red if true_key provided."""
    chips = ""
    for i, b in enumerate(key_arr):
        cls = ""
        if true_key is not None and not isinstance(true_key, (int, np.integer)):
            cls = "ok" if b == true_key[i] else "bad"
        chips += f'<span class="kb {cls}">0x{b:02X}</span>'
    return f'<div class="key-box">{chips}</div>'

def scalar_key(true_key, idx=0):
    """Extract scalar key byte from int or array."""
    if true_key is None: return None
    if isinstance(true_key, (int, np.integer)): return int(true_key)
    return int(true_key[idx]) if idx < len(true_key) else None

def plot_show(fig):
    dark_fig(fig); st.pyplot(fig); plt.close(fig)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="padding:.5rem 0 1rem">'
        '<div style="font:700 1rem/1 Inter,sans-serif;color:#e2e8f0">RF-SCA Lab</div>'
        '<div style="font:.72rem/1 Inter,sans-serif;color:#64748b;margin-top:3px">Side-Channel Analysis Pipeline</div>'
        '</div>', unsafe_allow_html=True,
    )
    st.markdown("---")

    s1, s2, s3 = ss.data_loaded, ss.aligned_traces is not None, ss.recovered_key is not None
    steps = [
        ("Generate Traces", s1,  not s1),
        ("Process & Align", s2,  s1 and not s2),
        ("CPA Attack",      s3,  s2 and not s3),
        ("Report",          False, s3),
    ]
    st.markdown(
        '<div style="font:.68rem/1 Inter,sans-serif;font-weight:700;color:#64748b;'
        'text-transform:uppercase;letter-spacing:.08em;margin-bottom:.65rem">Pipeline</div>',
        unsafe_allow_html=True,
    )
    for label, done, active in steps:
        dc = "done" if done else ("active" if active else "")
        lc = ("color:#10b981" if done else ("color:#00d4ff;font-weight:500" if active else "color:#64748b"))
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;padding:.35rem 0">'
            f'<div class="pipe-dot {dc}"></div>'
            f'<span style="font:.8rem/1 Inter,sans-serif;{lc}">{label}</span></div>',
            unsafe_allow_html=True,
        )

    if ss.traces is not None:
        st.markdown("---")
        t = ss.traces
        st.markdown(
            f'<div style="background:#070b14;border:1px solid rgba(0,212,255,.1);'
            f'border-radius:10px;padding:.85rem;font:.75rem/1.9 JetBrains Mono,monospace;color:#94a3b8">'
            f'<span style="color:#64748b">Traces&nbsp;&nbsp;</span><span style="color:#00d4ff">{t.shape[0]:,}</span><br>'
            f'<span style="color:#64748b">Samples </span><span style="color:#00d4ff">{t.shape[1]:,}</span><br>'
            f'<span style="color:#64748b">Source&nbsp;&nbsp;</span><span style="color:#00d4ff">{ss.data_source or "—"}</span>'
            f'</div>', unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<div style="font:.68rem/1.8 JetBrains Mono,monospace;color:#334155;text-align:center">'
        'AES-128 · CPA · Synthetic<br>Educational / Research<br>'
        '<span style="color:#1e3a5f">github.com/HardInCode</span></div>',
        unsafe_allow_html=True,
    )

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero">'
    '<div class="hero-title">📡 RF Side-Channel Analysis Pipeline</div>'
    '<div class="hero-sub">Simulate a complete CPA attack on AES-128 — from synthetic EM trace generation to full 16-byte key recovery.</div>'
    '<span class="badge">🔐 AES-128</span>'
    '<span class="badge">⚡ CPA Attack</span>'
    '<span class="badge v">🎓 Educational</span>'
    '<span class="badge v">🔬 Research</span>'
    '</div>', unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📶  Generate Traces", "⚙️  Process & Align", "🎯  CPA Attack", "📖  Documentation",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Generate
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    sec("📶", "Step 1 — Generate Synthetic Traces", "Configure leakage model and signal parameters")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown('<div class="pcard"><div class="pcard-title">Signal Parameters</div>', unsafe_allow_html=True)
        num_traces = st.number_input("# of Traces",        min_value=100,     max_value=20000,     value=5000,      step=100)
        samples    = st.number_input("Samples per Trace",  min_value=1000,    max_value=10000,     value=5000,      step=100)
        fs         = st.number_input("Sampling Freq (Hz)", min_value=100_000, max_value=10_000_000, value=1_000_000, step=100_000)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="pcard"><div class="pcard-title">Attack Model</div>', unsafe_allow_html=True)
        fc     = st.number_input("Carrier Freq (Hz)", min_value=10_000, max_value=500_000, value=100_000, step=10_000)
        snr    = st.number_input("SNR (dB)",          min_value=0,      max_value=100,     value=60,      step=5)
        model  = st.selectbox("Leakage Model", ["Hamming Weight (Classic)", "Hamming Distance (Enhanced)"])
        st.markdown('</div>', unsafe_allow_html=True)

    enhanced = model.startswith("Hamming Distance")

    if st.button("⚡  Generate Synthetic Traces"):
        with st.spinner("Generating traces…"):
            traces, plaintexts, true_key = generate_synthetic_traces(
                num_traces=num_traces, samples_per_trace=samples,
                fs=fs, fc=fc, snr=snr, enhanced_model=enhanced,
            )
        ss.update(dict(
            traces=traces, plaintexts=plaintexts, true_key=true_key,
            fs=fs, fc=fc, data_loaded=True, data_source="synthetic", processing_done=True,
        ))

        st.markdown("**True Key**")
        key_arr = [true_key] if isinstance(true_key, (int, np.integer)) else true_key
        st.markdown(key_html(key_arr), unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Traces",  f"{num_traces:,}")
        m2.metric("Samples", f"{samples:,}")
        m3.metric("SNR",     f"{snr} dB")
        st.success(f"✅  {num_traces:,} traces generated.")

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(traces[0], color="#00d4ff", lw=0.8, alpha=0.9)
        hw_info = f"  (HW = {bin(plaintexts[0][0] ^ int(true_key)).count('1')})" if not enhanced else ""
        ax.set_title(f"Example Trace — {'Hamming Distance' if enhanced else 'Hamming Weight'}{hw_info}")
        ax.set(xlabel="Sample Index", ylabel="Amplitude")
        plot_show(fig)

    # WAV export
    st.markdown("---")
    st.markdown('<div class="pcard-title">Export to WAV</div>', unsafe_allow_html=True)
    if ss.traces is not None and ss.data_source == "synthetic":
        if st.button("📥  Create WAV for Download"):
            with st.spinner("Exporting…"), tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                export_to_wav(ss.traces, ss.fs, tmp.name)
                data = open(tmp.name, "rb").read()
                os.remove(tmp.name)
            b64  = base64.b64encode(data).decode()
            href = (
                f'<a href="data:audio/wav;base64,{b64}" download="rf_traces.wav"'
                f' style="display:inline-block;padding:8px 16px;background:rgba(0,212,255,.08);'
                f'border:1px solid rgba(0,212,255,.25);border-radius:8px;color:#00d4ff;'
                f'font:.83rem/1 Inter,sans-serif;font-weight:500;text-decoration:none">⬇️&nbsp; Download rf_traces.wav</a>'
            )
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.info("Generate traces first to enable WAV export.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Process & Align
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    sec("⚙️", "Step 2 — Process & Align Traces", "Reduce jitter and improve CPA success rate")

    if not ss.data_loaded:
        st.warning("⚠️  Generate traces in Step 1 first.")
    else:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown('<div class="pcard"><div class="pcard-title">Alignment Method</div>', unsafe_allow_html=True)
            method = st.selectbox("Method", ["Cross-Correlation", "Sum of Absolute Differences", "Maximum Peak"])
            ref    = st.selectbox("Reference", ["First Trace", "Average of All Traces", "Trace with Highest SNR"])
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="pcard"><div class="pcard-title">Window</div>', unsafe_allow_html=True)
            use_win = st.checkbox("Enable window", value=True)
            if use_win:
                wc1, wc2 = st.columns(2)
                w_start = wc1.number_input("Start (%)", 0, 100, 25, 5)
                w_end   = wc2.number_input("End (%)",   0, 100, 75, 5)
            else:
                w_start, w_end = 0, 100
            st.markdown('</div>', unsafe_allow_html=True)

        n = ss.traces.shape[1]
        window = (int(n * w_start / 100), int(n * w_end / 100)) if use_win else None

        if st.button("⚙️  Align Traces"):
            with st.spinner("Aligning…"):
                aligned = align_traces(
                    ss.traces,
                    method=method.lower().replace(" ", "_"),
                    reference=ref.lower().replace(" ", "_"),
                    window=window,
                )
            ss.aligned_traces = aligned
            plot_show(visualize_alignment(ss.traces, aligned, num_to_show=5))

            if st.checkbox("Use aligned traces for attack", value=True):
                ss.traces = aligned
                st.success("Aligned traces set for attack.")
            else:
                st.info("Original traces retained.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CPA Attack
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    sec("🎯", "Step 3 — Correlation Power Analysis", "Recover AES key bytes via statistical correlation")

    data_ready = ss.processing_done and ss.traces is not None and ss.plaintexts is not None
    if not data_ready:
        st.warning("⚠️  Complete Step 1 first.")
        with st.expander("Debug"):
            st.json({"processing_done": ss.processing_done,
                     "traces": ss.traces is not None,
                     "plaintexts": ss.plaintexts is not None})
    else:
        mode = st.radio("Attack Mode", ["Single Byte", "Full Key (16 Bytes)"], horizontal=True)
        ss.full_key_recovery = mode.startswith("Full")
        st.markdown("---")

        # ── Full Key ──────────────────────────────────────────────────────────
        if ss.full_key_recovery:
            attack_traces = align_max_peak(ss.traces)
            if ss.aligned_traces is not None and st.checkbox("Use pre-aligned traces", value=True):
                attack_traces = ss.aligned_traces
                st.info("Using pre-aligned traces.")

            if st.button("🚀  Run Full Key Recovery"):
                prog = st.progress(0)
                status = st.empty()
                corrs, mcorrs, rkey = [], [], np.zeros(16, dtype=np.uint8)
                t0 = time.time()

                with st.spinner("Running CPA on all 16 bytes…"):
                    for i in range(16):
                        status.markdown(
                            f'<div class="chip">Byte <b>{i+1}/16</b></div>', unsafe_allow_html=True
                        )
                        cor, mcor, best = run_cpa(attack_traces, ss.plaintexts, i, vectorized=False)
                        corrs.append(cor); mcorrs.append(mcor); rkey[i] = best
                        prog.progress((i + 1) / 16)

                status.empty(); prog.progress(100)
                ss.recovered_key = rkey
                elapsed = time.time() - t0

                true_key = ss.get("true_key")
                match = (
                    all(rkey[i] == true_key[i] for i in range(16))
                    if true_key is not None and not isinstance(true_key, (int, np.integer)) else None
                )

                cls = "ok" if match else ("bad" if match is False else "")
                st.markdown(f'<div class="res-card {cls}">', unsafe_allow_html=True)
                st.markdown("**Recovered Key**")
                st.markdown(key_html(rkey, true_key), unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Time",      f"{elapsed:.2f}s")
                m2.metric("Recovered", "16 / 16")
                if match is not None:
                    m3.metric("Result", "✅ Correct" if match else "❌ Mismatch")
                st.markdown('</div>', unsafe_allow_html=True)

                if match:    st.success("✅  Full 16-byte key recovered!")
                elif match is False: st.error("❌  Key mismatch — try more traces.")

                plot_show(compare_keys(rkey, true_key=true_key))
                st.markdown("---")
                plot_show(plot_full_key_recovery(corrs, mcorrs, rkey, true_key=true_key))

        # ── Single Byte ───────────────────────────────────────────────────────
        else:
            c1, _ = st.columns([1, 2])
            with c1:
                byte_idx  = st.number_input("Target Byte (0–15)", 0, 15, 0, 1)
                vec       = st.checkbox("Vectorized", value=True)

            if st.button("🎯  Run CPA Attack"):
                with st.spinner("Running CPA…"):
                    aligned = align_max_peak(ss.traces)
                    t0 = time.time()
                    cor, mcor, best = run_cpa(aligned, ss.plaintexts, byte_idx, vectorized=vec)
                    elapsed = time.time() - t0

                ss.update(dict(correlations=cor, max_correlations=mcor, best_key=best))

                true_byte = scalar_key(ss.get("true_key"), byte_idx)
                correct   = true_byte is not None and true_byte == best
                cls = "ok" if correct else ("bad" if true_byte is not None else "")

                st.markdown(f'<div class="res-card {cls}">', unsafe_allow_html=True)
                if true_byte is not None:
                    if correct: st.success(f"✅  `0x{best:02X}` — matches true key!")
                    else:       st.error(f"❌  Got `0x{best:02X}`, true key is `0x{true_byte:02X}`")
                else:
                    st.info(f"Most likely key byte: `0x{best:02X}`")
                m1, m2, m3 = st.columns(3)
                m1.metric("Best Guess",      f"0x{best:02X}")
                m2.metric("Max Correlation", f"{mcor[best]:.4f}")
                m3.metric("Time",            f"{elapsed*1000:.1f} ms")
                st.markdown('</div>', unsafe_allow_html=True)

                plot_show(plot_cpa_results(cor, mcor, best, true_byte))

                with st.expander("🔬 Diagnostics"):
                    top10 = np.argsort(mcor)[::-1][:10]
                    rows  = ""
                    for rank, k in enumerate(top10):
                        c = "#00d4ff" if k == best else ("#10b981" if k == true_byte else "#64748b")
                        tag = " ← recovered" if k == best else (" ← TRUE" if k == true_byte else "")
                        rows += (
                            f'<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,.04)">'
                            f'<span style="color:#334155">#{rank+1:02d}</span> '
                            f'<span style="color:{c}">0x{k:02X}</span> '
                            f'<span style="color:#475569">corr={mcor[k]:.5f}</span>'
                            f'<span style="color:{c}">{tag}</span></div>'
                        )
                    st.markdown(
                        f'<div style="font:.8rem/1 JetBrains Mono,monospace">{rows}</div>',
                        unsafe_allow_html=True,
                    )
                    if true_byte is not None:
                        rank = int(np.where(np.argsort(mcor)[::-1] == true_byte)[0][0]) + 1
                        st.markdown(
                            f'<div style="margin-top:.65rem;font:.82rem/1 Inter,sans-serif;color:#94a3b8">'
                            f'True key <code style="color:#10b981">0x{true_byte:02X}</code> '
                            f'ranked <strong style="color:#e2e8f0">#{rank}</strong> / 256</div>',
                            unsafe_allow_html=True,
                        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Documentation
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    sec("📖", "Documentation & Theory", "Concepts, pipeline reference, and citations")

    d1, d2 = st.columns([3, 2], gap="large")
    with d1:
        st.markdown("""
## What Is a Side-Channel Attack?

Every time a microcontroller runs AES, it draws current in patterns that subtly **correlate
with the intermediate values** being computed. An attacker can capture those emanations and
recover the secret key without ever touching the software.

**CPA** (Correlation Power Analysis) tests all 256 key byte candidates by computing Pearson
correlation between a hypothetical power model and the measured traces. The highest peak wins.

---

## Pipeline Reference

### 1 · Generate Traces
- **Hamming Weight** — leakage ∝ bit-count in `SBox(plaintext ⊕ key)`
- **Hamming Distance** — leakage at multiple AES operation points (SubBytes, MixColumns, etc.)

### 2 · Align Traces

| Method | Best For |
|---|---|
| Cross-Correlation | General-purpose, robust to amplitude variation |
| Sum of Absolute Differences | Fast, low-noise traces |
| Maximum Peak | Trigger-style, amplitude peak alignment |

### 3 · CPA Attack
- **Single Byte** — detailed correlation plots + diagnostics
- **Full Key** — all 16 bytes with progress tracking

The per-sample implementation returns a correlation heat-map showing *where* in the trace
leakage is most exploitable.
        """)

    with d2:
        st.markdown("""
## References

> Mangard, Oswald & Popp  
> *Power Analysis Attacks: Revealing the Secrets of Smartcards*  
> Springer, 2007

---

> Guilley, Danger & Quisquater  
> *Electromagnetic Side-Channel Analysis*  
> Springer, 2015

---

> Collins et al.  
> *Software Defined Radio for Engineers*  
> Artech House, 2018
        """)
        st.markdown("---")
        st.markdown(
            '<div style="display:flex;flex-wrap:wrap;gap:5px;margin-top:.5rem">'
            + "".join(f'<div class="chip"><b>{v}</b></div>' for v in
                      ["AES-128","128-bit key","16 bytes","CPA","HW model","HD model"])
            + "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.caption(
            "Traces are synthetically generated — not captured from real hardware. "
            "Built for educational / research purposes only."
        )

    if ss.traces is not None:
        st.markdown("---")
        st.markdown("**Live Trace Preview**")
        fig, axes = plt.subplots(1, 2, figsize=(12, 3))
        axes[0].plot(ss.traces[0], color="#00d4ff", lw=0.75, alpha=.9, label="Trace 0")
        axes[0].plot(ss.traces[1], color="#7c3aed", lw=0.75, alpha=.7, label="Trace 1")
        axes[0].set_title("Sample Traces (first 2)"); axes[0].legend(fontsize=8, framealpha=.2)
        mean = ss.traces[:min(50, len(ss.traces))].mean(axis=0)
        axes[1].plot(mean, color="#10b981", lw=1.0)
        axes[1].set_title(f"Mean Trace (first {min(50, len(ss.traces))} traces)")
        for ax in axes: ax.set(xlabel="Sample Index", ylabel="Amplitude")
        plot_show(fig)
    else:
        st.info("ℹ️  Generate data in Step 1 to see a live trace preview here.")