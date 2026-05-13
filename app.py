"""KCNQ3-Lens Streamlit App.

Run with:
    streamlit run app.py

All processing happens locally. EEG files never leave your machine.
If you use AI interpretation, only derived numerical metrics are sent
to your chosen provider with your own API key.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.readers import load_eeg
from src.runner import run_all_analyses
from src.comparison import compare_findings
from src.ai import (
    interpret_findings,
    interpret_comparison,
    list_providers,
    get_provider_class,
)
from src.reports import build_doctor_pdf, build_parent_pdf
from src.utils.plots import plot_topomap, plot_time_of_night
from src.i18n import get_translator, LANGUAGES


# ─── Page setup ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KCNQ3-Lens",
    page_icon="🧠",
    layout="wide",
)

# Language preference persists in session
if "language" not in st.session_state:
    st.session_state.language = "en"

# Top-level: language selector (small, at top of sidebar)
lang = st.sidebar.selectbox(
    "🌐 " + LANGUAGES[st.session_state.language],
    options=list(LANGUAGES.keys()),
    format_func=lambda k: LANGUAGES[k],
    index=list(LANGUAGES.keys()).index(st.session_state.language),
    key="language_select",
    label_visibility="collapsed",
)
if lang != st.session_state.language:
    st.session_state.language = lang
    st.rerun()

T = get_translator(st.session_state.language).t


# ─── Header ──────────────────────────────────────────────────────────────────
st.title(T("app_title"))
st.caption(T("app_subtitle"))

with st.expander(T("disclaimer_header"), expanded=False):
    st.markdown(T("disclaimer_body"))


# ─── Sidebar: mode selector ─────────────────────────────────────────────────
st.sidebar.header(T("sidebar_mode"))
mode = st.sidebar.radio(
    label="mode",
    options=["single", "compare"],
    format_func=lambda x: T("mode_single") if x == "single" else T("mode_compare"),
    label_visibility="collapsed",
)


# ─── Sidebar: recording settings ────────────────────────────────────────────
st.sidebar.header(T("sidebar_recording_settings"))

age_years = st.sidebar.number_input(
    T("sidebar_age"),
    min_value=0.0, max_value=21.0, value=5.0, step=0.5,
    help=T("sidebar_age_help"),
)

variant = st.sidebar.text_input(
    T("sidebar_variant"),
    value="",
    placeholder=T("sidebar_variant_placeholder"),
    help=T("sidebar_variant_help"),
)

st.sidebar.header(T("sidebar_windows"))
st.sidebar.caption(T("sidebar_windows_caption"))

wake_start_s = st.sidebar.number_input(T("wake_start"), value=0)
wake_end_s = st.sidebar.number_input(T("wake_end"), value=3600)
sleep_start_s = st.sidebar.number_input(T("sleep_start"), value=25200)
sleep_end_s = st.sidebar.number_input(T("sleep_end"), value=54000)


# ─── Sidebar: AI provider settings ──────────────────────────────────────────
st.sidebar.header(T("sidebar_ai_header"))
st.sidebar.caption(T("sidebar_ai_caption"))

_providers = list_providers()
_provider_labels = {p.id: p.display_name for p in _providers}
_provider_by_id = {p.id: p for p in _providers}

provider_id = st.sidebar.selectbox(
    T("ai_provider"),
    options=list(_provider_labels.keys()),
    format_func=lambda x: _provider_labels[x],
    index=0,
)
_selected_info = _provider_by_id[provider_id]
_provider_cls = get_provider_class(provider_id)
_sdk_available = _provider_cls.is_available()

if not _sdk_available:
    st.sidebar.warning(T("ai_sdk_missing", package=_selected_info.pip_package))

api_key = st.sidebar.text_input(
    T("ai_api_key", provider=_selected_info.display_name),
    type="password",
    help=f"Get your key at: {_selected_info.api_key_url}",
)

ai_model = st.sidebar.selectbox(
    T("ai_model"),
    options=_selected_info.available_models,
    index=0,
)

st.sidebar.caption(
    f"[{T('ai_key_link', provider=_selected_info.display_name)}]"
    f"({_selected_info.api_key_url})"
)


# ─── Helpers: file loading and rendering ────────────────────────────────────
def _load_file(uploaded_file, local_path: str, slot_key: str):
    """Resolve a file source from the upload widget or text input, return path."""
    source_path = None
    if uploaded_file is not None:
        tmpdir = Path(tempfile.gettempdir()) / "kcnq3-lens-uploads"
        tmpdir.mkdir(exist_ok=True)
        source_path = tmpdir / f"{slot_key}_{uploaded_file.name}"
        source_path.write_bytes(uploaded_file.getvalue())
    elif local_path:
        source_path = Path(local_path).expanduser()
    return source_path


def _show_file_metrics(rec):
    """Show metric tiles for a loaded recording."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(T("metric_format"), rec.format_name)
    col2.metric(T("metric_sfreq"), f"{rec.sfreq:.0f} Hz")
    col3.metric(T("metric_duration"), f"{rec.duration_s / 3600:.2f} h")
    col4.metric(T("metric_eeg_channels"), str(rec.n_channels))
    with st.expander(T("channel_layout")):
        st.write(", ".join(rec.channel_names))


def _file_uploader_section(slot_key: str, header_key: str):
    """Render a file-upload section (one EEG file) and return loaded recording."""
    st.header(T(header_key))
    uploaded = st.file_uploader(
        T("file_picker"),
        type=["eeg", "edf", "bdf", "vhdr", "set"],
        help=T("file_picker_help"),
        key=f"upload_{slot_key}",
    )
    local_path = st.text_input(
        T("local_path"),
        value="",
        help=T("local_path_help"),
        key=f"path_{slot_key}",
    )
    source_path = _load_file(uploaded, local_path, slot_key)
    if source_path is None:
        return None
    try:
        with st.spinner(T("reading", filename=source_path.name)):
            rec = load_eeg(source_path)
        _show_file_metrics(rec)
        return rec
    except Exception as e:
        st.error(T("load_error", error=str(e)))
        return None


def _resolve_windows(rec):
    """Convert sidebar time values into epoch indices for this recording."""
    wake_start_ep = int(wake_start_s / 30)
    wake_end_ep = min(int(wake_end_s / 30), rec.n_epochs)
    sleep_start_ep = int(sleep_start_s / 30)
    sleep_end_ep = min(int(sleep_end_s / 30), rec.n_epochs)
    return wake_start_ep, wake_end_ep, sleep_start_ep, sleep_end_ep


def _render_findings_tabs(findings: dict, key_prefix: str = ""):
    """Render per-analysis tabs for one findings dict."""
    tab_topo, tab_spindle, tab_bg, tab_burst, tab_morph, tab_ton, tab_raw = st.tabs([
        T("tab_topography"), T("tab_spindles"), T("tab_background"),
        T("tab_bursts"), T("tab_morphology"), T("tab_time_of_night"), T("tab_raw"),
    ])

    with tab_topo:
        t = findings.get("topography", {})
        if t:
            st.subheader(T("topo_header"))
            st.caption(T("topo_caption"))

            # New: topographic scalp map
            try:
                channel_names = [c["name"] for c in t["all_channels"]]
                channel_values = [c["median"] for c in t["all_channels"]]
                st.markdown(f"**{T('topomap_title')}**")
                st.caption(T("topomap_caption"))
                fig_topo = plot_topomap(
                    channel_names, channel_values,
                    title="",
                )
                st.pyplot(fig_topo)
            except Exception as e:
                st.warning(f"Topographic map unavailable: {e}")

            # Existing bar chart (kept as quick numerical reference)
            df = pd.DataFrame(t["all_channels"])
            df = df.sort_values("median", ascending=False).reset_index(drop=True)
            st.dataframe(df, use_container_width=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(df["name"], df["median"], color="#5A8DEE")
            ax.set_ylabel("Median kurtosis")
            ax.set_xlabel("Channel")
            ax.axhline(y=5, color="gray", linestyle="--", linewidth=0.8,
                       label=T("topo_normal_upper"))
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

    with tab_spindle:
        s = findings.get("spindles", {})
        if s:
            st.subheader(T("spindle_header", channel=s["channel"]))
            col1, col2, col3 = st.columns(3)
            col1.metric(T("spindle_density"), f"{s['density_per_minute']:.2f}")
            col2.metric(T("spindle_count"), str(s["n_spindles"]))
            col3.metric(T("spindle_freq"), f"{s['median_peak_freq_hz']:.1f}")
            norm = s.get("age_normative_range")
            if norm:
                low, high = norm
                if s["interpretation"] == "below":
                    st.warning(T("spindle_below_norm", low=low, high=high))
                elif s["interpretation"] == "above":
                    st.info(T("spindle_above_norm", low=low, high=high))
                else:
                    st.success(T("spindle_in_norm", low=low, high=high))
            st.caption(T("spindle_caption"))

    with tab_bg:
        b = findings.get("background", {})
        if b:
            st.subheader(T("bg_header"))
            col1, col2 = st.columns(2)
            with col1:
                st.metric(T("bg_pdr"), f"{b['posterior_dominant_rhythm_hz']:.1f} Hz")
                norm = b.get("age_normative_pdr")
                if norm:
                    low, high = norm
                    st.caption(T("bg_pdr_normative", low=low, high=high))
            with col2:
                st.metric(T("bg_dar"), f"{b['delta_alpha_ratio']:.2f}")
                st.caption(T("bg_dar_normative"))

            df_band = pd.DataFrame({
                "Band": ["Delta (1–4 Hz)", "Theta (4–8 Hz)",
                         "Alpha (8–13 Hz)", "Beta (13–30 Hz)"],
                "Power (%)": [b["delta_pct"], b["theta_pct"],
                              b["alpha_pct"], b["beta_pct"]],
            })
            st.bar_chart(df_band.set_index("Band"))

            if b["interpretation"] == "severely_slow":
                st.warning(T("bg_severely_slow"))
            elif b["interpretation"] == "mildly_slow":
                st.info(T("bg_mildly_slow"))
            elif b["interpretation"] == "age_appropriate":
                st.success(T("bg_appropriate"))

    with tab_burst:
        br = findings.get("bursts", {})
        if br:
            st.subheader(T("bursts_header", channel=br["primary_channel"]))
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(T("bursts_total"), str(br["n_bursts"]))
            col2.metric(T("bursts_5s"), str(br["n_bursts_5s_or_longer"]))
            col3.metric(T("bursts_10s"), str(br["n_bursts_10s_or_longer"]))
            col4.metric(T("bursts_max"), f"{br['max_duration_s']:.1f}")
            if br["longest_bursts"]:
                st.write(T("bursts_longest_label"))
                st.dataframe(pd.DataFrame(br["longest_bursts"]), use_container_width=True)
            if br["n_bursts_10s_or_longer"] > 0:
                st.warning(T("bursts_warn_long", count=br["n_bursts_10s_or_longer"]))

    with tab_morph:
        m = findings.get("morphology", {})
        if m:
            st.subheader(T("morph_header", channel=m["channel"]))
            col1, col2, col3 = st.columns(3)
            col1.metric(T("morph_simple"), f"{m['pct_simple_spikes']:.0f}%")
            col2.metric(T("morph_sharp"), f"{m['pct_sharp_waves']:.0f}%")
            col3.metric(T("morph_complex"), f"{m['pct_complex_spike_wave']:.0f}%")
            st.write(T("morph_events", n=m["n_events"], rate=m["events_per_minute"]))
            st.write(T("morph_polyspike", pct=m["polyspike_fraction"]))
            cls = m["classification"]
            if cls == "predominantly_complex":
                st.warning(T("morph_complex_classification"))
            elif cls == "predominantly_simple":
                st.info(T("morph_simple_classification"))
            else:
                st.info(T("morph_mixed"))

    with tab_ton:
        tn = findings.get("time_of_night", {})
        if tn:
            st.subheader(T("ton_header"))
            st.caption(T("ton_caption"))
            col1, col2, col3 = st.columns(3)
            col1.metric("Peak (/min)", f"{tn.get('peak_count_per_min', 0):.1f}")
            col2.metric("Peak at (h)", f"{tn.get('peak_bin_hours', 0):.1f}")
            col3.metric("Total events", str(tn.get("total_events", 0)))

            try:
                fig_ton = plot_time_of_night(
                    bin_centers_hours=tn.get("bin_centers", []),
                    counts_per_min=tn.get("counts_per_min", []),
                    title="",
                    xlabel="Hours from recording start",
                )
                st.pyplot(fig_ton)
            except Exception as e:
                st.warning(f"Plot unavailable: {e}")

    with tab_raw:
        st.subheader(T("raw_header"))
        st.caption(T("raw_caption"))
        json_str = json.dumps(findings, indent=2, default=str)
        st.code(json_str, language="json")
        st.download_button(
            T("raw_download"),
            json_str,
            file_name=f"kcnq3-lens-findings-{key_prefix or 'single'}.json",
            mime="application/json",
            key=f"download_{key_prefix or 'single'}",
        )


def _run_analyses_with_progress(rec, label: str = ""):
    """Run all analyses and update Streamlit progress bar."""
    wake_start_ep, wake_end_ep, sleep_start_ep, sleep_end_ep = _resolve_windows(rec)
    progress = st.progress(0.0)
    status = st.empty()

    step_labels = {
        "topography": T("progress_topography"),
        "spindles": T("progress_spindles"),
        "background": T("progress_background"),
        "bursts": T("progress_bursts"),
        "morphology": T("progress_morphology"),
    }

    def _cb(name: str, frac: float):
        label_text = step_labels.get(name, name)
        if label:
            status.text(f"[{label}] {label_text}")
        else:
            status.text(label_text)
        progress.progress(frac)

    findings = run_all_analyses(
        rec,
        sleep_start_epoch=sleep_start_ep,
        sleep_end_epoch=sleep_end_ep,
        wake_epoch_indices=list(range(wake_start_ep, wake_end_ep)),
        age_years=age_years,
        progress_callback=_cb,
    )

    status.text(T("progress_done"))
    return findings


# ═══════════════════════════════════════════════════════════════════════════
# MODE A: Single recording
# ═══════════════════════════════════════════════════════════════════════════
if mode == "single":
    rec = _file_uploader_section("single", "step1_header")

    st.header(T("step2_header"))
    if rec is None:
        st.info(T("load_first_message"))
    else:
        if st.button(T("run_button"), type="primary"):
            findings = _run_analyses_with_progress(rec)
            for analysis, err in (findings.get("errors") or {}).items():
                st.warning(T("analysis_failed", analysis=analysis, error=err))
            st.session_state["single_findings"] = findings

    findings = st.session_state.get("single_findings", {})
    if findings:
        st.header(T("step3_header"))
        _render_findings_tabs(findings, key_prefix="single")

        # AI interpretation
        st.header(T("step4_header"))
        if not api_key:
            st.info(T("ai_need_key", provider=_selected_info.display_name))
        elif not _sdk_available:
            st.warning(T("ai_sdk_needed", package=_selected_info.pip_package))
        else:
            if st.button(T("ai_generate_button", provider=_selected_info.display_name),
                         key="ai_single"):
                with st.spinner(T("ai_thinking", provider=_selected_info.display_name)):
                    try:
                        result = interpret_findings(
                            provider_id=provider_id,
                            api_key=api_key,
                            findings=findings,
                            age_years=age_years,
                            variant=variant or None,
                            model=ai_model,
                        )
                        st.session_state["single_interp"] = result
                        st.session_state["single_interp_provider"] = _selected_info.display_name
                    except Exception as e:
                        st.error(T("ai_failed", error=str(e)))

        if "single_interp" in st.session_state:
            st.caption(T("ai_generated_by",
                         provider=st.session_state["single_interp_provider"]))
            st.markdown(st.session_state["single_interp"])
            st.download_button(
                T("ai_download_md"),
                st.session_state["single_interp"],
                file_name="kcnq3-lens-interpretation.md",
                mime="text/markdown",
                key="dl_single_interp",
            )

        # PDF reports
        st.header(T("pdf_header"))
        st.caption(T("pdf_caption"))
        col_doc, col_par = st.columns(2)
        with col_doc:
            try:
                pdf_bytes = build_doctor_pdf(
                    findings,
                    age_years=age_years,
                    variant=variant or None,
                )
                st.download_button(
                    T("pdf_doctor_button"),
                    pdf_bytes,
                    file_name="kcnq3-lens-doctor-report.pdf",
                    mime="application/pdf",
                    key="dl_single_pdf_doctor",
                )
            except Exception as e:
                st.warning(f"PDF generation failed: {e}")
        with col_par:
            try:
                pdf_bytes = build_parent_pdf(
                    findings,
                    age_years=age_years,
                    variant=variant or None,
                    interpretation=st.session_state.get("single_interp"),
                )
                st.download_button(
                    T("pdf_parent_button"),
                    pdf_bytes,
                    file_name="kcnq3-lens-parent-report.pdf",
                    mime="application/pdf",
                    key="dl_single_pdf_parent",
                )
            except Exception as e:
                st.warning(f"PDF generation failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# MODE B: Compare two recordings
# ═══════════════════════════════════════════════════════════════════════════
else:
    col_pre, col_post = st.columns(2)
    with col_pre:
        rec_pre = _file_uploader_section("pre", "step1_header_pre")
    with col_post:
        rec_post = _file_uploader_section("post", "step1_header_post")

    st.header(T("step2_header_compare"))
    if rec_pre is None or rec_post is None:
        st.info(T("load_both_message"))
    else:
        if st.button(T("run_button_compare"), type="primary"):
            with st.status(T("running_pre"), expanded=True) as status:
                findings_pre = _run_analyses_with_progress(rec_pre, label="pre")
                status.update(label=T("running_post"))
                findings_post = _run_analyses_with_progress(rec_post, label="post")
                status.update(label=T("progress_done"), state="complete")
            for analysis, err in (findings_pre.get("errors") or {}).items():
                st.warning(f"PRE — " + T("analysis_failed", analysis=analysis, error=err))
            for analysis, err in (findings_post.get("errors") or {}).items():
                st.warning(f"POST — " + T("analysis_failed", analysis=analysis, error=err))
            st.session_state["compare_pre"] = findings_pre
            st.session_state["compare_post"] = findings_post
            st.session_state["compare_result"] = compare_findings(findings_pre, findings_post)

    if "compare_result" in st.session_state:
        comp = st.session_state["compare_result"]
        st.header(T("step3_header_compare"))

        # Overall verdict banner
        verdict = comp["overall"]["verdict"]
        n_imp = comp["overall"]["n_improved"]
        n_wor = comp["overall"]["n_worsened"]
        n_unc = comp["overall"]["n_unchanged"]
        if verdict in ("clearly_improved", "mixed_mostly_improved"):
            st.success(f"✓ {n_imp} {T('compare_improved')} · "
                       f"{n_wor} {T('compare_worsened')} · "
                       f"{n_unc} {T('compare_unchanged')}")
        elif verdict in ("clearly_worsened", "mixed_mostly_worsened"):
            st.warning(f"✗ {n_imp} {T('compare_improved')} · "
                       f"{n_wor} {T('compare_worsened')} · "
                       f"{n_unc} {T('compare_unchanged')}")
        else:
            st.info(f"≈ {n_imp} {T('compare_improved')} · "
                    f"{n_wor} {T('compare_worsened')} · "
                    f"{n_unc} {T('compare_unchanged')}")

        # Delta table
        st.subheader(T("compare_summary_header"))
        delta_rows = []
        for d in comp["deltas"]:
            direction_label = {
                "improved": T("compare_improved"),
                "worsened": T("compare_worsened"),
                "unchanged": T("compare_unchanged"),
            }[d["direction"]]
            pct_str = f"{d['pct_change']:+.1f}%" if d["pct_change"] is not None else "—"
            delta_rows.append({
                "Metric": d["name"],
                T("compare_metric_pre"): d["pre_value"],
                T("compare_metric_post"): d["post_value"],
                T("compare_metric_change"): pct_str,
                "": direction_label,
            })
        st.dataframe(pd.DataFrame(delta_rows), use_container_width=True)

        # Per-recording findings tabs
        tab_pre, tab_post = st.tabs([
            f"📊 PRE  ({T('compare_metric_pre')})",
            f"📊 POST ({T('compare_metric_post')})",
        ])
        with tab_pre:
            _render_findings_tabs(st.session_state["compare_pre"], key_prefix="pre")
        with tab_post:
            _render_findings_tabs(st.session_state["compare_post"], key_prefix="post")

        # AI comparison interpretation
        st.header(T("step4_header"))
        st.caption(T("compare_ai_caption"))
        if not api_key:
            st.info(T("ai_need_key", provider=_selected_info.display_name))
        elif not _sdk_available:
            st.warning(T("ai_sdk_needed", package=_selected_info.pip_package))
        else:
            if st.button(T("compare_ai_button", provider=_selected_info.display_name),
                         key="ai_compare"):
                with st.spinner(T("ai_thinking", provider=_selected_info.display_name)):
                    try:
                        result = interpret_comparison(
                            provider_id=provider_id,
                            api_key=api_key,
                            comparison=comp,
                            age_years=age_years,
                            variant=variant or None,
                            model=ai_model,
                        )
                        st.session_state["compare_interp"] = result
                        st.session_state["compare_interp_provider"] = _selected_info.display_name
                    except Exception as e:
                        st.error(T("ai_failed", error=str(e)))

        if "compare_interp" in st.session_state:
            st.caption(T("ai_generated_by",
                         provider=st.session_state["compare_interp_provider"]))
            st.markdown(st.session_state["compare_interp"])
            st.download_button(
                T("ai_download_md"),
                st.session_state["compare_interp"],
                file_name="kcnq3-lens-comparison.md",
                mime="text/markdown",
                key="dl_compare_interp",
            )


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(T("footer"))
