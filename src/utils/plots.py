"""Plotting helpers — topographic scalp maps and time-of-night charts.

Uses MNE's plot_topomap for proper 10-20-system scalp visualization, and
matplotlib for time-series plotting.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_topomap(
    channel_names: list[str],
    values: list[float],
    title: str = "",
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] = (6, 5),
):
    """Render a topographic scalp map.

    Falls back to a 2-D scatter plot if MNE can't map all channels to the
    standard 10-20 montage (e.g., unusual or non-EEG channels in the list).

    Parameters
    ----------
    channel_names : list[str]
        Standard 10-20 names (Fp1, F3, Cz, Pz, etc.). Unknown names are skipped.
    values : list[float]
        Same length as channel_names. One value per channel (e.g., median kurtosis).
    title : str
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float, optional
        Color scale limits. If None, inferred from values.
    """
    import mne
    mne.set_log_level("ERROR")

    # Filter to channels MNE can map to the standard 10-20 montage
    montage = mne.channels.make_standard_montage("standard_1020")
    montage_ch = set(c.upper() for c in montage.ch_names)

    keep_names, keep_values = [], []
    for n, v in zip(channel_names, values):
        if n.upper() in montage_ch:
            keep_names.append(n)
            keep_values.append(v)

    if len(keep_names) < 3:
        # Not enough mappable channels for a topomap — fall back to bar chart
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(channel_names, values, color="#5A8DEE")
        ax.set_title(title + " (topomap unavailable — too few standard channels)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    info = mne.create_info(ch_names=keep_names, sfreq=200.0, ch_types="eeg")
    info.set_montage(montage, match_case=False, on_missing="ignore")

    fig, ax = plt.subplots(figsize=figsize)
    arr = np.array(keep_values, dtype=float)
    if vmin is None:
        vmin = arr.min()
    if vmax is None:
        vmax = arr.max()

    im, cn = mne.viz.plot_topomap(
        arr, info, axes=ax, show=False, contours=4,
        cmap=cmap, vlim=(vmin, vmax),
    )
    ax.set_title(title)

    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    return fig


def plot_eeg_trace(
    data: np.ndarray,
    channel_names: list[str],
    sfreq: float,
    title: str = "",
    duration_s: float = 10.0,
    figsize: tuple[float, float] = (10, 8),
    amplitude_uv: float = 100.0,
):
    """Render a clinical-style multi-channel EEG trace.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_channels, n_samples). Each row is one EEG channel.
    channel_names : list[str]
        Same length as data.shape[0].
    sfreq : float
        Sampling rate in Hz.
    title : str
        Plot title.
    duration_s : float
        Display duration; uses only the first duration_s of data.
    amplitude_uv : float
        Visual amplitude scale per channel — channels are stacked vertically
        with this nominal separation. The data is scaled to fit.
    """
    n_chan, n_samples = data.shape
    n_show = min(n_samples, int(duration_s * sfreq))
    seg = data[:, :n_show].astype(np.float64)
    times = np.arange(n_show) / sfreq

    # Normalize each channel: subtract mean, scale to ±amplitude_uv/2
    # using each channel's own peak-to-peak (so visually all channels fit)
    seg_norm = np.zeros_like(seg)
    for j in range(n_chan):
        s = seg[j] - np.mean(seg[j])
        scale = np.percentile(np.abs(s), 95) * 2 or 1.0  # avoid div-by-zero
        seg_norm[j] = s / scale * (amplitude_uv * 0.4)

    fig, ax = plt.subplots(figsize=figsize)
    offsets = np.arange(n_chan) * amplitude_uv
    for j in range(n_chan):
        ax.plot(times, seg_norm[j] + offsets[-(j + 1)], color="black", linewidth=0.6)
    ax.set_yticks(offsets)
    ax.set_yticklabels(channel_names[::-1], fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_xlim(0, n_show / sfreq)
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle=":", alpha=0.3)
    ax.set_facecolor("#FAFAFA")
    plt.tight_layout()
    return fig


def plot_longitudinal_trend(
    dates: list[str],
    values: list[float],
    title: str = "",
    ylabel: str = "",
    normative_range: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (10, 4),
):
    """Plot a single metric over time with optional normative band.

    Parameters
    ----------
    dates : list[str]
        Date strings on the x-axis (sorted).
    values : list[float]
        One value per date.
    normative_range : (low, high), optional
        If provided, draws a shaded green band for the age-typical range.
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = list(range(len(dates)))
    ax.plot(x, values, marker="o", color="#2E4A6B", linewidth=2)

    if normative_range is not None:
        low, high = normative_range
        ax.axhspan(low, high, color="#9CC58E", alpha=0.25,
                   label=f"age-typical {low}–{high}")
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_eeg_trace_with_events(
    data: np.ndarray,
    channel_names: list[str],
    sfreq: float,
    window_start_s: float,
    duration_s: float = 10.0,
    events: list[dict] | None = None,
    title: str = "",
    figsize: tuple[float, float] = (12, 9),
    highlight_channel: str | None = None,
):
    """Render a clinical-style multi-channel EEG trace with event overlays.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
        Raw EEG data (full recording).
    channel_names : list[str]
        Channel names in same order as `data` rows.
    sfreq : float
        Sampling rate (Hz).
    window_start_s : float
        Start time of the window to display (seconds from recording start).
    duration_s : float
        Display duration in seconds.
    events : list of dicts, optional
        Each dict has 'start_s', 'duration_s', optional 'label', optional
        'color' (default red), optional 'channel' (highlight that row).
    highlight_channel : str, optional
        Render this channel's row in red (e.g. the primary detection channel).
    """
    # `data` is the segment to display. `window_start_s` is the time label
    # for the x-axis (so events with absolute timestamps overlay correctly).
    # `duration_s` is the intended display length but actual display matches
    # data.shape[1].
    if data.size == 0 or data.shape[0] == 0 or data.shape[1] == 0:
        # Defensive empty-data fallback: render a friendly placeholder
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5,
                "EEG trace not available for this window\n"
                "(window outside recording or empty data)",
                ha="center", va="center", fontsize=11,
                color="#888", transform=ax.transAxes)
        ax.axis("off")
        return fig

    # Truncate to requested duration if longer; otherwise use as-is
    n_show = min(data.shape[1], int(duration_s * sfreq))
    seg = data[:, :n_show].astype(np.float64)
    n_chan = seg.shape[0]
    times = np.arange(seg.shape[1]) / sfreq + window_start_s

    # Per-channel normalization for visual stacking
    amplitude_uv = 100.0
    seg_norm = np.zeros_like(seg)
    for j in range(n_chan):
        s = seg[j] - np.mean(seg[j])
        scale = np.percentile(np.abs(s), 95) * 2 or 1.0
        seg_norm[j] = s / scale * (amplitude_uv * 0.4)

    fig, ax = plt.subplots(figsize=figsize)
    offsets = np.arange(n_chan) * amplitude_uv

    # Event overlays — draw shaded regions FIRST so they sit behind the traces
    if events:
        for ev in events:
            ev_start = float(ev.get("start_s", 0))
            ev_dur = float(ev.get("duration_s", 0))
            ev_end = ev_start + ev_dur
            # Only draw if visible in window
            if ev_end < window_start_s or ev_start > window_start_s + duration_s:
                continue
            color = ev.get("color", "#FFB1B1")
            label = ev.get("label", "")
            ax.axvspan(ev_start, ev_end, color=color, alpha=0.35,
                       label=label or None)

    # Channel traces
    for j in range(n_chan):
        ch_name = channel_names[j]
        is_highlight = (highlight_channel is not None
                        and ch_name.upper() == highlight_channel.upper())
        color = "#A02020" if is_highlight else "black"
        lw = 0.9 if is_highlight else 0.6
        ax.plot(times, seg_norm[j] + offsets[-(j + 1)],
                color=color, linewidth=lw)

    ax.set_yticks(offsets)
    ax.set_yticklabels(channel_names[::-1], fontsize=8)
    ax.set_xlabel("Time (seconds from recording start)")
    ax.set_xlim(window_start_s, window_start_s + duration_s)
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle=":", alpha=0.3)
    ax.set_facecolor("#FAFAFA")

    # Legend if any labeled events
    if events and any(ev.get("label") for ev in events):
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        unique = [(h, l) for h, l in zip(handles, labels)
                  if l and l not in seen and not seen.add(l)]
        if unique:
            ax.legend([h for h, _ in unique], [l for _, l in unique],
                      loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    return fig


def plot_longitudinal_comparison(delta) -> "plt.Figure":
    """Three-panel longitudinal comparison figure.

    Top-left : per-channel spike-rate bar chart (A vs B side by side).
    Top-right: scatter plot (A on x-axis, B on y-axis) with identity line.
    Bottom   : summary text block with key metric deltas.

    Parameters
    ----------
    delta : LongitudinalDelta
        Output of ``compare_recordings``.
    """
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], hspace=0.45, wspace=0.35)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, :])

    label_a = delta.recording_a.get("label") or delta.recording_a.get("date") or "A"
    label_b = delta.recording_b.get("label") or delta.recording_b.get("date") or "B"

    # ── Top-left: per-channel bar chart ──────────────────────────────────────
    channels = sorted(delta.spike_rate_per_channel.keys())
    # Limit to 20 channels for readability
    if len(channels) > 20:
        # Show only channels with any activity
        channels = sorted(
            channels,
            key=lambda c: -(delta.spike_rate_per_channel[c][0]
                            + delta.spike_rate_per_channel[c][1]),
        )[:20]
        channels = sorted(channels)

    rates_a = [delta.spike_rate_per_channel[c][0] for c in channels]
    rates_b = [delta.spike_rate_per_channel[c][1] for c in channels]

    x = np.arange(len(channels))
    w = 0.38
    bar_a = ax_bar.bar(x - w / 2, rates_a, w, label=label_a,
                       color="#5A8DEE", alpha=0.85)
    bar_b = ax_bar.bar(x + w / 2, rates_b, w, label=label_b,
                       color="#EE8D5A", alpha=0.85)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(channels, rotation=60, ha="right", fontsize=7)
    ax_bar.set_ylabel("Spike index (kurtosis or /min)", fontsize=8)
    ax_bar.set_title("Per-channel spike burden", fontsize=9)
    ax_bar.legend(fontsize=8, loc="upper right")
    ax_bar.grid(True, axis="y", linestyle=":", alpha=0.4)

    # ── Top-right: scatter A vs B ─────────────────────────────────────────────
    if channels:
        ax_scatter.scatter(rates_a, rates_b, s=35, color="#444",
                           alpha=0.7, zorder=3)
        for ch, ra, rb in zip(channels, rates_a, rates_b):
            ax_scatter.annotate(ch, (ra, rb), fontsize=6, alpha=0.6,
                                textcoords="offset points", xytext=(3, 2))

        all_vals = rates_a + rates_b
        lo, hi = min(all_vals) * 0.9, max(all_vals) * 1.1 if all_vals else (0, 1)
        if lo == hi:
            hi = lo + 1
        ax_scatter.plot([lo, hi], [lo, hi], linestyle="--", color="#AAA",
                        linewidth=1, label="identity (no change)")

        # Mean change line
        mean_a = sum(rates_a) / len(rates_a) if rates_a else 0
        mean_b = sum(rates_b) / len(rates_b) if rates_b else 0
        ax_scatter.axhline(mean_b, color="#EE8D5A", linewidth=1,
                           linestyle=":", alpha=0.7)
        ax_scatter.axvline(mean_a, color="#5A8DEE", linewidth=1,
                           linestyle=":", alpha=0.7)
        ax_scatter.legend(fontsize=7)

    ax_scatter.set_xlabel(f"{label_a} spike index", fontsize=8)
    ax_scatter.set_ylabel(f"{label_b} spike index", fontsize=8)
    ax_scatter.set_title("Channel-level A vs B", fontsize=9)
    ax_scatter.grid(True, linestyle=":", alpha=0.3)

    # ── Bottom: summary text ──────────────────────────────────────────────────
    ax_text.axis("off")
    delta_str = f"{delta.mean_spike_rate_delta_pct:+.1f}%"
    pdr_str = (f"{delta.pdr_delta_hz:+.1f} Hz" if delta.pdr_delta_hz is not None
               else "n/a")
    spindle_str = (f"{delta.spindle_delta_pct:+.1f}%" if delta.spindle_delta_pct
                   is not None else "not compared (sleep <2h in one recording)")
    shift_str = delta.topographic_shift.replace("_", " ")
    compat_str = "yes" if delta.duration_compatible else "NO (see methodology warning)"

    lines = [
        f"Mean spike-rate delta: {delta_str}   |   "
        f"Topographic shift: {shift_str}   |   "
        f"Duration compatible: {compat_str}",
        f"PDR delta: {pdr_str}   |   "
        f"Spindles: {spindle_str}   |   "
        f"Age delta: {delta.age_delta_years:.2f} yr",
    ]
    if delta.confounds:
        lines.append(f"Confounds: {len(delta.confounds)} detected — see report for details")
    ax_text.text(0.01, 0.85, "\n".join(lines), transform=ax_text.transAxes,
                 fontsize=8.5, verticalalignment="top",
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5",
                           edgecolor="#CCC"))

    date_a = delta.recording_a.get("date", "")
    date_b = delta.recording_b.get("date", "")
    fig.suptitle(
        f"Longitudinal comparison: {label_a} ({date_a}) → {label_b} ({date_b})",
        fontsize=11, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_metric_timeline(
    entries: list[dict],
    metric: str,
    interventions: list[dict] | None = None,
    title: str = "",
    ylabel: str = "",
    figsize: tuple[float, float] = (11, 4),
) -> "plt.Figure":
    """Plot a single metric over multiple recordings, with intervention markers.

    Parameters
    ----------
    entries : list of dicts
        Each dict must have 'date' (YYYY-MM-DD) and the metric key, e.g.:
        [{"date": "2024-08-01", "mean_spike_rate": 12.3}, ...]
        Missing values are skipped.
    metric : str
        Key to extract from each entry dict.
    interventions : list of dicts, optional
        Each dict: {"date": "YYYY-MM-DD", "label": "Supplements started",
                    "color": "#E07"}   (color optional, defaults cycle)
    title, ylabel : str
    """
    import datetime as _dt

    _INTERVENTION_COLORS = ["#D44", "#4A4", "#44D", "#D94", "#94D", "#D49"]

    points = []
    for e in entries:
        val = e.get(metric)
        date_s = e.get("date") or e.get("recording_date")
        if val is not None and date_s:
            try:
                d = _dt.date.fromisoformat(date_s)
                points.append((d, float(val)))
            except (ValueError, TypeError):
                pass

    fig, ax = plt.subplots(figsize=figsize)

    if points:
        points.sort(key=lambda p: p[0])
        dates = [p[0] for p in points]
        values = [p[1] for p in points]

        import matplotlib.dates as mdates
        ax.plot_date(
            [mdates.date2num(d) for d in dates],
            values,
            "-o",
            color="#2E4A6B",
            linewidth=2,
            markersize=6,
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate(rotation=35, ha="right")

        # Intervention markers
        if interventions:
            for i, interv in enumerate(interventions):
                idate_s = interv.get("date")
                if not idate_s:
                    continue
                try:
                    idate = _dt.date.fromisoformat(idate_s)
                    icolor = interv.get("color") or _INTERVENTION_COLORS[
                        i % len(_INTERVENTION_COLORS)
                    ]
                    ilabel = interv.get("label") or f"Intervention {i+1}"
                    ax.axvline(
                        mdates.date2num(idate),
                        color=icolor,
                        linestyle="--",
                        linewidth=1.5,
                        label=ilabel,
                        alpha=0.85,
                    )
                except (ValueError, TypeError):
                    pass
            ax.legend(fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, f"No data for metric '{metric}'",
                ha="center", va="center", transform=ax.transAxes, color="#888")

    ax.set_ylabel(ylabel or metric, fontsize=9)
    ax.set_title(title or f"Timeline: {metric}", fontsize=10)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_time_of_night(
    bin_centers_hours: list[float],
    counts_per_min: list[float],
    title: str = "Spike burden across the night",
    xlabel: str = "Hours after sleep onset",
    ylabel: str = "Spikes / minute",
    figsize: tuple[float, float] = (10, 4),
    highlight_peak: bool = True,
):
    """Plot a binned time-of-night chart of spike burden."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(bin_centers_hours, counts_per_min, width=0.45,
           color="#5A8DEE", edgecolor="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    if highlight_peak and counts_per_min:
        peak_idx = int(np.argmax(counts_per_min))
        peak_x = bin_centers_hours[peak_idx]
        peak_y = counts_per_min[peak_idx]
        ax.annotate(
            f"peak: {peak_y:.0f}/min",
            xy=(peak_x, peak_y),
            xytext=(peak_x, peak_y * 1.1),
            fontsize=9, color="#A05500",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#A05500", lw=0.8),
        )

    plt.tight_layout()
    return fig
