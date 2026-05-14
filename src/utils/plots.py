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
