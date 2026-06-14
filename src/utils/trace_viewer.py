"""A4 — Raw-trace viewer.  [BORROW: mne / matplotlib (already in stack)]

A metrics+PDF UI gives no way to eyeball the actual spike-wave behind a number.
This renders navigable multi-channel trace windows (a clinical-style page: fixed
time window, stacked channels, µV/division sensitivity, optional event overlays)
for the Streamlit UI, and offers a desktop hand-off to mne's interactive Qt
browser.

BORROW: builds on matplotlib (via the object-oriented Figure API, so it works
headless with no backend/pyplot global state) and mne — both already
dependencies. No new dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..readers.base import EEGRecording

# Cap channels rendered in one page for legibility.
_MAX_CHANNELS = 24


@dataclass
class TraceWindow:
    start_s: float
    duration_s: float
    channels: list[str]
    sfreq: float
    n_samples: int
    note: str = ""
    notes: list[str] = field(default_factory=list)


def _resolve_channels(rec: EEGRecording, channels: list[str] | None) -> list[int]:
    if channels is None:
        idx = list(rec.eeg_channel_indices) or list(range(rec.n_channels_in_file))
        return idx[:_MAX_CHANNELS]
    out = []
    for c in channels:
        i = rec.channel_index(c)
        if i is not None:
            out.append(i)
    return out


def _read_window_by_index(
    rec: EEGRecording, start_s: float, duration_s: float, ch_idx: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_seconds, data[len(ch_idx), n_samples]) for a window, reading by
    file-channel INDEX (no name round-trip → duplicate names can't alias).

    Slices the in-memory buffer directly when available (so a sub-30 s recording
    and the trailing partial epoch of an N×30 s+ recording are read exactly);
    otherwise falls back to the lazy 30 s-epoch reader (whose trailing partial
    epoch is dropped — acceptable on the multi-hour recordings that path serves).
    """
    sf = float(rec.sfreq)
    start = max(0, int(round(start_s * sf)))
    n = max(1, int(round(duration_s * sf)))
    end = start + n
    full = rec._full_data
    if full is not None:
        seg = full[:, start:end]
    else:
        spe = int(round(30.0 * sf))
        if spe <= 0:
            seg = np.zeros((rec.n_channels_in_file, 0))
        else:
            ep0, ep1 = start // spe, (end - 1) // spe
            chunks = []
            for ep in range(ep0, ep1 + 1):
                d = rec.read_epoch(ep, 30.0)
                if d is None:
                    break
                chunks.append(np.asarray(d, dtype=float))
            if chunks:
                f = np.concatenate(chunks, axis=1)
                local = start - ep0 * spe
                seg = f[:, local: local + n]
            else:
                seg = np.zeros((rec.n_channels_in_file, 0))
    data = seg[ch_idx] if ch_idx else seg
    t = start_s + np.arange(data.shape[1]) / sf
    return t, np.asarray(data, dtype=float)


def read_trace_window(
    rec: EEGRecording,
    start_s: float = 0.0,
    duration_s: float = 10.0,
    channels: list[str] | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (channel_names, t_seconds, data[n_ch, n_samples]) for a window."""
    ch_idx = _resolve_channels(rec, channels)
    names = [rec.channel_names[i] for i in ch_idx]
    t, data = _read_window_by_index(rec, start_s, duration_s, ch_idx)
    return names, t, data


def render_trace_window(
    rec: EEGRecording,
    start_s: float = 0.0,
    duration_s: float = 10.0,
    channels: list[str] | None = None,
    sensitivity_uv: float = 100.0,
    event_times_s: list[float] | None = None,
):
    """Return a matplotlib Figure of one stacked-trace window.

    sensitivity_uv sets the µV spacing between channel baselines (clinical
    "sensitivity"). event_times_s draws red markers (e.g. detected spikes).
    """
    from matplotlib.figure import Figure

    names, t, data = read_trace_window(rec, start_s, duration_s, channels)
    n_ch = data.shape[0]
    fig = Figure(figsize=(12, max(3.0, 0.4 * max(n_ch, 1))))
    ax = fig.subplots()
    spacing = float(sensitivity_uv) if sensitivity_uv > 0 else 100.0
    baselines = []
    for row in range(n_ch):
        offset = (n_ch - 1 - row) * spacing      # first channel at the top
        baselines.append(offset)
        if data.shape[1]:
            ax.plot(t, data[row] + offset, lw=0.5, color="black")
    ax.set_yticks(baselines)
    ax.set_yticklabels(names, fontsize=7)
    if t.size:
        ax.set_xlim(float(t[0]), float(t[-1]))
    ax.set_xlabel("time (s)")
    for et in (event_times_s or []):
        if t.size and t[0] <= et <= t[-1]:
            ax.axvline(et, color="red", lw=0.8, alpha=0.6)
    ax.set_title(f"{duration_s:.0f}s window @ {start_s:.1f}s · {spacing:.0f} µV/division")
    fig.tight_layout()
    return fig


def to_mne_raw(rec: EEGRecording, max_seconds: float = 600.0):
    """Build an mne.io.RawArray (EEG channels) for the desktop browser.

    Loads up to max_seconds to bound memory on long recordings.
    """
    import mne

    sf = float(rec.sfreq)
    ch_idx = rec.eeg_channel_indices or list(range(rec.n_channels_in_file))
    names = [str(rec.channel_names[i]) for i in ch_idx]
    n = int(min(rec.duration_s, max_seconds))
    # Read by INDEX (not by name) so a file with duplicate channel labels does
    # not alias two channels onto the same data.
    _t, data = _read_window_by_index(rec, 0.0, n, ch_idx)
    info = mne.create_info(ch_names=names, sfreq=sf, ch_types="eeg")
    return mne.io.RawArray(data * 1e-6, info, verbose="ERROR")  # µV → V for MNE


def launch_desktop_browser(rec: EEGRecording, max_seconds: float = 600.0) -> None:
    """Open mne's interactive Qt raw browser (desktop only; needs a Qt backend)."""
    raw = to_mne_raw(rec, max_seconds=max_seconds)
    raw.plot(block=True, title="KCNQ3-Lens raw trace")
