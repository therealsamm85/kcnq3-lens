"""A4 — Raw-trace viewer.  [BORROW: mne / matplotlib (already in stack)]

A metrics+PDF UI gives no way to eyeball the actual spike-wave behind a number.
This renders navigable multi-channel trace windows (a clinical-style page: fixed
time window, selectable montage, µV sensitivity, optional event overlays) for the
Streamlit UI, and offers a desktop hand-off to mne's interactive Qt browser.

BORROW: builds on mne's plotting / matplotlib that the project already depends
on — no new dependency. The Streamlit-friendly static renderer is the primary
path (fully local, no Qt requirement); `launch_desktop_browser()` is an optional
convenience for users running locally with a Qt backend.

SCAFFOLD — implemented in wave A4.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..readers.base import EEGRecording


@dataclass
class TraceWindow:
    start_s: float
    duration_s: float
    channels: list[str]
    sfreq: float
    note: str = ""
    notes: list[str] = field(default_factory=list)


def render_trace_window(
    rec: EEGRecording,
    start_s: float = 0.0,
    duration_s: float = 10.0,
    channels: list[str] | None = None,
    sensitivity_uv: float = 100.0,
    event_times_s: list[float] | None = None,
):
    """Return a matplotlib Figure for one trace window. SCAFFOLD — wave A4."""
    raise NotImplementedError("scaffold — implemented in wave A4")


def launch_desktop_browser(rec: EEGRecording) -> None:
    """Open mne's interactive Qt raw browser (desktop only). SCAFFOLD — wave A4."""
    raise NotImplementedError("scaffold — implemented in wave A4")
