"""C1 — Spike-triggered averaging → peak voltage topography.  [BUILD on mne]

Averaging the detected IEDs at their peaks yields a clean spike field whose
topography answers the ESES focal-vs-secondary-bilateral-synchrony question that
a per-channel count cannot. Reuses the spike event times the morphology detector
already exports (`_morphology_events`) + the existing topography machinery.

BUILD: epoch extraction + averaging + peak-topography is straightforward numpy/
mne. Equivalent-dipole / template-MRI source localisation is intentionally a
documented stub (needs a head model + electrode coregistration that a routine
clinical montage cannot reliably supply).

SCAFFOLD — implemented in wave C1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..readers.base import EEGRecording


@dataclass
class SpikeAverageResult:
    n_spikes_averaged: int
    window_ms: tuple[float, float]
    peak_channel: str | None
    peak_topography: dict[str, float] = field(default_factory=dict)
    field_spread: str = ""          # "focal" | "regional" | "bilateral" | "n/a"
    notes: list[str] = field(default_factory=list)


def compute_spike_average(
    rec: EEGRecording,
    spike_events: list[dict] | None = None,
    window_ms: tuple[float, float] = (-100.0, 100.0),
    max_spikes: int = 500,
) -> SpikeAverageResult:
    """Average detected spikes → peak topography. SCAFFOLD — wave C1."""
    raise NotImplementedError("scaffold — implemented in wave C1")


def summarize_spike_average(result: SpikeAverageResult) -> dict:
    raise NotImplementedError("scaffold — implemented in wave C1")
