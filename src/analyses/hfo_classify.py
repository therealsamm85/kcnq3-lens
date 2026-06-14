"""C2 — Two-stage HFO classification.  [BUILD on existing HFO events]

The HFO detector finds candidate ripples but does not say which are real vs
artifact, or which co-occur with a spike (spkHFO) — the distinction that cuts
the false-positive burden plaguing scalp HFOs. This adds a transparent
feature-based second stage on the events already in `_hfo_ripples_events`.

BUILD (not borrow): PyHFO's deep classifiers need torch + a UCLA academic
licence — too heavy / wrong licence for a local family tool. The interpretable
features (oscillatory autocorrelation, spectral peak, amplitude stability, blip
rejection, spike-time coincidence) are computed directly here.

SCAFFOLD — implemented in wave C2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..readers.base import EEGRecording


@dataclass
class HfoClassifyResult:
    n_input: int
    n_artifact: int
    n_real: int
    n_spike_coupled: int
    per_event: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def classify_hfos(
    rec: EEGRecording,
    hfo_events: list[dict] | None = None,
    spike_events: list[dict] | None = None,
) -> HfoClassifyResult:
    """Classify HFO events artifact/real/spike-coupled. SCAFFOLD — wave C2."""
    raise NotImplementedError("scaffold — implemented in wave C2")


def summarize_hfo_classify(result: HfoClassifyResult) -> dict:
    raise NotImplementedError("scaffold — implemented in wave C2")
