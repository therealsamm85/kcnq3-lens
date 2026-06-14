"""B1 — ICA decomposition + automatic component classification.  [BORROW]

The project detects/masks blinks and rejects epochs but does no ICA source
separation, so ocular/muscle/cardiac activity is dropped (whole epochs) rather
than removed (component) — costly on long sleep recordings. This wraps mne's ICA
with mne-icalabel's ICLabel classifier to remove artifact components and keep the
rest of the data.

BORROW: mne ICA (already a dep) + mne-icalabel (optional, fully local via
onnxruntime/torch). Degrades gracefully when mne-icalabel is absent (returns
available=False, like the yasa/specparam fallbacks). The HAPPE wavelet-enhanced
ICA (W-ICA) idea is offered as an option flag on top of the same decomposition.

SCAFFOLD — implemented in wave B1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..readers.base import EEGRecording


@dataclass
class IcaResult:
    available: bool
    n_components: int = 0
    labels: list[str] = field(default_factory=list)          # per-component ICLabel class
    removed_components: list[int] = field(default_factory=list)
    removed_classes: dict[str, int] = field(default_factory=dict)
    backend: str = ""
    notes: list[str] = field(default_factory=list)


def run_ica_cleanup(
    rec: EEGRecording,
    remove_classes: tuple[str, ...] = ("eye blink", "muscle artifact", "heart beat"),
    wavelet_enhanced: bool = False,
) -> IcaResult:
    """Fit ICA, label components, remove artifact classes. SCAFFOLD — wave B1."""
    raise NotImplementedError("scaffold — implemented in wave B1")


def summarize_ica(result: IcaResult) -> dict:
    raise NotImplementedError("scaffold — implemented in wave B1")
