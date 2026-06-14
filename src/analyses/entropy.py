"""A2 — Entropy / complexity metrics.  [BUILD on numpy; antropy optional accel]

Sample / permutation / spectral entropy, Hjorth parameters, Higuchi fractal
dimension and Lempel-Ziv complexity — published markers of encephalopathic
background disorganization. Cheap, local, single-patient-friendly; feed straight
into the longitudinal trackers as per-recording background-quality features.

BUILD (not borrow): each metric is a few lines of numpy, and the project is
dependency-conservative. antropy, if installed, is used as a faster/validated
backend; otherwise the numpy implementations run.

SCAFFOLD — implemented in wave A2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..readers.base import EEGRecording


@dataclass
class EntropyResult:
    channel: str
    metrics: dict[str, float] = field(default_factory=dict)
    backend: str = "numpy"
    notes: list[str] = field(default_factory=list)


def compute_entropy(
    rec: EEGRecording,
    target_channel: str = "Pz",
    max_epochs: int = 200,
) -> EntropyResult:
    """Compute entropy/complexity metrics on one channel. SCAFFOLD — wave A2."""
    raise NotImplementedError("scaffold — implemented in wave A2")


def summarize_entropy(result: EntropyResult) -> dict:
    raise NotImplementedError("scaffold — implemented in wave A2")
