"""C3 — Ictal (electrographic seizure) screener.  [BUILD, flag-for-review]

The tool is otherwise interictal-only; it would silently ignore an electrographic
seizure in an overnight recording. This is a sensitivity-first heuristic screener
that flags candidate evolving rhythmic runs for HUMAN review — explicitly not a
diagnosis.

BUILD (not borrow): DeepSOZ and the TUSZ/CHB-MIT DL detectors are adult-trained
on specific montages, unvalidated in pediatric ESES, and pull in torch. A
transparent screener (sustained line-length elevation + rhythmic spectral
evolution / frequency drift over a sustained window) is auditable, local, and
appropriately humble. Every flag carries a "confirm by reading the trace" caveat.

SCAFFOLD — implemented in wave C3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..readers.base import EEGRecording


@dataclass
class IctalCandidate:
    start_s: float
    duration_s: float
    channel: str
    peak_line_length_z: float
    freq_drift_hz: float
    confidence: str             # "low" | "moderate" (never "high" — screening only)


@dataclass
class IctalScreenResult:
    n_candidates: int
    candidates: list[IctalCandidate] = field(default_factory=list)
    minutes_screened: float = 0.0
    notes: list[str] = field(default_factory=list)
    caveat: str = ""


def screen_ictal(
    rec: EEGRecording,
    target_channels: list[str] | None = None,
    min_event_s: float = 10.0,
) -> IctalScreenResult:
    """Flag candidate electrographic seizures for review. SCAFFOLD — wave C3."""
    raise NotImplementedError("scaffold — implemented in wave C3")


def summarize_ictal(result: IctalScreenResult) -> dict:
    raise NotImplementedError("scaffold — implemented in wave C3")
