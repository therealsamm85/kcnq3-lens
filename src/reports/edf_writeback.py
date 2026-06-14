"""A1 — Annotated EDF+ write-back.  [BORROW: edfio, already in stack]

Export the recording's signal plus the tool's detected events (spikes, SWI
bursts, HFOs, sleep stages) as an EDF+ that any neurologist can open in free
EDFbrowser / Persyst / a clinical reviewer — no proprietary platform. edfio is
already a dependency, so this is wiring, not a new dep.

SCAFFOLD — implemented in wave A1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..readers.base import EEGRecording


@dataclass
class EdfWritebackResult:
    out_path: str
    n_annotations: int
    channels_written: int
    note: str = ""
    notes: list[str] = field(default_factory=list)


def export_annotated_edf(
    rec: EEGRecording,
    out_path: str | Path,
    events: list[dict] | None = None,
    *,
    anonymize: bool = True,
) -> EdfWritebackResult:
    """Write an EDF+ with the recording signal + `events` as annotations.

    SCAFFOLD — built in wave A1.
    """
    raise NotImplementedError("scaffold — implemented in wave A1")
