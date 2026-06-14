"""B2 — Artifact Subspace Reconstruction (ASR).  [BORROW: asrpy, optional]

ASR corrects transient high-amplitude bursts (movement, electrode pops) by
reconstructing them from a clean calibration subspace — it *salvages* segments
instead of dropping them, valuable in fidgety pediatric recordings, and is
complementary to the epoch rejection already present.

BORROW: asrpy is a maintained MNE-native Python port of EEGLAB clean_rawdata.
Optional dependency, graceful degrade when absent (available=False). Because ASR
can distort genuine epileptiform transients if the cutoff is aggressive, the
default cutoff is conservative and the result warns that epileptiform morphology
must be re-checked on the corrected signal.

SCAFFOLD — implemented in wave B2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..readers.base import EEGRecording


@dataclass
class AsrResult:
    available: bool
    cutoff: float = 0.0
    fraction_corrected: float = 0.0
    backend: str = ""
    notes: list[str] = field(default_factory=list)


def run_asr(
    rec: EEGRecording,
    cutoff: float = 20.0,
    calibration_seconds: float = 60.0,
) -> AsrResult:
    """Apply ASR burst correction (optional asrpy). SCAFFOLD — wave B2."""
    raise NotImplementedError("scaffold — implemented in wave B2")


def summarize_asr(result: AsrResult) -> dict:
    raise NotImplementedError("scaffold — implemented in wave B2")
