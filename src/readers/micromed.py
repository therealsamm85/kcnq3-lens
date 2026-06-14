"""E1 — Micromed .TRC reader (+ Natus note).  [BORROW: python-neo, optional]

Micromed Brain-Quick is the dominant long-term/EMU epilepsy-monitoring format in
Europe; MNE has no native .TRC reader, so the existing MNE fallback silently
cannot open these files. python-neo's MicromedRawIO can, and is local-only.

BORROW: thin adapter from neo's MicromedRawIO to the project's EEGRecording.
Optional dependency — if neo is absent, `can_read_micromed()` is False and the
loader raises a clear, actionable error (install neo) rather than mis-reading.

Natus/XLTek NeuroWorks has no robust open reader (the one community script is
archived/unlicensed), so it is intentionally NOT implemented here — see
`NATUS_NOTE`; the documented mitigation is an EDF+ export at the acquisition site.

SCAFFOLD — implemented in wave E1.
"""
from __future__ import annotations

from pathlib import Path

from .base import EEGRecording

NATUS_NOTE = (
    "Natus/XLTek NeuroWorks (.eeg/.erd/.ent) has no robust open-source reader. "
    "Export the study to EDF+ at the acquisition site (Natus/Persyst can do this) "
    "and load the EDF+ instead."
)


def can_read_micromed() -> bool:
    """True if python-neo is importable. SCAFFOLD — wave E1."""
    raise NotImplementedError("scaffold — implemented in wave E1")


def read_micromed(path: str | Path) -> EEGRecording:
    """Read a Micromed .TRC file into an EEGRecording. SCAFFOLD — wave E1."""
    raise NotImplementedError("scaffold — implemented in wave E1")
