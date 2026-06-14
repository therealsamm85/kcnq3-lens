"""Auto-detect EEG file format and dispatch to the right reader."""

from __future__ import annotations

from pathlib import Path

from .base import EEGRecording
from .nihon_kohden import is_nihon_kohden, read_nihon_kohden
from .edf import is_edf_compatible, read_edf


def load_eeg(path: str | Path, **kwargs) -> EEGRecording:
    """Load an EEG file, auto-detecting its format.

    Currently supports:
    - Nihon Kohden EEG-1200A (.eeg)
    - EDF / EDF+ (.edf)
    - BDF (.bdf)
    - BrainVision (.vhdr)
    - EEGLAB (.set)

    Returns an EEGRecording with a uniform interface.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EEG file not found: {path}")

    # Order matters: check the more specific format first
    if is_nihon_kohden(path):
        return read_nihon_kohden(path, **kwargs)

    if is_edf_compatible(path):
        return read_edf(path)

    # Micromed .TRC (European EMU systems) via the optional python-neo backend.
    if path.suffix.lower() == ".trc":
        from .micromed import read_micromed
        return read_micromed(path)

    raise ValueError(
        f"Could not detect EEG format for {path.name}. "
        f"Supported: .eeg (Nihon Kohden), .edf, .bdf, .vhdr, .set, "
        f".trc (Micromed, needs python-neo)"
    )
