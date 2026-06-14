"""E1 — Micromed .TRC reader (+ Natus note).  [BORROW: python-neo, optional]

Micromed Brain-Quick is the dominant long-term/EMU epilepsy-monitoring format in
Europe; MNE has no native .TRC reader, so the existing MNE fallback silently
cannot open these files. python-neo's MicromedRawIO can, and is local-only.

BORROW: thin adapter from neo's MicromedRawIO to the project's EEGRecording.
Optional dependency — if neo is absent, ``can_read_micromed()`` is False and the
loader raises a clear, actionable error (install neo) rather than mis-reading.

Natus/XLTek NeuroWorks has no robust open reader (the one community script is
archived/unlicensed), so it is intentionally NOT implemented here — see
``NATUS_NOTE``; the documented mitigation is an EDF+ export at the acquisition
site.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import EEGRecording

NATUS_NOTE = (
    "Natus/XLTek NeuroWorks (.eeg/.erd/.ent) has no robust open-source reader. "
    "Export the study to EDF+ at the acquisition site (Natus/Persyst can do this) "
    "and load the EDF+ instead."
)

# Reuse the project's EEG-name allowlist to flag real EEG channels.
try:
    from .edf import _STANDARD_EEG_NAMES_UPPER as _EEG_NAMES_UPPER
except Exception:  # pragma: no cover - fallback if the symbol moves
    _EEG_NAMES_UPPER = set()


def can_read_micromed() -> bool:
    """True if python-neo (with MicromedRawIO) is importable."""
    try:
        from neo.rawio import MicromedRawIO  # noqa: F401
        return True
    except ImportError:
        return False


def read_micromed(path: str | Path) -> EEGRecording:
    """Read a Micromed .TRC file into an EEGRecording (requires python-neo)."""
    path = Path(path)
    try:
        from neo.rawio import MicromedRawIO
    except ImportError as e:
        raise ImportError(
            "Reading Micromed .TRC files requires python-neo. "
            "Install it with `pip install neo`."
        ) from e
    if not path.exists():
        raise FileNotFoundError(f"EEG file not found: {path}")

    reader = MicromedRawIO(filename=str(path))
    reader.parse_header()

    sig_chans = reader.header["signal_channels"]
    raw_names = [str(n) for n in sig_chans["name"]]
    # Normalize like the EDF reader so referenced/prefixed labels ('EEG Fp1',
    # 'Fp1-G2', 'Fp1-Ref') still match the EEG allowlist — otherwise NONE match
    # and every channel (incl. ECG/EMG/markers) is misclassified as EEG.
    names = [n.replace("EEG ", "").replace("eeg ", "").split("-")[0].strip()
             for n in raw_names]
    sfreq = float(reader.get_signal_sampling_rate())
    n_samples = int(reader.get_signal_size(block_index=0, seg_index=0))

    raw = reader.get_analogsignal_chunk(block_index=0, seg_index=0)
    # Rescale ADC → physical units (Micromed stores µV).
    data = reader.rescale_signal_raw_to_float(raw, dtype="float64").T  # (n_ch, n_samp)

    if _EEG_NAMES_UPPER:
        eeg_idx = [i for i, nm in enumerate(names) if nm.upper() in _EEG_NAMES_UPPER]
    else:
        eeg_idx = []
    if not eeg_idx:
        eeg_idx = list(range(len(names)))  # treat all as EEG if none recognized

    rec = EEGRecording(
        path=path, sfreq=sfreq, n_channels=len(eeg_idx),
        duration_s=n_samples / sfreq if sfreq else 0.0,
        channel_names=names, n_channels_in_file=len(names),
        eeg_channel_indices=eeg_idx, format_name="Micromed .TRC (via neo)",
    )
    rec._full_data = np.ascontiguousarray(data, dtype=np.float32)
    return rec
