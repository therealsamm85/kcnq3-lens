"""EDF / EDF+ reader using MNE-Python.

Covers EDF, EDF+, BDF, BrainVision (.vhdr), and most other formats MNE supports.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import EEGRecording

_STANDARD_EEG_NAMES = {
    "Fp1", "Fp2", "F3", "F4", "Fz", "F7", "F8",
    "C3", "C4", "Cz",
    "P3", "P4", "Pz",
    "T3", "T4", "T5", "T6",
    "O1", "O2",
    # 10-20 system alternates
    "T7", "T8", "P7", "P8",
}


def is_edf_compatible(path: Path) -> bool:
    """Detect EDF / EDF+ / BDF / BrainVision by extension and header."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".edf", ".bdf", ".vhdr", ".set"):
        return True
    # EDF files start with "0       " (8 bytes, version number)
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
        return head.startswith(b"0       ") or head.startswith(b"\xff")  # BDF
    except OSError:
        return False


def read_edf(path: Path) -> EEGRecording:
    """Read EDF / EDF+ / BDF using MNE-Python and return an EEGRecording."""
    import mne

    path = Path(path)
    mne.set_log_level("WARNING")

    ext = path.suffix.lower()
    if ext == ".bdf":
        raw = mne.io.read_raw_bdf(str(path), preload=True)
    elif ext == ".vhdr":
        raw = mne.io.read_raw_brainvision(str(path), preload=True)
    elif ext == ".set":
        raw = mne.io.read_raw_eeglab(str(path), preload=True)
    else:
        raw = mne.io.read_raw_edf(str(path), preload=True)

    sfreq = float(raw.info["sfreq"])
    channel_names = list(raw.ch_names)
    n_channels = len(channel_names)
    duration_s = float(raw.times[-1])

    # Normalize channel names: strip prefixes/suffixes like "EEG Fp1-A1"
    normalized = []
    for n in channel_names:
        # Common patterns: "EEG Fp1", "EEG Fp1-A1", "Fp1-Ref", "Fp1"
        cleaned = n.replace("EEG ", "").split("-")[0].strip()
        normalized.append(cleaned)

    eeg_channel_indices = [
        i for i, name in enumerate(normalized) if name in _STANDARD_EEG_NAMES
    ]

    data = raw.get_data()  # shape (n_ch, n_samples), in volts
    # Convert to microvolts for consistency with NK ADC-style numbers
    data_uv = data * 1e6

    # Extract recording start time from MNE's meas_date (timezone-aware or None)
    import datetime as _dt
    start_datetime = None
    tz_stripped = False
    meas_date = raw.info.get("meas_date")
    if meas_date is not None:
        try:
            had_tz = getattr(meas_date, "tzinfo", None) is not None
            if hasattr(meas_date, "replace"):
                start_datetime = meas_date.replace(tzinfo=None)
            tz_stripped = bool(had_tz)
        except Exception:
            start_datetime = None
            tz_stripped = False

    rec = EEGRecording(
        path=path,
        sfreq=sfreq,
        n_channels=len(eeg_channel_indices),
        duration_s=duration_s,
        channel_names=normalized,
        n_channels_in_file=n_channels,
        eeg_channel_indices=eeg_channel_indices,
        format_name=f"EDF / {ext.lstrip('.').upper()}",
        start_datetime=start_datetime,
        start_datetime_tz_stripped=tz_stripped,
    )
    rec._full_data = data_uv.astype(np.float32)
    return rec
