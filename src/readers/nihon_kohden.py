"""Nihon Kohden EEG-1200A binary reader.

This format has a two-block control structure: the first control block at 0x0400
points to a 1-second setup-data block (which is what MNE and EDFbrowser see), while
the actual multi-hour recording is in a second control block at 0x1F80, pointing
to data starting at 0x38E3. The n_ctlblocks byte at 0x0091 is set to 1, hiding
the second block from standard readers.

Data layout from 0x38E3 onwards:
- Uncompressed int16, multiplexed across channels
- Offset-binary encoding (XOR with 0x8000 to get signed)
- All channels per time-point, then next time-point
- Default: 29 channels (28 EEG + 1 marker), 200 Hz

This reader was reverse-engineered from one recording family; other Nihon Kohden
EEG-1200A files may have minor variations. Test before clinical use.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

from .base import EEGRecording

# Header constants for the common EEG-1200A layout seen in NKT EEG2100 systems
_DEFAULT_DATA_START = 0x38E3
_DEFAULT_N_CH_FILE = 29

# ADC→µV calibration for EEG channels (v0.18.4).
# The reverse-engineered long-form reader decodes offset-binary int16 to signed
# ADC counts but historically returned those counts directly, leaving the
# fallback path ~10x mis-scaled vs the MNE path (which applies a gain). The
# Nihon Kohden EEG-1200A fixed calibration for standard EEG channels is
# phys_min=-3200 µV, phys_max=3199.902 µV over the 16-bit digital range — i.e.
# the same constants MNE's nihon reader uses (mne/io/nihon/nihon.py). The
# per-count gain is therefore (phys_max - phys_min) / 65535. Applying it makes
# the fallback path produce µV consistent with the MNE path (verified: the 24h
# FA06301E file lands at ~27 µV median per-channel std, matching the four
# MNE-read short recordings at 25-47 µV).
_NK_EEG_PHYS_MIN_UV = -3200.0
_NK_EEG_PHYS_MAX_UV = 3199.902
_NK_ADC_TO_UV = (_NK_EEG_PHYS_MAX_UV - _NK_EEG_PHYS_MIN_UV) / 65535.0  # ≈ 0.09766 µV/count
_DEFAULT_SFREQ = 200
_DEFAULT_CH_NAMES_29 = [
    "Fp1", "F4", "F3", "C4", "C3", "P4", "P3", "O2", "O1",
    "F8", "F7", "T4", "T3", "T6", "T5", "Fz", "Cz", "Pz",
    "E", "A2", "A1", "X1", "X2", "X3", "X4", "$A2", "$A1", "Fp2",
    "MARK",
]
_STANDARD_EEG_NAMES = {
    "Fp1", "Fp2", "F3", "F4", "Fz", "F7", "F8",
    "C3", "C4", "Cz",
    "P3", "P4", "Pz",
    "T3", "T4", "T5", "T6",
    "O1", "O2",
}


def is_nihon_kohden(path: Path) -> bool:
    """Detect Nihon Kohden EEG-1200A by file signature."""
    path = Path(path)
    if path.suffix.lower() not in (".eeg", ".m00"):
        # M00 is older format, EEG is the modern one
        return False
    try:
        with open(path, "rb") as fh:
            head = fh.read(32)
        return b"EEG-1200A" in head or b"EEG-1100" in head
    except OSError:
        return False


def _read_via_mne(path: Path) -> EEGRecording | None:
    """Try MNE-Python's read_raw_nihon. Returns None if MNE can't parse.

    MNE-Python correctly handles per-file sample rate, gain calibration
    (returns volts → we convert to µV), and channel naming for the
    "modern" Nihon Kohden file layout used by short routine recordings.
    It does NOT handle the long-form (multi-hour) variant that ships with
    only the second control block at 0x1F80; for those we fall back to
    the reverse-engineered reader below. v0.18.3.
    """
    try:
        import mne
    except ImportError:
        return None
    try:
        mne.set_log_level("WARNING")
        raw = mne.io.read_raw_nihon(str(path), preload=True)
    except Exception:
        return None
    # Sanity-check: MNE sometimes "succeeds" on the long-form variant but
    # returns a 1-second, 1-channel stub (header parsing breaks). Reject
    # that and fall back to the custom reader.
    if raw.n_times < int(raw.info["sfreq"] * 30) or len(raw.ch_names) < 5:
        return None

    sfreq = float(raw.info["sfreq"])
    channel_names = list(raw.ch_names)
    n_channels_in_file = len(channel_names)
    duration_s = float(raw.times[-1])

    # MNE returns volts; multiply by 1e6 → µV (matches our pipeline).
    data_uv = (raw.get_data() * 1e6).astype(np.float32)

    # Filter to standard 10-20 channels for the EEG-only subset, matching
    # the custom-reader behaviour. Case-insensitive (NK files may use
    # "Fp1" or "FP1" depending on the device firmware).
    upper = {n.upper() for n in _STANDARD_EEG_NAMES}
    eeg_channel_indices = [
        i for i, n in enumerate(channel_names)
        if n.upper() in upper
    ]

    # Recording start time
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

    rec = EEGRecording(
        path=path,
        sfreq=sfreq,
        n_channels=len(eeg_channel_indices),
        duration_s=duration_s,
        channel_names=channel_names,
        n_channels_in_file=n_channels_in_file,
        eeg_channel_indices=eeg_channel_indices,
        format_name="Nihon Kohden (via MNE)",
        start_datetime=start_datetime,
        start_datetime_tz_stripped=tz_stripped,
    )
    rec._full_data = data_uv
    return rec


def read_nihon_kohden(
    path: Path,
    sfreq: float | None = None,
    n_channels: int | None = None,
    channel_names: list[str] | None = None,
    data_start: int | None = None,
    duration_s: float | None = None,
) -> EEGRecording:
    """Read a Nihon Kohden EEG-1200A file and return an EEGRecording.

    Strategy: try MNE-Python's read_raw_nihon first (correct µV gain,
    per-file sample rate, real channel names). If MNE can't parse the
    file — which happens for the long-form recording variant — fall
    back to the reverse-engineered reader below.

    Parameters
    ----------
    path : Path
        Path to the .EEG file.
    sfreq, n_channels, channel_names, data_start, duration_s :
        Manual overrides for the reverse-engineered fallback reader.
        When any of these is supplied we skip the MNE attempt and use
        the explicit parameters (preserves the original API for
        recordings the user knows MNE can't handle).
    """
    path = Path(path)

    # When no manual overrides are given, try MNE first. MNE delivers
    # correct µV scaling so downstream YASA-based analyses work.
    if all(p is None for p in (sfreq, n_channels, channel_names,
                                data_start, duration_s)):
        rec = _read_via_mne(path)
        if rec is not None:
            return rec


    sfreq = sfreq or _DEFAULT_SFREQ
    n_channels = n_channels or _DEFAULT_N_CH_FILE
    channel_names = channel_names or _DEFAULT_CH_NAMES_29[:n_channels]
    data_start = data_start or _DEFAULT_DATA_START

    # Parse recording start time from header offset 64.
    # The EEG-1200A stores 14-char ASCII 'YYYYMMDDHHMMSS' at that offset.
    # This was verified on an NKT EEG2100 recording; other device families
    # may vary. Failure is silent — start_datetime stays None.
    start_datetime: dt.datetime | None = None
    try:
        with open(path, "rb") as _fh:
            _head = _fh.read(78)
        if len(_head) >= 78:
            raw_dt = _head[64:78].decode("ascii", errors="replace").strip("\x00 ")
            if len(raw_dt) >= 14 and raw_dt[:14].isdigit():
                start_datetime = dt.datetime.strptime(raw_dt[:14], "%Y%m%d%H%M%S")
    except Exception:
        start_datetime = None

    # v0.14.3 H3: plausibility check — reject obviously bogus year values
    if start_datetime is not None:
        if not (1990 <= start_datetime.year <= 2030):
            start_datetime = None

    file_size = path.stat().st_size
    data_bytes = file_size - data_start
    bytes_per_sample = 2  # int16
    samples_per_ch = data_bytes // (n_channels * bytes_per_sample)
    auto_duration_s = samples_per_ch / sfreq

    if duration_s is None:
        duration_s = auto_duration_s

    eeg_channel_indices = [
        i for i, name in enumerate(channel_names) if name in _STANDARD_EEG_NAMES
    ]

    def _read_epoch(rec: EEGRecording, ep: int, eps_s: float = 30.0) -> np.ndarray | None:
        n_samples = int(eps_s * rec.sfreq)
        bytes_per_epoch = n_samples * rec.n_channels_in_file * 2
        with open(rec.path, "rb") as fh:
            fh.seek(rec.data_start_byte + ep * bytes_per_epoch)
            buf = fh.read(bytes_per_epoch)
        if len(buf) < bytes_per_epoch:
            return None
        u16 = np.frombuffer(buf, dtype="<u2")
        # Offset-binary to signed int16
        s16 = (u16.astype(np.int32) + 0x8000).astype(np.int16)
        # v0.18.4: scale signed ADC counts to µV so the fallback path matches
        # the MNE path (which already returns µV). Without this the long-form
        # 24h recording was ~10x mis-scaled, breaking YASA amplitude-threshold
        # detectors (spindles, slow waves) and any µV-based interpretation.
        counts = s16.reshape(-1, rec.n_channels_in_file).T.astype(np.float32)
        return counts * np.float32(_NK_ADC_TO_UV)

    return EEGRecording(
        path=path,
        sfreq=sfreq,
        n_channels=len(eeg_channel_indices),
        duration_s=duration_s,
        channel_names=channel_names,
        n_channels_in_file=n_channels,
        eeg_channel_indices=eeg_channel_indices,
        format_name="Nihon Kohden EEG-1200A",
        data_start_byte=data_start,
        bytes_per_sample=2,
        is_offset_binary=True,
        start_datetime=start_datetime,
        _read_epoch_fn=_read_epoch,
    )
