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


def read_nihon_kohden(
    path: Path,
    sfreq: float | None = None,
    n_channels: int | None = None,
    channel_names: list[str] | None = None,
    data_start: int | None = None,
    duration_s: float | None = None,
) -> EEGRecording:
    """Read a Nihon Kohden EEG-1200A file and return an EEGRecording.

    Parameters
    ----------
    path : Path
        Path to the .EEG file.
    sfreq : float, optional
        Sampling rate. Defaults to 200 Hz (standard for long-term overnight).
    n_channels : int, optional
        Channels per time-point in file. Defaults to 29 (28 EEG + 1 marker).
    channel_names : list[str], optional
        Channel names in file order. Defaults to standard EEG-1200A layout.
    data_start : int, optional
        Byte offset where waveform data begins. Defaults to 0x38E3.
    duration_s : float, optional
        Recording duration in seconds. Auto-inferred from file size if not given.
    """
    path = Path(path)
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
        return s16.reshape(-1, rec.n_channels_in_file).T.astype(np.float32)

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
