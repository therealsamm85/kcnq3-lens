"""Base class for EEG recordings — a uniform interface across readers."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class EEGRecording:
    """In-memory representation of an EEG recording.

    Data is accessed lazily through `read_epoch(epoch_index, epoch_seconds)`
    so multi-hour recordings don't have to fit in RAM.
    """

    path: Path
    sfreq: float                  # samples per second
    n_channels: int
    duration_s: float
    channel_names: list[str]      # length == n_channels
    n_channels_in_file: int       # may include marker/reference channels
    eeg_channel_indices: list[int]  # subset of channel_names that are real EEG
    format_name: str              # e.g. "Nihon Kohden EEG-1200A", "EDF"

    # Optional fields set by reader implementations:
    data_start_byte: int = 0
    bytes_per_sample: int = 2
    is_offset_binary: bool = False
    start_datetime: datetime.datetime | None = None
    # v0.14.3 H4: True when reader stripped a non-None tzinfo from meas_date.
    # UI can surface this as "UTC-normalized" so clock times aren't taken
    # to be local-tz exact.
    start_datetime_tz_stripped: bool = False
    # v0.18.1: True when reader detected a bipolar montage in the source file
    # (channel names like 'FP1-F7', 'F7-T7'). All analyses assume monopolar
    # referenced data; bipolar files load but produce scientifically wrong
    # results, so the app gates analyses on this flag.
    bipolar_montage_detected: bool = False
    bipolar_channel_examples: list[str] = field(default_factory=list)
    _read_epoch_fn: callable = field(default=None, repr=False)
    _full_data: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_epochs(self) -> int:
        """Number of 30-second epochs in the recording (whole epochs only)."""
        return int(self.duration_s) // 30

    def read_epoch(self, epoch_idx: int, epoch_seconds: float = 30.0) -> np.ndarray:
        """Read one epoch of multi-channel data.

        Returns array of shape (n_channels_in_file, n_samples).
        """
        if self._read_epoch_fn is not None:
            return self._read_epoch_fn(self, epoch_idx, epoch_seconds)
        if self._full_data is not None:
            n = int(epoch_seconds * self.sfreq)
            start = int(epoch_idx * n)
            end = start + n
            if end > self._full_data.shape[1]:
                return None
            return self._full_data[:, start:end]
        raise RuntimeError("No epoch reader configured for this recording.")

    def iter_epochs(
        self, epoch_seconds: float = 30.0, start: int = 0, end: int | None = None
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over epochs, yielding (epoch_idx, data)."""
        if end is None:
            end = self.n_epochs
        for ep in range(start, end):
            d = self.read_epoch(ep, epoch_seconds)
            if d is None:
                continue
            yield ep, d

    def get_eeg_data(self, channel_names: list[str] | None = None) -> np.ndarray:
        """Return EEG channels only, optionally filtered to a subset."""
        if channel_names is None:
            indices = self.eeg_channel_indices
        else:
            # v0.18.5: case-insensitive, consistent with channel_index().
            # Previously used case-sensitive `.index()`/`in`, so a request for
            # "cz" against a file storing "Cz" silently returned fewer channels.
            indices = [
                idx for n in channel_names
                if (idx := self.channel_index(n)) is not None
            ]
        if self._full_data is not None:
            return self._full_data[indices]
        raise NotImplementedError(
            "get_eeg_data() requires full in-memory loading; "
            "use read_epoch() for lazy access."
        )

    def time_at_hour(self, hour: float) -> str | None:
        """Return wall-clock 'HH:MM dayName' for a given recording hour, or None.

        Parameters
        ----------
        hour : float
            Hours elapsed since recording start (e.g. 7.17 = 7h 10min in).

        Returns
        -------
        str | None
            Formatted as 'HH:MM Mon' (locale-independent 3-letter day), or None
            if start_datetime is not available.
        """
        if self.start_datetime is None:
            return None
        t = self.start_datetime + datetime.timedelta(hours=hour)
        return t.strftime("%H:%M %a")

    def channel_index(self, name: str) -> int | None:
        """Return file-channel index for a named channel, or None if absent."""
        for i, n in enumerate(self.channel_names):
            if n.upper() == name.upper():
                return i
        return None

    def is_channel_live(self, ch_idx: int, flat_uv_threshold: float = 1.5,
                        sample_epochs: int = 3) -> bool:
        """Return False if a channel is essentially flat (dead / unplugged).

        A present-but-dead channel (e.g. an electrode that was disconnected
        but still occupies a slot in the file) silently corrupts any analysis
        that lands on it — the slow-wave detector once reported 0 slow waves
        on a 0.4 µV flat 'Fz'. Analyses use this to skip such channels.

        Samples a few epochs spread across the recording and returns True iff
        the median per-epoch std is at or above flat_uv_threshold µV. On any
        read failure it returns True (never block detection on a probe error).
        """
        try:
            n = self.n_epochs
            if n <= 0:
                return True
            if n >= sample_epochs:
                eps = [int((k + 1) * n / (sample_epochs + 1))
                       for k in range(sample_epochs)]
            else:
                eps = list(range(n))
            stds: list[float] = []
            for ep in eps:
                d = self.read_epoch(ep, 30.0)
                if d is None:
                    continue
                if 0 <= ch_idx < d.shape[0]:
                    stds.append(float(d[ch_idx].std()))
            if not stds:
                return True
            return float(np.median(stds)) >= flat_uv_threshold
        except Exception:
            return True

    def resolve_live_channel(
        self, candidates: list[str],
    ) -> tuple[int | None, str | None, bool]:
        """Resolve the first present-and-live channel from a candidate list.

        Returns (index, name, is_fallback). Tries each candidate in order and
        returns the first that is present AND live. If a candidate is present
        but dead, it is skipped. If none of the named candidates are live, it
        scans all EEG channels for any live one (is_fallback=True). If still
        none, returns the first present candidate (or first EEG channel) with
        is_fallback=True so behaviour degrades loudly rather than silently.
        Returns (None, None, False) only when the recording has no channels.
        """
        first_present_idx: int | None = None
        first_present_name: str | None = None
        for nm in candidates:
            idx = self.channel_index(nm)
            if idx is None:
                continue
            if first_present_idx is None:
                first_present_idx = idx
                first_present_name = self.channel_names[idx]
            if self.is_channel_live(idx):
                return idx, self.channel_names[idx], False
        # No named candidate was live — scan all EEG channels.
        for idx in self.eeg_channel_indices:
            if self.is_channel_live(idx):
                return idx, self.channel_names[idx], True
        if first_present_idx is not None:
            return first_present_idx, first_present_name, True
        if self.eeg_channel_indices:
            idx = self.eeg_channel_indices[0]
            return idx, self.channel_names[idx], True
        return None, None, False
