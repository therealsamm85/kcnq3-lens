"""Base class for EEG recordings — a uniform interface across readers."""

from __future__ import annotations

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
            indices = [
                self.channel_names.index(n)
                for n in channel_names
                if n in self.channel_names
            ]
        if self._full_data is not None:
            return self._full_data[indices]
        raise NotImplementedError(
            "get_eeg_data() requires full in-memory loading; "
            "use read_epoch() for lazy access."
        )

    def channel_index(self, name: str) -> int | None:
        """Return file-channel index for a named channel, or None if absent."""
        for i, n in enumerate(self.channel_names):
            if n.upper() == name.upper():
                return i
        return None
