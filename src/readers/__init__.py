"""EEG file format readers."""

from .auto_detect import load_eeg, EEGRecording

__all__ = ["load_eeg", "EEGRecording"]
