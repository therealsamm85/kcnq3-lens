"""Quantitative EEG analyses."""

from .topography import compute_topography, TopographyResult
from .spindles import compute_spindle_density, SpindleResult
from .background import compute_background_power, BackgroundResult
from .bursts import compute_sustained_bursts, BurstResult
from .morphology import compute_spike_morphology, MorphologyResult
from .time_of_night import compute_time_of_night, TimeOfNightResult

__all__ = [
    "compute_topography", "TopographyResult",
    "compute_spindle_density", "SpindleResult",
    "compute_background_power", "BackgroundResult",
    "compute_sustained_bursts", "BurstResult",
    "compute_spike_morphology", "MorphologyResult",
    "compute_time_of_night", "TimeOfNightResult",
]
