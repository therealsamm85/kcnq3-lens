"""Quantitative EEG analyses."""

from .topography import compute_topography, TopographyResult
from .spindles import compute_spindle_density, SpindleResult
from .background import compute_background_power, BackgroundResult
from .bursts import compute_sustained_bursts, BurstResult
from .morphology import compute_spike_morphology, MorphologyResult
from .time_of_night import compute_time_of_night, TimeOfNightResult
from .sleep_onset import detect_sleep_window, SleepWindowResult
from .quality import assess_quality, QualityResult
from .sleep_stages import compute_sleep_stages, SleepStageResult
from .swi import compute_swi, SWIResult
from .state_split import compute_state_split, StateSplitResult
from .synchrony import compute_synchrony, SynchronyResult
from .sleep_architecture import (
    compute_sleep_architecture, SleepArchitectureResult,
)
from .slow_waves import compute_slow_waves, SlowWaveResult
from .hfo_ripples import compute_hfo_ripples, summarize_hfo_ripples, HFORippleResult

__all__ = [
    "compute_topography", "TopographyResult",
    "compute_spindle_density", "SpindleResult",
    "compute_background_power", "BackgroundResult",
    "compute_sustained_bursts", "BurstResult",
    "compute_spike_morphology", "MorphologyResult",
    "compute_time_of_night", "TimeOfNightResult",
    "detect_sleep_window", "SleepWindowResult",
    "assess_quality", "QualityResult",
    "compute_sleep_stages", "SleepStageResult",
    "compute_swi", "SWIResult",
    "compute_state_split", "StateSplitResult",
    "compute_synchrony", "SynchronyResult",
    "compute_sleep_architecture", "SleepArchitectureResult",
    "compute_slow_waves", "SlowWaveResult",
    "compute_hfo_ripples", "summarize_hfo_ripples", "HFORippleResult",
]
