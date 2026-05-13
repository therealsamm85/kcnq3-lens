"""Run the five core analyses on a single EEGRecording and collect findings.

Used by both the single-recording and pre/post-comparison views in the app.
Pure function — no Streamlit dependency.
"""

from __future__ import annotations

from typing import Any, Callable

from .readers.base import EEGRecording
from .analyses import (
    compute_topography,
    compute_spindle_density,
    compute_background_power,
    compute_sustained_bursts,
    compute_spike_morphology,
    compute_time_of_night,
)
from .analyses.topography import summarize_topography
from .analyses.spindles import summarize_spindles
from .analyses.background import summarize_background
from .analyses.bursts import summarize_bursts
from .analyses.morphology import summarize_morphology
from .analyses.time_of_night import summarize_time_of_night


def run_all_analyses(
    rec: EEGRecording,
    sleep_start_epoch: int,
    sleep_end_epoch: int,
    wake_epoch_indices: list[int],
    age_years: float | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    """Run all five analyses and return a structured findings dict.

    Parameters
    ----------
    rec : EEGRecording
    sleep_start_epoch, sleep_end_epoch : int
        Epoch range covering the sleep period.
    wake_epoch_indices : list[int]
        Specific epoch indices considered wake.
    age_years : float, optional
        Used for age-normative comparisons.
    progress_callback : callable, optional
        Called as progress_callback(step_name, fraction_complete) for UI updates.

    Returns
    -------
    dict with keys: topography, spindles, background, bursts, morphology, errors
    """
    findings: dict[str, Any] = {}
    errors: dict[str, str] = {}

    def _emit(name: str, frac: float):
        if progress_callback:
            progress_callback(name, frac)

    # --- 1. Topography ---
    _emit("topography", 0.05)
    try:
        topo = compute_topography(
            rec, start_epoch=sleep_start_epoch, end_epoch=sleep_end_epoch
        )
        findings["topography"] = summarize_topography(topo, top_n=10)
    except Exception as e:
        errors["topography"] = str(e)
    _emit("topography", 0.30)

    # --- 2. Spindles ---
    try:
        spindle = compute_spindle_density(
            rec,
            sleep_start_epoch=sleep_start_epoch,
            sleep_end_epoch=sleep_end_epoch,
            age_years=age_years,
        )
        findings["spindles"] = summarize_spindles(spindle)
    except Exception as e:
        errors["spindles"] = str(e)
    _emit("spindles", 0.50)

    # --- 3. Background ---
    try:
        bg = compute_background_power(
            rec,
            wake_epoch_indices=wake_epoch_indices,
            age_years=age_years,
        )
        findings["background"] = summarize_background(bg)
    except Exception as e:
        errors["background"] = str(e)
    _emit("background", 0.70)

    # --- 4. Bursts ---
    try:
        bursts = compute_sustained_bursts(
            rec, start_epoch=sleep_start_epoch, end_epoch=sleep_end_epoch
        )
        findings["bursts"] = summarize_bursts(bursts)
    except Exception as e:
        errors["bursts"] = str(e)
    _emit("bursts", 0.90)

    # --- 5. Morphology ---
    try:
        morph = compute_spike_morphology(
            rec, start_epoch=sleep_start_epoch, end_epoch=sleep_end_epoch
        )
        findings["morphology"] = summarize_morphology(morph)
    except Exception as e:
        errors["morphology"] = str(e)
    _emit("morphology", 0.95)

    # --- 6. Time-of-night burden ---
    try:
        tn = compute_time_of_night(
            rec,
            start_epoch=sleep_start_epoch,
            end_epoch=sleep_end_epoch,
            bin_minutes=30.0,
        )
        findings["time_of_night"] = summarize_time_of_night(tn)
    except Exception as e:
        errors["time_of_night"] = str(e)
    _emit("time_of_night", 1.0)

    findings["errors"] = errors
    return findings
