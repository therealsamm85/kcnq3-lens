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
    assess_quality,
    compute_sleep_stages,
    compute_swi,
    compute_state_split,
    compute_synchrony,
    compute_sleep_architecture,
    compute_slow_waves,
)
from .analyses.topography import summarize_topography
from .analyses.spindles import summarize_spindles
from .analyses.background import summarize_background
from .analyses.bursts import summarize_bursts
from .analyses.morphology import summarize_morphology
from .analyses.time_of_night import summarize_time_of_night
from .analyses.quality import summarize_quality
from .analyses.sleep_stages import summarize_sleep_stages
from .analyses.swi import summarize_swi
from .analyses.state_split import summarize_state_split
from .analyses.synchrony import summarize_synchrony
from .analyses.sleep_architecture import summarize_sleep_architecture
from .analyses.slow_waves import summarize_slow_waves
from .clinical.impression import build_impression, build_recommendations
from .clinical.negative_findings import build_negative_findings
from .utils.sanitize import safe_round_dict


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

    # --- 0. Quality control (runs first, fast) ---
    _emit("quality", 0.02)
    try:
        qc = assess_quality(
            rec, start_epoch=sleep_start_epoch, end_epoch=sleep_end_epoch
        )
        findings["quality"] = summarize_quality(qc)
    except Exception as e:
        errors["quality"] = str(e)

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
    _emit("time_of_night", 0.92)

    # --- 7. Sleep stages (v0.5) ---
    sleep_stage_result = None
    try:
        sleep_stage_result = compute_sleep_stages(rec, age_years=age_years)
        findings["sleep_stages"] = summarize_sleep_stages(sleep_stage_result)
    except Exception as e:
        errors["sleep_stages"] = str(e)
    _emit("sleep_stages", 0.95)

    # --- 8. Formal SWI per stage (v0.5) — depends on sleep_stages ---
    if sleep_stage_result is not None:
        try:
            swi = compute_swi(rec, sleep_stage_result)
            findings["swi"] = summarize_swi(swi)
        except Exception as e:
            errors["swi"] = str(e)
    _emit("swi", 0.97)

    # --- 9. State split — wake vs sleep spike rate (v0.5) ---
    if sleep_stage_result is not None:
        try:
            ss = compute_state_split(rec, sleep_stage_result)
            findings["state_split"] = summarize_state_split(ss)
        except Exception as e:
            errors["state_split"] = str(e)
    _emit("state_split", 0.98)

    # --- 10. Synchrony / spread (v0.5) ---
    try:
        syn = compute_synchrony(
            rec, start_epoch=sleep_start_epoch, end_epoch=sleep_end_epoch
        )
        findings["synchrony"] = summarize_synchrony(syn)
    except Exception as e:
        errors["synchrony"] = str(e)
    _emit("synchrony", 0.99)

    # --- 11. Sleep architecture (v0.6) — depends on sleep_stages ---
    if sleep_stage_result is not None:
        try:
            arch = compute_sleep_architecture(sleep_stage_result)
            findings["sleep_architecture"] = summarize_sleep_architecture(arch)
        except Exception as e:
            errors["sleep_architecture"] = str(e)

    # --- 11b. Slow-wave detection (v0.13.0) — Tier 2 ---
    try:
        sw = compute_slow_waves(
            rec,
            sleep_stages=sleep_stage_result,
            channel="Fz",
            age_years=age_years,
        )
        findings["slow_waves"] = summarize_slow_waves(sw)
    except Exception as e:
        errors["slow_waves"] = str(e)

    # --- 12. Clinical impression + recommendations (v0.6) ---
    # Built from all preceding findings; deterministic, no LLM.
    try:
        from .insights import build_narrative
        narrative = build_narrative(findings)
        patterns = narrative.get("patterns", [])
        findings["clinical_impression"] = build_impression(findings, patterns)
        findings["clinical_recommendations"] = build_recommendations(
            findings, patterns
        )
    except Exception as e:
        errors["clinical_impression"] = str(e)

    # --- 13. Negative findings (v0.7) ---
    # What was checked and not found — clinically informative.
    try:
        findings["negative_findings"] = build_negative_findings(findings)
    except Exception as e:
        errors["negative_findings"] = str(e)

    _emit("clinical", 1.0)

    findings["errors"] = errors

    # ── Final sanitization pass ──────────────────────────────────────────
    # Every numeric leaf is guaranteed finite (no NaN/Inf) and a Python type
    # (no numpy scalars). Prevents JSON-serialization failures, garbled PDFs,
    # and LLM confusion when degenerate inputs produce non-finite values.
    findings = safe_round_dict(findings, ndigits=3, default=0.0)
    return findings
