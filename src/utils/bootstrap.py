"""Bootstrap confidence intervals for clinical numbers.

Spike counts, burst counts, and rate metrics are point estimates — but the
underlying detection step has stochastic noise (especially under different
threshold choices). A doctor reading "104 bursts ≥10s" deserves to know
whether that's "100 ± 10" or "100 ± 50."

This module provides a simple percentile-bootstrap helper. Given a
detection function and the underlying data, it resamples the data with
replacement N times, runs detection on each, and returns the 95% CI of
the count.

For analyses that already store the per-epoch breakdown (morphology /
time-of-night / bursts), bootstrap is over EPOCHS, not raw samples — that
preserves the spike-wave structure within each epoch.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CIResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    n_bootstrap: int
    confidence_level: float


def bootstrap_count_ci(
    per_epoch_values: list[float] | np.ndarray,
    *,
    aggregate: str = "mean",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> CIResult:
    """Compute a confidence interval for an aggregate (mean/sum/median).

    Parameters
    ----------
    per_epoch_values : array-like
        One value per epoch (e.g. spike count per epoch, burst seconds per
        epoch).
    aggregate : str
        How to aggregate within a resample: 'mean', 'sum', 'median'.
    n_bootstrap : int
        Number of resamples. 1000 is plenty for 95% CI; faster computation
        for many calls.
    confidence_level : float
        E.g. 0.95 for 95% CI.
    seed : int
        Reproducibility.
    """
    arr = np.asarray(per_epoch_values, dtype=np.float64)
    if arr.size == 0:
        return CIResult(0.0, 0.0, 0.0, 0, confidence_level)

    rng = np.random.default_rng(seed)
    n = arr.size
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        resample = arr[idx]
        if aggregate == "mean":
            samples[i] = resample.mean()
        elif aggregate == "sum":
            samples[i] = resample.sum()
        elif aggregate == "median":
            samples[i] = np.median(resample)
        else:
            raise ValueError(f"Unknown aggregate: {aggregate}")

    alpha = 1 - confidence_level
    low = np.quantile(samples, alpha / 2)
    high = np.quantile(samples, 1 - alpha / 2)

    if aggregate == "mean":
        point = float(arr.mean())
    elif aggregate == "sum":
        point = float(arr.sum())
    else:
        point = float(np.median(arr))

    return CIResult(
        point_estimate=point,
        ci_low=float(low),
        ci_high=float(high),
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )


def format_ci(ci: CIResult, ndigits: int = 1, unit: str = "") -> str:
    """Render as e.g. '104 (95% CI 89–119)' or '19.5 /min (95% CI 17.2–22.0)'."""
    suffix = f" {unit}" if unit else ""
    pct = int(ci.confidence_level * 100)
    return (
        f"{ci.point_estimate:.{ndigits}f}{suffix} "
        f"({pct}% CI {ci.ci_low:.{ndigits}f}–{ci.ci_high:.{ndigits}f})"
    )
