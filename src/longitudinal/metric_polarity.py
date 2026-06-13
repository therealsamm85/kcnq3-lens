"""Clinical polarity of each longitudinal biomarker — which direction is "better".

Single source of truth shared by the treatment-response dashboard and the
word-tracker correlation, so the two features can never disagree on what a
falling spike rate or a rising PDR *means*.

Polarity is a clinical-interpretation call, encoded conservatively:

    +1  higher is better   (more mature / more restorative)
    -1  lower  is better   (less pathological)
     0  ambiguous          (no monotonic "better" — report the delta only)

For KCNQ3 gain-of-function with an ESES-spectrum picture, the load-bearing
biomarkers are the spike burden (lower better) and the maturation markers
(PDR, spindles — higher better). Where a metric has a real but maturation-
confounded reading (PDR rises with age regardless of treatment), the polarity
still holds but callers should surface the age confound — see CONFOUNDED.
"""

from __future__ import annotations

# metric label (as used by trends.METRICS) → polarity
METRIC_POLARITY: dict[str, int] = {
    "spike_rate_per_min": -1,        # interictal spikes degrade learning
    "spindle_density_per_min": +1,   # spindles drive memory consolidation
    "pdr_hz": +1,                    # faster posterior rhythm = maturation
    "delta_alpha_ratio": -1,         # high delta/alpha = slowing / dysrhythmia
    "swi_n3_pct": -1,                # spike-wave index is pathological
    "swi_nrem_combined_pct": -1,
    "activation_factor": -1,         # sleep activation of spikes is the ESES hallmark
    "nrem_rate_per_min": -1,
    "wake_rate_per_min": -1,
    "bursts_10s_count": -1,          # sustained bursts are pathological
    "sleep_efficiency_pct": +1,
    "rem_latency_minutes": 0,        # both too-short and too-long are abnormal
    "first_cycle_n3_minutes": +1,    # deep slow-wave sleep is restorative (low conf.)
    "fragmentation_index": -1,       # fragmented sleep is worse
}

# Metrics whose value also moves with normal maturation, so an improvement
# across recordings months apart cannot be cleanly attributed to treatment.
CONFOUNDED_BY_MATURATION: frozenset[str] = frozenset({
    "pdr_hz", "spindle_density_per_min", "first_cycle_n3_minutes",
})


def polarity_of(metric: str) -> int:
    """Return +1 / -1 / 0 for a metric, defaulting to 0 (ambiguous) if unknown."""
    return METRIC_POLARITY.get(metric, 0)


def direction_label(metric: str, delta: float, tol: float = 0.0) -> str:
    """Map a raw change (follow-up − baseline) to a clinical direction.

    Returns one of: "improved" | "worsened" | "no_clear_change" | "ambiguous".
    ``ambiguous`` is returned when the metric has no defined "better" direction
    (polarity 0) — the magnitude is still meaningful, the sign is not.
    ``tol`` is an absolute dead-band: |delta| ≤ tol → "no_clear_change".
    """
    pol = polarity_of(metric)
    if pol == 0:
        return "ambiguous"
    if abs(delta) <= tol:
        return "no_clear_change"
    improved = (delta > 0 and pol > 0) or (delta < 0 and pol < 0)
    return "improved" if improved else "worsened"
