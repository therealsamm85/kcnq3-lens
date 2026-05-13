"""Pre/post-treatment comparison: diff two findings dicts.

The output is a structured comparison that can be displayed in the UI and
sent to an LLM for clinical interpretation of what changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ─── Directionality: which way is "better"? ─────────────────────────────────
# For each metric, store the direction that means improvement so we can label
# changes as ↓ improved / ↑ worsened correctly.
BETTER_DIRECTION = {
    "topo_max_kurtosis": "lower",          # fewer/weaker spikes = better
    "topo_mean_top5": "lower",
    "spindle_density": "higher",           # more spindles = better
    "pdr_hz": "higher",                    # faster posterior rhythm = better
    "delta_alpha_ratio": "lower",          # less slowing = better
    "n_bursts": "lower",
    "n_bursts_10s_or_longer": "lower",
    "max_burst_duration_s": "lower",
    "pct_complex_spike_wave": "lower",     # less CSWS-like = better
    "events_per_minute": "lower",
}


@dataclass
class MetricDelta:
    name: str                  # human-readable label
    key: str                   # machine key (matches BETTER_DIRECTION)
    pre_value: float
    post_value: float
    absolute_change: float
    pct_change: float | None   # None if pre_value is 0
    direction: str             # "improved", "worsened", "unchanged"


def _classify(key: str, pre: float, post: float, threshold_pct: float = 5.0) -> str:
    """Classify a change as improved / worsened / unchanged.

    A change is "unchanged" if absolute pct change < threshold_pct.
    """
    if pre == 0 and post == 0:
        return "unchanged"
    if pre == 0:
        # Can't compute pct change — fall back to absolute
        return "improved" if post < pre else "worsened" if post > pre else "unchanged"
    pct = abs((post - pre) / pre * 100)
    if pct < threshold_pct:
        return "unchanged"
    better = BETTER_DIRECTION.get(key, "lower")
    if better == "lower":
        return "improved" if post < pre else "worsened"
    return "improved" if post > pre else "worsened"


def _metric(name: str, key: str, pre: float, post: float) -> MetricDelta:
    pct = None if pre == 0 else (post - pre) / pre * 100
    return MetricDelta(
        name=name,
        key=key,
        pre_value=pre,
        post_value=post,
        absolute_change=post - pre,
        pct_change=pct,
        direction=_classify(key, pre, post),
    )


def compare_findings(pre: dict[str, Any], post: dict[str, Any]) -> dict[str, Any]:
    """Compute structured deltas between two findings dicts.

    Each `findings` dict is what `runner.run_all_analyses` produces.
    """
    deltas: list[MetricDelta] = []

    # --- Topography ---
    pre_topo = pre.get("topography", {})
    post_topo = post.get("topography", {})
    if pre_topo and post_topo:
        # Max kurtosis across all channels
        pre_max = max((c["median"] for c in pre_topo.get("all_channels", [])), default=0)
        post_max = max((c["median"] for c in post_topo.get("all_channels", [])), default=0)
        deltas.append(_metric("Max channel kurtosis", "topo_max_kurtosis", pre_max, post_max))

        # Mean of top 5
        pre_top5 = sorted([c["median"] for c in pre_topo.get("all_channels", [])], reverse=True)[:5]
        post_top5 = sorted([c["median"] for c in post_topo.get("all_channels", [])], reverse=True)[:5]
        if pre_top5 and post_top5:
            deltas.append(_metric(
                "Top-5 channels mean kurtosis", "topo_mean_top5",
                sum(pre_top5) / len(pre_top5),
                sum(post_top5) / len(post_top5),
            ))

    # --- Spindles ---
    pre_sp = pre.get("spindles", {})
    post_sp = post.get("spindles", {})
    if pre_sp and post_sp:
        deltas.append(_metric(
            "Spindle density (/min)", "spindle_density",
            pre_sp.get("density_per_minute", 0),
            post_sp.get("density_per_minute", 0),
        ))

    # --- Background ---
    pre_bg = pre.get("background", {})
    post_bg = post.get("background", {})
    if pre_bg and post_bg:
        deltas.append(_metric(
            "Posterior dominant rhythm (Hz)", "pdr_hz",
            pre_bg.get("posterior_dominant_rhythm_hz", 0),
            post_bg.get("posterior_dominant_rhythm_hz", 0),
        ))
        deltas.append(_metric(
            "Delta / Alpha ratio", "delta_alpha_ratio",
            pre_bg.get("delta_alpha_ratio", 0),
            post_bg.get("delta_alpha_ratio", 0),
        ))

    # --- Bursts ---
    pre_br = pre.get("bursts", {})
    post_br = post.get("bursts", {})
    if pre_br and post_br:
        deltas.append(_metric(
            "Bursts ≥3s (count)", "n_bursts",
            pre_br.get("n_bursts", 0),
            post_br.get("n_bursts", 0),
        ))
        deltas.append(_metric(
            "Bursts ≥10s (count)", "n_bursts_10s_or_longer",
            pre_br.get("n_bursts_10s_or_longer", 0),
            post_br.get("n_bursts_10s_or_longer", 0),
        ))
        deltas.append(_metric(
            "Longest burst (s)", "max_burst_duration_s",
            pre_br.get("max_duration_s", 0),
            post_br.get("max_duration_s", 0),
        ))

    # --- Morphology ---
    pre_m = pre.get("morphology", {})
    post_m = post.get("morphology", {})
    if pre_m and post_m:
        deltas.append(_metric(
            "% complex spike-wave", "pct_complex_spike_wave",
            pre_m.get("pct_complex_spike_wave", 0),
            post_m.get("pct_complex_spike_wave", 0),
        ))
        deltas.append(_metric(
            "Events per minute", "events_per_minute",
            pre_m.get("events_per_minute", 0),
            post_m.get("events_per_minute", 0),
        ))

    # Overall direction tally
    improved = sum(1 for d in deltas if d.direction == "improved")
    worsened = sum(1 for d in deltas if d.direction == "worsened")
    unchanged = sum(1 for d in deltas if d.direction == "unchanged")

    return {
        "deltas": [_delta_to_dict(d) for d in deltas],
        "overall": {
            "n_improved": improved,
            "n_worsened": worsened,
            "n_unchanged": unchanged,
            "verdict": _overall_verdict(improved, worsened, unchanged),
        },
        "pre_findings": pre,
        "post_findings": post,
    }


def _delta_to_dict(d: MetricDelta) -> dict:
    return {
        "name": d.name,
        "key": d.key,
        "pre_value": round(d.pre_value, 3),
        "post_value": round(d.post_value, 3),
        "absolute_change": round(d.absolute_change, 3),
        "pct_change": round(d.pct_change, 1) if d.pct_change is not None else None,
        "direction": d.direction,
    }


def _overall_verdict(improved: int, worsened: int, unchanged: int) -> str:
    total = improved + worsened + unchanged
    if total == 0:
        return "no_data"
    if improved > worsened * 2:
        return "clearly_improved"
    if improved > worsened:
        return "mixed_mostly_improved"
    if worsened > improved * 2:
        return "clearly_worsened"
    if worsened > improved:
        return "mixed_mostly_worsened"
    return "mixed_neutral"
