"""Sleep architecture report — the standard polysomnography metrics
clinicians expect: REM latency, WASO, fragmentation index, cycle structure.

Builds on the per-epoch labels from sleep_stages.SleepStageResult.

Definitions (standard polysomnography):
- **REM latency**: minutes from sleep onset (first non-W epoch) to first REM epoch.
  Normal in children: 90–180 min. Short REM latency suggests REM-dominant
  sleep architecture or fragmentation.
- **WASO** (Wake After Sleep Onset): total wake minutes between sleep onset
  and the final awakening. Should be low in healthy sleep (<30 min).
- **Sleep fragmentation index**: count of stage transitions per hour. High
  values (>30/h) indicate fragmented, non-restorative sleep.
- **NREM cycle structure**: number and duration of NREM→REM cycles. Healthy
  pediatric sleep has 4–6 cycles per night of ~90 min each.
- **First-NREM-cycle SWS**: minutes of N3 in the first sleep cycle. Most
  slow-wave sleep typically occurs in the first two cycles; if it's
  drastically reduced, memory consolidation is compromised.
"""

from __future__ import annotations

from dataclasses import dataclass

from .sleep_stages import SleepStageResult


@dataclass
class SleepArchitectureResult:
    rem_latency_minutes: float | None
    waso_minutes: float
    fragmentation_index_per_hour: float
    n_complete_cycles: int
    mean_cycle_duration_minutes: float | None
    first_cycle_n3_minutes: float
    total_sleep_time_minutes: float
    sleep_onset_minute: float | None
    final_awakening_minute: float | None


NREM_STAGES = {"N1", "N2", "N3"}


def compute_sleep_architecture(
    sleep_stages: SleepStageResult,
) -> SleepArchitectureResult:
    """Compute standard PSG architecture metrics from per-epoch labels."""
    labels = sleep_stages.epoch_labels
    epoch_min = sleep_stages.epoch_seconds / 60.0

    # Sleep onset = first non-W epoch
    sleep_onset_idx = None
    for i, lab in enumerate(labels):
        if lab != "W":
            sleep_onset_idx = i
            break

    if sleep_onset_idx is None:
        # All wake — no sleep
        return SleepArchitectureResult(
            rem_latency_minutes=None,
            waso_minutes=0.0,
            fragmentation_index_per_hour=0.0,
            n_complete_cycles=0,
            mean_cycle_duration_minutes=None,
            first_cycle_n3_minutes=0.0,
            total_sleep_time_minutes=0.0,
            sleep_onset_minute=None,
            final_awakening_minute=None,
        )

    # Final awakening = index AFTER last non-W epoch
    final_awakening_idx = len(labels)
    for i in range(len(labels) - 1, sleep_onset_idx - 1, -1):
        if labels[i] != "W":
            final_awakening_idx = i + 1
            break

    sleep_period = labels[sleep_onset_idx:final_awakening_idx]
    tst = sum(1 for lab in sleep_period if lab != "W") * epoch_min

    # WASO = wake epochs between sleep onset and final awakening
    waso = sum(1 for lab in sleep_period if lab == "W") * epoch_min

    # REM latency = minutes from sleep onset to first REM
    rem_latency = None
    for i, lab in enumerate(sleep_period):
        if lab == "REM":
            rem_latency = i * epoch_min
            break

    # Fragmentation index = stage transitions per hour
    transitions = sum(
        1 for i in range(1, len(sleep_period)) if sleep_period[i] != sleep_period[i - 1]
    )
    duration_hours = len(sleep_period) * epoch_min / 60.0
    frag_idx = transitions / duration_hours if duration_hours > 0 else 0.0

    # NREM cycles: a cycle = sustained NREM (>= 15 min) followed by REM (>= 5 min).
    # Walk through the sleep period; only count a cycle when we cross from a
    # qualified NREM block into a qualified REM block, then skip past that
    # REM block before considering the next cycle (prevents counting micro-
    # transitions inside one cycle as multiple cycles).
    cycles_complete = 0
    cycle_starts: list[int] = []
    i = 0
    nrem_block_min = 0.0
    in_nrem_block_start: int | None = None
    while i < len(sleep_period):
        lab = sleep_period[i]
        if lab in NREM_STAGES:
            if in_nrem_block_start is None:
                in_nrem_block_start = i
            nrem_block_min += epoch_min
            i += 1
        elif lab == "REM":
            if in_nrem_block_start is not None and nrem_block_min >= 15:
                # Measure REM block length
                j = i
                rem_run_min = 0.0
                while j < len(sleep_period) and sleep_period[j] == "REM":
                    rem_run_min += epoch_min
                    j += 1
                if rem_run_min >= 5:
                    cycles_complete += 1
                    cycle_starts.append(in_nrem_block_start)
                # Skip past the REM block whether or not it qualified, and
                # restart NREM-block tracking from the post-REM index.
                i = j
                in_nrem_block_start = None
                nrem_block_min = 0.0
            else:
                # REM without a qualified preceding NREM block — skip ahead
                while i < len(sleep_period) and sleep_period[i] == "REM":
                    i += 1
                in_nrem_block_start = None
                nrem_block_min = 0.0
        else:
            # Wake or unknown: do not break NREM block but advance
            i += 1

    mean_cycle_min = None
    if cycles_complete >= 2:
        diffs = [
            (cycle_starts[i + 1] - cycle_starts[i]) * epoch_min
            for i in range(len(cycle_starts) - 1)
        ]
        if diffs:
            mean_cycle_min = sum(diffs) / len(diffs)

    # First-cycle N3 minutes: N3 between sleep_onset_idx and first REM
    first_cycle_n3_min = 0.0
    for lab in sleep_period:
        if lab == "REM":
            break
        if lab == "N3":
            first_cycle_n3_min += epoch_min

    return SleepArchitectureResult(
        rem_latency_minutes=float(rem_latency) if rem_latency is not None else None,
        waso_minutes=float(waso),
        fragmentation_index_per_hour=float(frag_idx),
        n_complete_cycles=int(cycles_complete),
        mean_cycle_duration_minutes=(
            float(mean_cycle_min) if mean_cycle_min is not None else None
        ),
        first_cycle_n3_minutes=float(first_cycle_n3_min),
        total_sleep_time_minutes=float(tst),
        sleep_onset_minute=float(sleep_onset_idx * epoch_min),
        final_awakening_minute=float(final_awakening_idx * epoch_min),
    )


def summarize_sleep_architecture(r: SleepArchitectureResult) -> dict:
    return {
        "rem_latency_minutes": (
            round(r.rem_latency_minutes, 1) if r.rem_latency_minutes is not None else None
        ),
        "waso_minutes": round(r.waso_minutes, 1),
        "fragmentation_index_per_hour": round(r.fragmentation_index_per_hour, 1),
        "n_complete_cycles": r.n_complete_cycles,
        "mean_cycle_duration_minutes": (
            round(r.mean_cycle_duration_minutes, 1)
            if r.mean_cycle_duration_minutes is not None else None
        ),
        "first_cycle_n3_minutes": round(r.first_cycle_n3_minutes, 1),
        "total_sleep_time_minutes": round(r.total_sleep_time_minutes, 1),
        "sleep_onset_minute": (
            round(r.sleep_onset_minute, 1) if r.sleep_onset_minute is not None else None
        ),
        "final_awakening_minute": (
            round(r.final_awakening_minute, 1)
            if r.final_awakening_minute is not None else None
        ),
    }
