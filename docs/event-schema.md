# Event Schema — Absolute Recording Time Contract

## Overview

All `_*_events` lists stored in the `findings` dict by `run_all_analyses()`
use **absolute recording time**: seconds elapsed from the start of the EEG
recording file, not channel-local or epoch-local time.

## Affected fields

| findings key | module | time field(s) |
|---|---|---|
| `_spindle_events` | `analyses/spindles.py` | `time_s` |
| `_slow_waves_events` | `analyses/slow_waves.py` | `start_s`, `neg_peak_s`, `zero_cross_s`, `end_s` |
| `_morphology_events` | `analyses/morphology.py` | `time_s` |
| `_hfo_ripples_events` | `analyses/hfo_ripples.py` | `start_s`, `peak_time_s`, `end_s` |
| `_ied_events` | `analyses/ied_ml.py` | `time_s` |

## Contract

All `time_s` / `peak_time_s` / `neg_peak_s` / `start_s` / `end_s` values in
these `events` lists refer to **absolute recording time** (seconds from
recording start).

Cross-module consumers (coupling, hfo_ripples co-occurrence with morphology,
ied_ml) depend on this contract. **Do not change the time reference frame of
any individual module without coordinating all consumers.**

## Private keys

Keys prefixed with `_` (underscore) are internal to the pipeline. The registry
submission builder uses an explicit allowlist and will not export these fields.
The `safe_round_dict` sanitization pass also excludes them to preserve
full float precision for downstream cross-module computation.

## See also

- `src/runner.py` — block comment at the top of the file
- `src/analyses/spindles.py`, `slow_waves.py`, `hfo_ripples.py`, `coupling.py`,
  `ied_ml.py`, `morphology.py` — individual module docstrings
