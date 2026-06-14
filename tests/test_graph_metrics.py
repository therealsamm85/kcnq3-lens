"""Tests for A3 — graph-theory metrics on wPLI matrices (analyses/graph_metrics.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyses.graph_metrics import (
    compute_graph_metrics, summarize_graph_metrics,
    _weighted_clustering, _char_path_length, _distance_matrix,
    _global_efficiency, _as_matrix,
)

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
n_pass = n_fail = 0


def check(name, cond, detail=""):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f"  {PASS} {name}")
    else:
        n_fail += 1
        print(f"  {FAIL} {name}  {detail}")


def complete(n, w=1.0):
    W = np.full((n, n), w)
    np.fill_diagonal(W, 0.0)
    return W


def two_cliques(size=4, cross=0.05):
    n = 2 * size
    W = np.zeros((n, n))
    for a, b in ((0, size), (size, n)):
        W[a:b, a:b] = 1.0
    np.fill_diagonal(W, 0.0)
    W[0, n - 1] = W[n - 1, 0] = cross   # one weak bridge
    return W


def small_world(n=20, k=4, shortcuts=8, seed=1, w=1.0):
    """Watts-Strogatz-style: a ring lattice (high clustering) + a few random
    shortcuts (short paths) → genuinely small-world (sigma > 1)."""
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(1, k // 2 + 1):
            W[i, (i + j) % n] = w
            W[(i + j) % n, i] = w
    rng = np.random.default_rng(seed)
    for _ in range(shortcuts):
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a != b:
            W[a, b] = W[b, a] = w
    return W


# ── estimators on known graphs ─────────────────────────────────────────────
print("\n── A3: graph estimators on known structures ───────────────────────")

K6 = complete(6, 1.0)
check("complete graph clustering ≈ 1", abs(_weighted_clustering(K6) - 1.0) < 1e-6,
      f"got {_weighted_clustering(K6)}")
check("complete graph global efficiency ≈ 1",
      abs(_global_efficiency(K6) - 1.0) < 1e-6, f"got {_global_efficiency(K6)}")
check("complete graph char path length ≈ 1 (unit weights)",
      abs(_char_path_length(_distance_matrix(K6)) - 1.0) < 1e-6)

# Distance = 1/weight: stronger weights → shorter paths.
strong = _char_path_length(_distance_matrix(complete(6, 1.0)))
weak = _char_path_length(_distance_matrix(complete(6, 0.5)))
check("stronger weights → shorter char path length", weak > strong,
      f"weak={weak} strong={strong}")
check("char path length scales as 1/weight (0.5 → ~2.0)", abs(weak - 2.0) < 1e-6,
      f"got {weak}")

# Modular graph is more clustered than a random null of the same weights.
mod = two_cliques()
cr = np.mean([_weighted_clustering(
    _as_matrix(np.random.default_rng(s).permutation(mod.flatten()).reshape(mod.shape)))
    for s in range(5)])
check("modular graph clustering > random-null clustering",
      _weighted_clustering(mod) > cr, f"mod={_weighted_clustering(mod):.3f} null≈{cr:.3f}")

# Empty graph: no crash, zeroed/None metrics.
empty = np.zeros((6, 6))
check("empty graph clustering 0", _weighted_clustering(empty) == 0.0)
check("empty graph char path length is inf", not np.isfinite(_char_path_length(_distance_matrix(empty))))


# ── full pipeline over bands ───────────────────────────────────────────────
print("\n── A3: compute_graph_metrics pipeline ─────────────────────────────")

matrices = {
    "alpha": complete(8, 0.8).tolist(),
    "delta": small_world().tolist(),
}
res = compute_graph_metrics(matrices, channels=[f"C{i}" for i in range(8)])
keys = {"strength", "clustering", "char_path_length", "global_efficiency",
        "local_efficiency", "small_world_sigma"}
check("per-band metrics computed for both bands", set(res.per_band) == {"alpha", "delta"})
check("each band has all 6 metric keys", set(res.per_band["alpha"]) == keys,
      f"got {set(res.per_band['alpha'])}")
check("Watts-Strogatz graph is small-world (sigma > 1)",
      (res.per_band["delta"]["small_world_sigma"] or 0) > 1.0,
      f"got {res.per_band['delta']['small_world_sigma']}")
check("complete graph sigma ≈ 1 (it is its own random null)",
      abs((res.per_band["alpha"]["small_world_sigma"] or 0) - 1.0) < 0.1,
      f"got {res.per_band['alpha']['small_world_sigma']}")
import json
check("summary JSON-serializable", isinstance(json.dumps(summarize_graph_metrics(res)), str))

# Degenerate: non-square matrix is skipped with a note, not a crash.
res_bad = compute_graph_metrics({"x": [[0, 1, 0], [1, 0, 0]]})
check("non-square matrix skipped with note",
      "x" not in res_bad.per_band and any("skipped" in n for n in res_bad.notes))


# ── real integration with compute_connectivity ─────────────────────────────
print("\n── A3: integration with compute_connectivity ──────────────────────")
from src.readers.base import EEGRecording
from src.analyses.connectivity import compute_connectivity

sf = 100.0
n = int(120 * sf)
rng = np.random.RandomState(0)
# 4 channels: 0&1 phase-lagged coupled at 10 Hz; 2&3 independent noise.
t = np.arange(n) / sf
base = np.sin(2 * np.pi * 10 * t)
data = np.vstack([
    30 * base + 5 * rng.randn(n),
    30 * np.sin(2 * np.pi * 10 * t + 0.6) + 5 * rng.randn(n),
    20 * rng.randn(n),
    20 * rng.randn(n),
]).astype(np.float32)
rec = EEGRecording(path=Path("/tmp/c.eeg"), sfreq=sf, n_channels=4,
                   duration_s=n / sf, channel_names=["A", "B", "C", "D"],
                   n_channels_in_file=4, eeg_channel_indices=[0, 1, 2, 3],
                   format_name="synthetic")
rec._full_data = data
conn = compute_connectivity(rec)
gm = compute_graph_metrics(conn.matrices_by_band, channels=conn.channels)
check("graph metrics computed from real connectivity output",
      "alpha" in gm.per_band and gm.n_nodes == 4)
check("metrics are finite/None (no NaN leakage)",
      all(v is None or np.isfinite(v) for v in gm.per_band["alpha"].values()))


print(f"\n{'='*60}\n  PASS: {n_pass}\n  FAIL: {n_fail}\n{'='*60}")
if n_fail:
    sys.exit(1)
