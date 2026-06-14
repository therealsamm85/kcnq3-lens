"""A3 — Graph-theory network metrics on the wPLI matrices.  [BUILD on numpy]

The connectivity module already produces debiased wPLI matrices per band but
stops at the raw matrices. ESES/CSWS alters network topology (global efficiency,
clustering, small-worldness), so deriving graph metrics from matrices we already
have is high-leverage and pure-local.

BUILD: weighted clustering (Onnela 2005), characteristic path length and
global/local efficiency (Rubinov & Sporns 2010 weighted definitions on a
length = 1/weight transform), and a small-world index relative to a
weight-permuted random null — all standard formulas in numpy (networkx not
required). Descriptive only: no normative pediatric graph cohort exists, so
values are for intra-patient comparison over time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Random nulls for the small-world index (weight-preserving topology shuffles).
_N_NULL = 10


@dataclass
class GraphMetricsResult:
    bands: list[str]
    n_nodes: int
    per_band: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _as_matrix(m) -> np.ndarray:
    W = np.asarray(m, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("connectivity matrix must be square")
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    W = np.clip(W, 0.0, None)          # wPLI ≥ 0
    np.fill_diagonal(W, 0.0)
    return (W + W.T) / 2.0              # enforce symmetry


def _strength(W: np.ndarray) -> float:
    n = W.shape[0]
    return float(W.sum() / n) if n else 0.0


def _weighted_clustering(W: np.ndarray) -> float:
    """Mean Onnela weighted clustering coefficient."""
    n = W.shape[0]
    wmax = W.max()
    if n < 3 or wmax <= 0:
        return 0.0
    A = (W > 0).astype(float)
    k = A.sum(axis=1)                  # degree
    Wc = (W / wmax) ** (1.0 / 3.0)     # cube-root of normalized weights
    cyc = np.diag(Wc @ Wc @ Wc)        # sum of (w_ij w_ih w_jh)^(1/3) triangles
    denom = k * (k - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ci = np.where(denom > 0, cyc / denom, 0.0)
    return float(np.mean(ci))


def _distance_matrix(W: np.ndarray) -> np.ndarray:
    """Shortest-path distances using length = 1/weight (Floyd-Warshall)."""
    n = W.shape[0]
    with np.errstate(divide="ignore"):
        D = np.where(W > 0, 1.0 / W, np.inf)
    np.fill_diagonal(D, 0.0)
    for k in range(n):
        D = np.minimum(D, D[:, k][:, None] + D[k, :][None, :])
    return D


def _char_path_length(D: np.ndarray) -> float:
    n = D.shape[0]
    off = ~np.eye(n, dtype=bool)
    finite = D[off & np.isfinite(D)]
    return float(np.mean(finite)) if finite.size else float("inf")


def _global_efficiency(W: np.ndarray) -> float:
    n = W.shape[0]
    if n < 2:
        return 0.0
    D = _distance_matrix(W)
    off = ~np.eye(n, dtype=bool)
    with np.errstate(divide="ignore"):
        inv = np.where(np.isfinite(D) & (D > 0), 1.0 / D, 0.0)
    return float(inv[off].sum() / (n * (n - 1)))


def _local_efficiency(W: np.ndarray) -> float:
    """Mean over nodes of the global efficiency of each node's neighborhood."""
    n = W.shape[0]
    effs = []
    for i in range(n):
        nbrs = np.where(W[i] > 0)[0]
        if nbrs.size < 2:
            effs.append(0.0)
            continue
        effs.append(_global_efficiency(W[np.ix_(nbrs, nbrs)]))
    return float(np.mean(effs)) if effs else 0.0


def _permuted_null(W: np.ndarray, rng) -> np.ndarray:
    """Shuffle the upper-triangle edge weights among node pairs (keeps the weight
    distribution, destroys topology) — a transparent small-world null."""
    n = W.shape[0]
    iu = np.triu_indices(n, k=1)
    w = W[iu].copy()
    rng.shuffle(w)
    R = np.zeros_like(W)
    R[iu] = w
    return R + R.T


def _band_metrics(W: np.ndarray) -> dict[str, float]:
    C = _weighted_clustering(W)
    D = _distance_matrix(W)
    L = _char_path_length(D)
    metrics = {
        "strength": round(_strength(W), 4),
        "clustering": round(C, 4),
        "char_path_length": (round(L, 4) if np.isfinite(L) else None),
        "global_efficiency": round(_global_efficiency(W), 4),
        "local_efficiency": round(_local_efficiency(W), 4),
    }
    # Small-world sigma = (C/Cr) / (L/Lr) vs a weight-permuted null.
    rng = np.random.default_rng(0)
    cr, lr = [], []
    for _ in range(_N_NULL):
        R = _permuted_null(W, rng)
        cr.append(_weighted_clustering(R))
        lr.append(_char_path_length(_distance_matrix(R)))
    cr_m = float(np.mean(cr)) if cr else 0.0
    lr_f = [x for x in lr if np.isfinite(x)]
    lr_m = float(np.mean(lr_f)) if lr_f else float("inf")
    if cr_m > 0 and np.isfinite(L) and np.isfinite(lr_m) and L > 0:
        gamma = C / cr_m
        lam = L / lr_m
        metrics["small_world_sigma"] = round(gamma / lam, 4) if lam > 0 else None
    else:
        metrics["small_world_sigma"] = None
    return metrics


def compute_graph_metrics(
    matrices_by_band: dict,
    channels: list[str] | None = None,
) -> GraphMetricsResult:
    """Graph metrics per band from wPLI matrices (list-of-lists or ndarray)."""
    bands = list(matrices_by_band.keys())
    per_band: dict[str, dict[str, float]] = {}
    n_nodes = 0
    sizes: set[int] = set()
    notes: list[str] = []
    for b in bands:
        try:
            W = _as_matrix(matrices_by_band[b])
        except ValueError as e:
            notes.append(f"{b}: skipped ({e})")
            continue
        n_nodes = W.shape[0]
        sizes.add(n_nodes)
        if n_nodes < 3:
            notes.append(f"{b}: <3 nodes — graph metrics unstable")
        # Flag disconnection: char_path_length then averages reachable pairs only,
        # underestimating it (and inflating small-world sigma) — say so.
        if n_nodes >= 2:
            D = _distance_matrix(W)
            off = ~np.eye(n_nodes, dtype=bool)
            if np.isinf(D[off]).any():
                notes.append(f"{b}: graph disconnected — char_path_length covers "
                             "reachable pairs only (small-world sigma inflated).")
        per_band[b] = _band_metrics(W)
    if not per_band:
        notes.append("no usable connectivity matrices")
    if len(sizes) > 1:
        notes.append(f"bands have differing node counts {sorted(sizes)} — n_nodes "
                     "reflects the last band; metrics remain per-band.")
    notes.append("descriptive, intra-patient only — no normative pediatric graph cohort.")
    return GraphMetricsResult(bands=list(per_band.keys()), n_nodes=n_nodes,
                              per_band=per_band, notes=notes)


def summarize_graph_metrics(result: GraphMetricsResult) -> dict:
    return {
        "n_nodes": result.n_nodes,
        "per_band": result.per_band,
        "notes": result.notes,
    }
