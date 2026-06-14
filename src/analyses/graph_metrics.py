"""A3 — Graph-theory network metrics on the wPLI matrices.  [BUILD on numpy]

The connectivity module already produces debiased wPLI matrices per band but
stops at the raw matrices. ESES/CSWS alters network topology (global efficiency,
clustering, small-worldness), so deriving graph metrics from matrices we already
have is high-leverage and pure-local.

BUILD: clustering coefficient, characteristic path length, global/local
efficiency and a small-world index are standard formulas on a weighted
adjacency matrix — numpy only (networkx optional, not required).

SCAFFOLD — implemented in wave A3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GraphMetricsResult:
    bands: list[str]
    per_band: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def compute_graph_metrics(
    matrices_by_band: dict[str, list[list[float]]] | dict[str, np.ndarray],
    channels: list[str] | None = None,
) -> GraphMetricsResult:
    """Graph metrics per band from wPLI matrices. SCAFFOLD — wave A3."""
    raise NotImplementedError("scaffold — implemented in wave A3")


def summarize_graph_metrics(result: GraphMetricsResult) -> dict:
    raise NotImplementedError("scaffold — implemented in wave A3")
