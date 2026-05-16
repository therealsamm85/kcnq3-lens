"""Download + lookup peer-comparison aggregates.

The registry repo publishes `releases/v1/aggregates.json` on every
merge to main. This module is the consumer side: download (with cache),
look up the best-matching cohort cell for a given child, and rank a
local metric value as a percentile within that cell.

Cache
-----
Cached at `~/.kcnq3-lens/aggregates_cache.json` with the fetch
timestamp embedded. Default TTL: 24h. The UI offers "Refresh now"
which forces an HTTP fetch regardless of TTL.

Network failure handling
------------------------
If the fetch fails (no internet, GitHub down, file missing), we fall
back to the cached copy if any, with a clear staleness warning. If no
cache exists either, we return `None` and the UI hides the peer
comparison section — never errors out.

Trust model
-----------
The aggregates file is plain JSON from a public GitHub repo. We
validate its structure on load (schema_version match, expected keys)
and reject anything malformed. There is no executable content.

Percentile estimation
---------------------
The aggregates publish p10, p25, p50, p75, p90 per metric per cell.
Given a child's value, we linearly interpolate within the percentile
grid to estimate the percentile rank. Outside p10/p90 we extrapolate
conservatively, capped at [0, 100].
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request as _urlreq
from urllib.error import URLError, HTTPError

from .upload import DEFAULT_OWNER, DEFAULT_REPO


CACHE_TTL_SECONDS = 24 * 3600
SCHEMA_VERSION = 1

# B3: Hard cap on aggregates payload to prevent memory exhaustion from a
# malicious or misconfigured registry serving a multi-GB JSON response.
_MAX_AGG_BYTES = 5 * 1024 * 1024   # 5 MB


def _cache_path() -> Path:
    base = Path(os.environ.get(
        "KCNQ3_LENS_DATA", str(Path.home() / ".kcnq3-lens")
    ))
    return base / "aggregates_cache.json"


def _aggregates_url(owner: str | None = None, repo: str | None = None) -> str:
    owner = owner or DEFAULT_OWNER
    repo = repo or DEFAULT_REPO
    return (
        f"https://raw.githubusercontent.com/{owner}/{repo}/"
        f"main/releases/v1/aggregates.json"
    )


@dataclass
class AggregatesCache:
    fetched_at: float          # unix epoch
    source_url: str
    payload: dict[str, Any]    # the parsed JSON from the registry


# ─── Validation ────────────────────────────────────────────────────────────

def _validate_aggregates_shape(obj: Any) -> bool:
    """Lightweight structural validation — reject garbage early."""
    if not isinstance(obj, dict):
        return False
    if obj.get("schema_version") != SCHEMA_VERSION:
        return False
    if "cells" not in obj or not isinstance(obj["cells"], list):
        return False
    for c in obj["cells"]:
        if not isinstance(c, dict):
            return False
        if "cell" not in c or "n" not in c:
            return False
        if not isinstance(c.get("stats", {}), dict):
            return False
    return True


# ─── Cache I/O ─────────────────────────────────────────────────────────────

def load_cache() -> AggregatesCache | None:
    """Return the cached aggregates, or None if no cache exists / is
    corrupt. Does not check TTL — that's the caller's call."""
    p = _cache_path()
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        payload = d.get("payload")
        if not _validate_aggregates_shape(payload):
            return None
        return AggregatesCache(
            fetched_at=float(d.get("fetched_at", 0.0)),
            source_url=str(d.get("source_url", "")),
            payload=payload,
        )
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def save_cache(cache: AggregatesCache) -> None:
    p = _cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "fetched_at": cache.fetched_at,
        "source_url": cache.source_url,
        "payload": cache.payload,
    }, indent=2), encoding="utf-8")


# ─── Fetch ─────────────────────────────────────────────────────────────────

def fetch_aggregates(
    *,
    owner: str | None = None,
    repo: str | None = None,
    timeout_s: float = 10.0,
) -> AggregatesCache:
    """HTTP GET the aggregates.json. Raises on failure.

    Caller decides whether to fall back to cache."""
    url = _aggregates_url(owner=owner, repo=repo)
    req = _urlreq.Request(
        url,
        headers={"User-Agent": "kcnq3-lens-aggregates-fetcher"},
    )
    with _urlreq.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read(_MAX_AGG_BYTES + 1).decode("utf-8")
    if len(raw.encode("utf-8")) > _MAX_AGG_BYTES:
        raise ValueError(
            f"Aggregates payload exceeds {_MAX_AGG_BYTES} bytes — "
            f"refusing to load."
        )
    payload = json.loads(raw)
    if not _validate_aggregates_shape(payload):
        raise ValueError(
            f"Fetched aggregates did not match expected schema "
            f"(version {SCHEMA_VERSION})"
        )
    cache = AggregatesCache(
        fetched_at=time.time(), source_url=url, payload=payload,
    )
    save_cache(cache)
    return cache


def get_aggregates(
    *,
    force_refresh: bool = False,
    ttl_s: float = CACHE_TTL_SECONDS,
    owner: str | None = None,
    repo: str | None = None,
) -> tuple[AggregatesCache | None, str | None]:
    """Return (cache, error). If `force_refresh` is True, always hits
    the network. Otherwise honors TTL. On network failure, falls back
    to whatever cache exists. `error` is a human-readable string when
    we fell back to (stale or no) cache, else None.
    """
    cached = load_cache()
    now = time.time()

    if not force_refresh and cached is not None:
        if (now - cached.fetched_at) < ttl_s:
            return cached, None  # cache is fresh

    try:
        fresh = fetch_aggregates(owner=owner, repo=repo)
        return fresh, None
    except (URLError, HTTPError, json.JSONDecodeError,
            ValueError, OSError) as e:
        if cached is not None:
            age_h = (now - cached.fetched_at) / 3600.0
            return cached, (
                f"Network fetch failed ({type(e).__name__}: {e}). "
                f"Using cached copy from {age_h:.1f}h ago."
            )
        return None, (
            f"Network fetch failed ({type(e).__name__}: {e}) and no "
            f"cache exists. Peer comparison unavailable."
        )


# ─── Cell lookup ──────────────────────────────────────────────────────────

def find_best_cell(
    aggregates: dict | AggregatesCache | None,
    *,
    variant_gene: str,
    variant_protein: str | None = None,
    age_years_bucket: str | None = None,
    sex: str | None = None,
) -> dict | None:
    """Walk the cell hierarchy finest → coarsest, return the first
    cell that matches the supplied keys.

    Hierarchy levels:
      1. gene_protein_age_sex
      2. gene_protein_age
      3. gene_protein
      4. gene_age
      5. gene
    """
    if aggregates is None:
        return None
    if isinstance(aggregates, AggregatesCache):
        aggregates = aggregates.payload

    cells = aggregates.get("cells", [])

    levels: list[tuple[str, dict[str, Any]]] = []
    if variant_protein and age_years_bucket and sex:
        levels.append(("gene_protein_age_sex", {
            "variant_gene": variant_gene,
            "variant_protein": variant_protein,
            "age_years_bucket": age_years_bucket,
            "sex": sex,
        }))
    if variant_protein and age_years_bucket:
        levels.append(("gene_protein_age", {
            "variant_gene": variant_gene,
            "variant_protein": variant_protein,
            "age_years_bucket": age_years_bucket,
        }))
    if variant_protein:
        levels.append(("gene_protein", {
            "variant_gene": variant_gene,
            "variant_protein": variant_protein,
        }))
    if age_years_bucket:
        levels.append(("gene_age", {
            "variant_gene": variant_gene,
            "age_years_bucket": age_years_bucket,
        }))
    levels.append(("gene", {"variant_gene": variant_gene}))

    for level, match in levels:
        for c in cells:
            cell_id = c.get("cell", {})
            if cell_id.get("level") != level:
                continue
            if all(cell_id.get(k) == v for k, v in match.items()):
                return c
    return None


# ─── Percentile estimation ─────────────────────────────────────────────────

def percentile_rank(
    value: float,
    stat: dict[str, Any],
) -> float | None:
    """Estimate the percentile rank of `value` within a cell stat block.

    Uses the published p10, p25, p50, p75, p90 (and optionally min/max)
    to linearly interpolate. Returns a float in [0, 100], or None if
    not enough info.
    """
    if not isinstance(stat, dict):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None

    # Anchors: (percentile, value)
    anchors: list[tuple[float, float]] = []
    if "min" in stat:
        anchors.append((0.0, float(stat["min"])))
    if "p10" in stat:
        anchors.append((10.0, float(stat["p10"])))
    if "p25" in stat:
        anchors.append((25.0, float(stat["p25"])))
    if "median" in stat:
        anchors.append((50.0, float(stat["median"])))
    if "p75" in stat:
        anchors.append((75.0, float(stat["p75"])))
    if "p90" in stat:
        anchors.append((90.0, float(stat["p90"])))
    if "max" in stat:
        anchors.append((100.0, float(stat["max"])))

    if len(anchors) < 2:
        return None

    # Sort by value (in case some are tied)
    anchors.sort(key=lambda t: t[1])

    # Find bracketing pair
    if v <= anchors[0][1]:
        return anchors[0][0]
    if v >= anchors[-1][1]:
        return anchors[-1][0]
    for i in range(len(anchors) - 1):
        p_lo, v_lo = anchors[i]
        p_hi, v_hi = anchors[i + 1]
        if v_lo <= v <= v_hi:
            if v_hi == v_lo:
                return (p_lo + p_hi) / 2.0
            return p_lo + (p_hi - p_lo) * (v - v_lo) / (v_hi - v_lo)
    return None  # unreachable


def cohort_summary(cell: dict | None) -> str:
    """One-line description of which cell the comparison came from."""
    if not cell:
        return "no matching cohort"
    cid = cell.get("cell", {})
    parts = [cid.get("variant_gene", "?")]
    if "variant_protein" in cid:
        parts.append(cid["variant_protein"])
    if "age_years_bucket" in cid:
        parts.append(f"age {cid['age_years_bucket']}")
    if "sex" in cid:
        parts.append(f"sex={cid['sex']}")
    return f"{' / '.join(parts)} (n={cell.get('n', '?')})"
