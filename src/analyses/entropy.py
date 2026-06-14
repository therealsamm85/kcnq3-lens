"""A2 — Entropy / complexity metrics.  [BUILD on numpy]

Sample / permutation / spectral entropy, Hjorth parameters, Higuchi fractal
dimension and a normalized Lempel-Ziv complexity — published markers of
encephalopathic background disorganization. Cheap, local, single-patient-friendly;
feed straight into the longitudinal trackers as per-recording background features.

BUILD (not borrow): each metric is a few lines of numpy, and the project is
dependency-conservative (BIDS export was hand-rolled for the same reason). The
``antropy`` package is a validated reference implementation of the same formulas
if a user wants to cross-check, but it is not required.

Metrics are computed per epoch over an evenly-sampled subset of the recording and
aggregated by median (robust to a few artifact epochs).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import welch

from ..readers.base import EEGRecording

# Cap the per-epoch window fed to the O(N^2) sample-entropy estimator.
_SAMPEN_MAX_SAMPLES = 2000


@dataclass
class EntropyResult:
    channel: str
    is_fallback_channel: bool
    n_epochs_used: int
    metrics: dict[str, float] = field(default_factory=dict)
    backend: str = "numpy"
    notes: list[str] = field(default_factory=list)


# ── individual estimators ──────────────────────────────────────────────────

def _perm_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Normalized Bandt-Pompe permutation entropy in [0, 1]."""
    n = len(x)
    if n < delay * (order - 1) + 1:
        return float("nan")
    counts: dict[tuple, int] = {}
    total = 0
    for i in range(n - delay * (order - 1)):
        window = x[i: i + delay * order: delay]
        pattern = tuple(np.argsort(window))
        counts[pattern] = counts.get(pattern, 0) + 1
        total += 1
    if total == 0:
        return float("nan")
    p = np.array(list(counts.values()), dtype=float) / total
    pe = -np.sum(p * np.log2(p))
    return float(pe / np.log2(math.factorial(order)))


def _spectral_entropy(x: np.ndarray, sf: float) -> float:
    """Normalized Shannon entropy of the power spectral density in [0, 1]."""
    nper = min(len(x), 256)
    if nper < 8:
        return float("nan")
    _f, psd = welch(x, sf, nperseg=nper)
    psd = psd[psd > 0]
    if psd.size < 2:
        return float("nan")
    p = psd / psd.sum()
    se = -np.sum(p * np.log2(p))
    return float(se / np.log2(p.size))


def _sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """SampEn(m, r=r_factor·std). Higher = less self-similar / more irregular."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    r = r_factor * np.std(x)
    if r <= 0 or n < m + 2:
        return float("nan")

    def _count(mm: int) -> int:
        templates = np.array([x[i: i + mm] for i in range(n - mm + 1)])
        total = 0
        for i in range(len(templates)):
            d = np.max(np.abs(templates - templates[i]), axis=1)
            total += int(np.sum(d <= r) - 1)   # exclude self-match
        return total

    b = _count(m)
    a = _count(m + 1)
    if b <= 0 or a <= 0:
        return float("nan")
    return float(-np.log(a / b))


def _hjorth(x: np.ndarray) -> tuple[float, float]:
    """Hjorth mobility and complexity."""
    dx = np.diff(x)
    ddx = np.diff(dx)
    v0, v1, v2 = np.var(x), np.var(dx), np.var(ddx)
    if v0 <= 0 or v1 <= 0:
        return float("nan"), float("nan")
    mobility = math.sqrt(v1 / v0)
    complexity = (math.sqrt(v2 / v1) / mobility) if mobility > 0 else float("nan")
    return float(mobility), float(complexity)


def _higuchi_fd(x: np.ndarray, kmax: int = 10) -> float:
    """Higuchi fractal dimension (≈1 smooth … ≈2 noise-like)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2 * kmax:
        kmax = max(2, n // 2)
    lk: list[float] = []
    ln_inv_k: list[float] = []
    for k in range(1, kmax + 1):
        lengths = []
        for m in range(k):
            idx = np.arange(1, (n - m - 1) // k + 1)
            if idx.size == 0:
                continue
            lmk = np.sum(np.abs(x[m + idx * k] - x[m + (idx - 1) * k]))
            norm = (n - 1) / (idx.size * k)
            lengths.append(lmk * norm / k)
        if lengths:
            lk.append(math.log(np.mean(lengths)))
            ln_inv_k.append(math.log(1.0 / k))
    if len(lk) < 2:
        return float("nan")
    return float(np.polyfit(ln_inv_k, lk, 1)[0])


def _lziv_complexity(x: np.ndarray) -> float:
    """Lempel-Ziv phrase-complexity index (distinct-substring count, median-binarized).

    Binarizes the signal about its median, counts distinct substrings (LZ78-style
    parsing), and scales by log2(n)/n. This is a relative complexity index for
    intra-patient comparison — it is NOT normalized to a [0,1] range; the random-
    signal baseline is length-dependent (~1.5–1.7 at typical epoch lengths), and
    smoother/more-regular signals score lower.
    """
    n = len(x)
    if n < 4:
        return float("nan")
    seq = (np.asarray(x) > np.median(x)).astype(np.uint8)
    s = seq.tobytes()
    substrings: set[bytes] = set()
    ind = 0
    inc = 1
    while ind + inc <= n:
        sub = s[ind: ind + inc]
        if sub in substrings:
            inc += 1
        else:
            substrings.add(sub)
            ind += inc
            inc = 1
    c = len(substrings)
    return float(c * math.log2(n) / n)


# ── public API ─────────────────────────────────────────────────────────────

def compute_entropy(
    rec: EEGRecording,
    target_channel: str = "Pz",
    max_epochs: int = 60,
    epoch_seconds: float = 30.0,
) -> EntropyResult:
    """Compute entropy/complexity metrics on one live channel, median-aggregated."""
    candidates = [target_channel, "Pz", "Cz", "C3", "C4", "O1", "O2", "Fz"]
    if hasattr(rec, "resolve_live_channel"):
        ch_idx, ch_name, is_fb = rec.resolve_live_channel(candidates)
    else:  # duck-typing fallback for synthetic recs in tests
        ch_idx = rec.channel_index(target_channel)
        ch_name, is_fb = (target_channel, False)
        if ch_idx is None and rec.eeg_channel_indices:
            ch_idx = rec.eeg_channel_indices[0]
            ch_name, is_fb = rec.channel_names[ch_idx], True
    if ch_idx is None:
        return EntropyResult(channel="", is_fallback_channel=False, n_epochs_used=0,
                             notes=["no usable channel for entropy"])

    n_ep = rec.n_epochs
    if n_ep <= 0:
        return EntropyResult(channel=ch_name, is_fallback_channel=is_fb,
                             n_epochs_used=0, notes=["recording shorter than one epoch"])
    targets = list(range(n_ep)) if n_ep <= max_epochs else [
        int(k * n_ep / max_epochs) for k in range(max_epochs)
    ]

    acc: dict[str, list[float]] = {
        k: [] for k in ("perm_entropy", "spectral_entropy", "sample_entropy",
                        "hjorth_mobility", "hjorth_complexity", "higuchi_fd",
                        "lziv_complexity")
    }
    used = 0
    for ep in targets:
        d = rec.read_epoch(ep, epoch_seconds)
        if d is None or not (0 <= ch_idx < d.shape[0]):
            continue
        sig = np.asarray(d[ch_idx], dtype=float)
        if sig.size < 16 or np.std(sig) < 1e-6:
            continue
        used += 1
        acc["perm_entropy"].append(_perm_entropy(sig))
        acc["spectral_entropy"].append(_spectral_entropy(sig, rec.sfreq))
        acc["sample_entropy"].append(_sample_entropy(sig[:_SAMPEN_MAX_SAMPLES]))
        mob, cmplx = _hjorth(sig)
        acc["hjorth_mobility"].append(mob)
        acc["hjorth_complexity"].append(cmplx)
        acc["higuchi_fd"].append(_higuchi_fd(sig))
        acc["lziv_complexity"].append(_lziv_complexity(sig))

    metrics: dict[str, float] = {}
    for k, vals in acc.items():
        finite = [v for v in vals if v is not None and np.isfinite(v)]
        metrics[k] = round(float(np.median(finite)), 4) if finite else float("nan")

    notes: list[str] = []
    if is_fb:
        notes.append(f"requested channel unavailable/dead — used {ch_name}")
    if used == 0:
        notes.append("no usable epochs (flat or too short)")
    return EntropyResult(
        channel=ch_name, is_fallback_channel=is_fb, n_epochs_used=used,
        metrics=metrics, backend="numpy", notes=notes,
    )


def summarize_entropy(result: EntropyResult) -> dict:
    return {
        "channel": result.channel,
        "is_fallback_channel": result.is_fallback_channel,
        "n_epochs_used": result.n_epochs_used,
        "metrics": {k: (None if (isinstance(v, float) and math.isnan(v)) else v)
                    for k, v in result.metrics.items()},
        "backend": result.backend,
        "notes": result.notes,
    }
