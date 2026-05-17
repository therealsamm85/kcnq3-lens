"""EEG Microstate analysis (states A–D).

EEG microstates are brief (~60-120 ms) periods of stable scalp potential
topography. Four canonical states (A, B, C, D) account for ~80% of all
scalp EEG variability. Their coverage, duration, and transition probabilities
are sensitive biomarkers for cortical network dysfunction.

Scientific basis
----------------
- Koenig T et al. (2002) Millisecond by millisecond, year by year: normative
  EEG microstates and developmental stages. NeuroImage 16:41–48.
  doi:10.1006/nimg.2002.1070  [canonical A/B/C/D templates and norms]
- Michel CM & Koenig T (2018) EEG microstates as a tool for studying
  the temporal dynamics of whole-brain neuronal networks: a review.
  NeuroImage 180:577–593. doi:10.1016/j.neuroimage.2017.11.062
- Jaime Santana M et al. (2025) EEG microstates as biomarkers of epilepsy.
  Sci Reports 15, 10982. PMID s41598-025-93385-8
- Mofrad MH et al. (2024) Microstate dynamics in pediatric epilepsy.
  Epilepsy & Behavior 156:109784. S1525-5050(24)00110-0
- Kumral D et al. (2025) Pediatric microstate normative data. PMC13078684.
  [B coverage 20-30%, D coverage 20-25%]

Canonical microstate topographies (Koenig 2002 convention)
----------------------------------------------------------
A: Antero-posterior gradient, occipital-left to frontal-right polarity
B: Antero-posterior gradient, occipital-right to frontal-left polarity
C: Anterior-posterior midline (bipolar, max at Cz/Fz)
D: Fronto-central asymmetry (strong frontal dominance)

DISCLAIMER
----------
EEG microstate analysis on scalp EEG with k=4 is a research metric.
The canonical template matching used here is approximate (based on channel
correlations with idealized templates). Pediatric normative values vary
substantially by age, methodology, and reference electrode. The values
cited are from PMC13078684 (Kumral 2025) and should be treated as ROUGH
GUIDES. Clinical interpretation requires expert EEG review.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.signal import find_peaks

from ..readers.base import EEGRecording

_DISCLAIMER = (
    "DISCLAIMER: EEG microstate analysis with k=4 is a RESEARCH METRIC. "
    "Canonical template matching is approximate. Pediatric normative ranges "
    "(B: 20-30%, D: 20-25%) are from Kumral 2025 (PMC13078684) and vary by "
    "age and methodology. Do not use standalone for clinical decisions. "
    "Reference: Koenig 2002 NeuroImage 16:41-48; "
    "Sci Reports 2025 PMID s41598-025-93385-8; "
    "Epilepsy & Behavior 2024 S1525-5050(24)00110-0."
)

_MS_LABELS = ("A", "B", "C", "D")

# Pediatric normative ranges (PMC13078684, Kumral 2025)
_PEDIATRIC_NORM_COVERAGE: dict[str, tuple[float, float]] = {
    "A": (20.0, 28.0),
    "B": (20.0, 30.0),
    "C": (18.0, 26.0),
    "D": (20.0, 25.0),
}


@dataclass
class MicrostateResult:
    """EEG microstate analysis results.

    Fields
    ------
    coverage_pct : dict[str, float]
        Percentage time spent in each microstate A/B/C/D.
    mean_duration_ms : dict[str, float]
        Mean microstate duration in milliseconds.
    occurrence_per_sec : dict[str, float]
        Mean occurrence rate (events/second).
    transition_matrix : dict[tuple[str,str], float]
        Conditional transition probabilities P(to | from).
    n_topomaps : int
        Number of k-means clusters (should be 4).
    method : str
        "pycrostates" or "kmeans_gfp".
    notes : list[str]
    """

    coverage_pct: dict[str, float]
    mean_duration_ms: dict[str, float]
    occurrence_per_sec: dict[str, float]
    transition_matrix: dict[tuple[str, str], float]
    n_topomaps: int
    method: str
    notes: list[str] = field(default_factory=list)


def _compute_gfp(data: np.ndarray) -> np.ndarray:
    """Compute Global Field Power (std across channels at each timepoint).

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)

    Returns
    -------
    gfp : np.ndarray, shape (n_samples,)
    """
    return np.std(data, axis=0)


def _polarity_invariant_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation ignoring polarity (max of corr, -corr)."""
    if np.linalg.norm(a) < 1e-10 or np.linalg.norm(b) < 1e-10:
        return 0.0
    a_n = a / np.linalg.norm(a)
    b_n = b / np.linalg.norm(b)
    c = float(np.dot(a_n, b_n))
    return abs(c)


def _kmeans_polarity_invariant(
    data: np.ndarray, k: int = 4, n_init: int = 20, max_iter: int = 200
) -> np.ndarray:
    """K-means clustering with polarity-invariant distance on topographies.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_channels)
        Topographies at GFP peaks.
    k : int
        Number of clusters.

    Returns
    -------
    centers : np.ndarray, shape (k, n_channels)
        Cluster centers (unit-normalized).
    """
    # Normalize rows
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    X = data / norms

    best_centers = None
    best_inertia = -np.inf

    rng = np.random.default_rng(42)

    for _ in range(n_init):
        # Random init: pick k samples as seeds
        idx = rng.choice(len(X), size=k, replace=False)
        centers = X[idx].copy()

        for _iter in range(max_iter):
            # Assignment step: polarity-invariant
            # corr(x, c) = |x @ c^T| since both are unit vectors
            corr_matrix = np.abs(X @ centers.T)  # (n_samples, k)
            labels = np.argmax(corr_matrix, axis=1)

            # Update step
            new_centers = np.zeros_like(centers)
            for j in range(k):
                mask = labels == j
                if not mask.any():
                    # Reinitialize dead center
                    new_centers[j] = X[rng.integers(len(X))]
                    continue
                cluster = X[mask]
                # Align polarities before averaging
                c = centers[j]
                signs = np.sign(cluster @ c)
                signs[signs == 0] = 1
                aligned = cluster * signs[:, np.newaxis]
                m = aligned.mean(axis=0)
                nm = np.linalg.norm(m)
                new_centers[j] = m / nm if nm > 1e-10 else m

            if np.allclose(new_centers, centers, atol=1e-6):
                break
            centers = new_centers

        # Compute "inertia" = sum of max correlations (higher is better)
        corr_matrix = np.abs(X @ centers.T)
        inertia = float(np.sum(np.max(corr_matrix, axis=1)))
        if inertia > best_inertia:
            best_inertia = inertia
            best_centers = centers.copy()

    return best_centers


def _match_to_canonical(
    centers: np.ndarray, channel_names: list[str]
) -> list[str]:
    """Map k cluster centers to canonical labels A/B/C/D.

    Uses a heuristic channel-based scoring:
    - A: high positive in left occipital (O1, P3), high negative in right frontal
    - B: high positive in right occipital (O2, P4), high negative in left frontal
    - C: high at midline Cz/Pz, negative at Fz (or bipolar frontal-posterior)
    - D: high at frontal (Fz/F3/F4), weaker posterior

    Returns list of labels (same length as n_clusters), using each label at most once.
    If fewer channels are available, falls back to arbitrary labeling.
    """
    k = len(centers)

    # Build channel position proxy
    def _ch_score(center: np.ndarray, positive_chs: list[str], negative_chs: list[str]) -> float:
        score = 0.0
        cn_up = [c.upper() for c in channel_names]
        for ch in positive_chs:
            if ch.upper() in cn_up:
                i = cn_up.index(ch.upper())
                score += center[i]
        for ch in negative_chs:
            if ch.upper() in cn_up:
                i = cn_up.index(ch.upper())
                score -= center[i]
        return score

    canonical_templates = {
        "A": (["O1", "P3", "T5"], ["F8", "T4", "F4"]),
        "B": (["O2", "P4", "T6"], ["F7", "T3", "F3"]),
        "C": (["Cz", "Pz", "P3", "P4"], ["Fz", "Fpz", "Fp1", "Fp2"]),
        "D": (["Fz", "F3", "F4", "Fpz"], ["O1", "O2", "Pz"]),
    }

    # Polarity-invariant score: try both polarities, take max
    scores = np.zeros((k, 4))
    for i, center in enumerate(centers):
        for j, label in enumerate(_MS_LABELS):
            pos_chs, neg_chs = canonical_templates[label]
            s_pos = _ch_score(center, pos_chs, neg_chs)
            s_neg = _ch_score(-center, pos_chs, neg_chs)
            scores[i, j] = max(s_pos, s_neg)

    # Hungarian-style greedy assignment (best score first)
    assigned_labels = ["?"] * k
    used_labels: set[str] = set()
    used_clusters: set[int] = set()

    # Get sorted indices by score descending
    flat_indices = np.argsort(scores.ravel())[::-1]
    for flat_idx in flat_indices:
        ci, li = divmod(int(flat_idx), 4)
        if ci in used_clusters or _MS_LABELS[li] in used_labels:
            continue
        assigned_labels[ci] = _MS_LABELS[li]
        used_clusters.add(ci)
        used_labels.add(_MS_LABELS[li])
        if len(used_clusters) == min(k, 4):
            break

    # Fill any remaining with unused labels
    remaining = [l for l in _MS_LABELS if l not in used_labels]
    for i, lbl in enumerate(assigned_labels):
        if lbl == "?" and remaining:
            assigned_labels[i] = remaining.pop(0)

    return assigned_labels


def _backfit_labels(
    data: np.ndarray, centers: np.ndarray
) -> np.ndarray:
    """Back-fit all timepoints to nearest cluster center (polarity-invariant).

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
    centers : np.ndarray, shape (k, n_channels)

    Returns
    -------
    labels : np.ndarray, shape (n_samples,)
        Cluster index for each timepoint.
    """
    # Normalize data column-wise
    norms = np.linalg.norm(data, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    X = (data / norms).T  # (n_samples, n_channels)

    # Polarity-invariant correlations
    corr = np.abs(X @ centers.T)  # (n_samples, k)
    return np.argmax(corr, axis=1)


def _compute_microstate_metrics(
    labels: np.ndarray,
    cluster_to_ms: list[str],
    sfreq: float,
) -> tuple[dict, dict, dict, dict]:
    """Compute coverage, duration, occurrence, and transition probabilities.

    Returns (coverage_pct, mean_duration_ms, occurrence_per_sec, transition_matrix).
    """
    n_total = len(labels)
    if n_total == 0:
        empty = {ms: 0.0 for ms in _MS_LABELS}
        return empty, empty, empty, {(a, b): 0.0 for a in _MS_LABELS for b in _MS_LABELS}

    # Convert cluster indices to MS labels
    ms_seq = np.array([cluster_to_ms[l] if l < len(cluster_to_ms) else "A"
                       for l in labels])

    coverage_pct: dict[str, float] = {}
    mean_dur_ms: dict[str, float] = {}
    occurrence: dict[str, float] = {}
    recording_s = n_total / sfreq

    for ms in _MS_LABELS:
        mask = ms_seq == ms
        coverage_pct[ms] = float(100.0 * mask.sum() / n_total)

        # Find runs
        runs = []
        in_run = False
        run_len = 0
        for is_ms in mask:
            if is_ms:
                run_len += 1
                in_run = True
            else:
                if in_run:
                    runs.append(run_len)
                    run_len = 0
                    in_run = False
        if in_run:
            runs.append(run_len)

        if runs:
            mean_dur_ms[ms] = float(1000.0 * np.mean(runs) / sfreq)
            occurrence[ms] = float(len(runs) / recording_s)
        else:
            mean_dur_ms[ms] = 0.0
            occurrence[ms] = 0.0

    # Transition matrix: P(to | from), exclude self-transitions
    trans_counts: dict[tuple[str, str], int] = {
        (a, b): 0 for a in _MS_LABELS for b in _MS_LABELS
    }
    for i in range(len(ms_seq) - 1):
        a, b = ms_seq[i], ms_seq[i + 1]
        if a != b:
            trans_counts[(a, b)] += 1

    trans_prob: dict[tuple[str, str], float] = {}
    for a in _MS_LABELS:
        row_total = sum(trans_counts[(a, b)] for b in _MS_LABELS if b != a)
        for b in _MS_LABELS:
            if row_total > 0 and a != b:
                trans_prob[(a, b)] = float(trans_counts[(a, b)] / row_total)
            else:
                trans_prob[(a, b)] = 0.0

    return coverage_pct, mean_dur_ms, occurrence, trans_prob


def compute_microstates(
    rec: EEGRecording,
    sleep_stages: object | None = None,
    target_state: str = "wake",
    n_microstates: int = 4,
    max_epochs: int = 60,
    method: str = "auto",
    channels: list[str] | None = None,
) -> MicrostateResult:
    """Compute EEG microstate analysis (states A-D).

    Parameters
    ----------
    rec : EEGRecording
    sleep_stages : SleepStageResult, optional
        If provided, epochs in `target_state` are used. If None, uses first
        available epochs (heuristic: first 20% of recording as wake proxy).
    target_state : str
        Sleep stage to analyze (default: "wake"). Microstates are best
        characterized during wakefulness.
    n_microstates : int
        Number of microstate classes (default: 4 = canonical A/B/C/D).
    max_epochs : int
        Maximum epochs to process (caps compute time; 60 epochs = 30 min).
    method : str
        "auto" → try pycrostates, fall back to kmeans_gfp.
        "kmeans_gfp" → always use internal k-means on GFP peaks.
        "pycrostates" → require pycrostates library.
    channels : list[str], optional
        Subset of EEG channels to use.

    Returns
    -------
    MicrostateResult
    """
    notes: list[str] = []

    # --- Resolve channels ---
    if channels is None:
        use_indices = rec.eeg_channel_indices[:64]
        use_names = [rec.channel_names[i] for i in use_indices]
    else:
        use_indices = []
        use_names = []
        for ch in channels:
            idx = rec.channel_index(ch)
            if idx is not None:
                use_indices.append(idx)
                use_names.append(ch)
        if not use_indices:
            raise ValueError(f"None of the requested channels found: {channels}")

    n_ch = len(use_indices)
    if n_ch < 4:
        raise ValueError(
            f"Microstate analysis requires at least 4 channels; got {n_ch}"
        )

    # --- Select epochs ---
    if sleep_stages is not None and hasattr(sleep_stages, "stage_per_epoch"):
        epoch_indices = [
            i for i, s in enumerate(sleep_stages.stage_per_epoch)
            if s.lower() == target_state.lower()
        ][:max_epochs]
    else:
        # Heuristic: first 20% of recording
        n_ep = rec.n_epochs
        epoch_indices = list(range(min(max_epochs, max(1, int(n_ep * 0.20)))))
        if sleep_stages is None:
            notes.append(
                f"No sleep staging provided; using first {len(epoch_indices)} epochs"
            )

    if not epoch_indices:
        raise ValueError(
            f"No epochs found for state '{target_state}'. "
            "Check sleep staging or provide explicit epoch indices."
        )

    # --- Load all selected epoch data ---
    epoch_data_list = []
    for ep_idx in epoch_indices:
        data = rec.read_epoch(ep_idx, 30.0)
        if data is None:
            continue
        if data.shape[0] <= max(use_indices):
            continue
        epoch_data = data[use_indices].astype(float)
        # Skip flat epochs
        if np.std(epoch_data) < 1e-10:
            continue
        # Re-reference to average
        epoch_data -= epoch_data.mean(axis=0, keepdims=True)
        epoch_data_list.append(epoch_data)

    if not epoch_data_list:
        raise ValueError("No valid epochs after quality filtering.")

    # Concatenate all epochs: (n_channels, total_samples)
    all_data = np.concatenate(epoch_data_list, axis=1)

    # --- Try pycrostates first ---
    actual_method = "kmeans_gfp"

    if method in ("auto", "pycrostates"):
        try:
            result = _run_pycrostates(
                all_data, use_names, rec.sfreq, n_microstates
            )
            result.notes.insert(0, f"Used pycrostates on {len(epoch_data_list)} epochs")
            result.notes.append(_DISCLAIMER)
            return result
        except Exception as e:
            if method == "pycrostates":
                raise
            notes.append(f"pycrostates failed ({e}); falling back to kmeans_gfp")

    # --- Internal k-means on GFP peaks ---
    result = _run_kmeans_gfp(
        all_data, use_names, rec.sfreq, n_microstates, notes
    )
    result.notes.append(_DISCLAIMER)
    return result


def _run_pycrostates(
    data: np.ndarray,
    channel_names: list[str],
    sfreq: float,
    n_microstates: int,
) -> MicrostateResult:
    """Run pycrostates library if available."""
    import pycrostates  # type: ignore
    from pycrostates.cluster import ModKMeans  # type: ignore
    import mne

    info = mne.create_info(channel_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    model = ModKMeans(n_clusters=n_microstates, random_state=42, n_init=20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(raw, picks="eeg", verbose=False)

    # Extract metrics from pycrostates model
    segmentation = model.predict(raw, verbose=False)
    labels_arr = segmentation.labels  # numpy array of cluster indices

    cluster_to_ms = _match_to_canonical(model.cluster_centers_, channel_names)
    coverage_pct, mean_dur_ms, occ, trans = _compute_microstate_metrics(
        labels_arr, cluster_to_ms, sfreq
    )

    return MicrostateResult(
        coverage_pct=coverage_pct,
        mean_duration_ms=mean_dur_ms,
        occurrence_per_sec=occ,
        transition_matrix=trans,
        n_topomaps=n_microstates,
        method="pycrostates",
        notes=[],
    )


def _run_kmeans_gfp(
    data: np.ndarray,
    channel_names: list[str],
    sfreq: float,
    n_microstates: int,
    notes: list[str],
) -> MicrostateResult:
    """Run internal k-means on GFP peaks."""
    # 1. Compute GFP
    gfp = _compute_gfp(data)  # (n_samples,)

    # 2. Find GFP peaks
    min_dist = max(1, int(0.025 * sfreq))  # ~25 ms minimum separation
    peaks, _ = find_peaks(gfp, distance=min_dist)

    if len(peaks) < n_microstates:
        # Fall back: use all samples
        peaks = np.arange(data.shape[1])
        notes.append("Too few GFP peaks; using all timepoints for clustering")

    # Limit to max 10000 peaks for k-means speed
    if len(peaks) > 10000:
        rng = np.random.default_rng(42)
        peaks = rng.choice(peaks, size=10000, replace=False)

    # 3. Topographies at GFP peaks: (n_peaks, n_channels)
    peak_topos = data[:, peaks].T

    # 4. K-means clustering
    centers = _kmeans_polarity_invariant(peak_topos, k=n_microstates)

    # 5. Match to canonical A/B/C/D
    cluster_to_ms = _match_to_canonical(centers, channel_names)

    # 6. Back-fit all timepoints
    labels = _backfit_labels(data, centers)

    # 7. Compute metrics
    coverage_pct, mean_dur_ms, occ, trans = _compute_microstate_metrics(
        labels, cluster_to_ms, sfreq
    )

    notes.append(
        f"GFP peaks found: {len(peaks)}; "
        f"k-means on {len(peaks)} topographies; "
        f"back-fitted {data.shape[1]} timepoints"
    )

    return MicrostateResult(
        coverage_pct=coverage_pct,
        mean_duration_ms=mean_dur_ms,
        occurrence_per_sec=occ,
        transition_matrix=trans,
        n_topomaps=n_microstates,
        method="kmeans_gfp",
        notes=notes,
    )


def summarize_microstates(result: MicrostateResult) -> dict:
    """Return a JSON-serializable summary dict."""
    # Identify dominant microstate by coverage
    dominant = max(result.coverage_pct, key=lambda k: result.coverage_pct[k])

    # Serialize transition matrix with string keys
    trans_str = {
        f"{a}->{b}": round(result.transition_matrix.get((a, b), 0.0), 3)
        for a in _MS_LABELS for b in _MS_LABELS if a != b
    }

    # Check against pediatric norms
    norm_flags: dict[str, str] = {}
    for ms, (lo, hi) in _PEDIATRIC_NORM_COVERAGE.items():
        cov = result.coverage_pct.get(ms, 0.0)
        if cov < lo:
            norm_flags[ms] = f"below_norm (<{lo}%)"
        elif cov > hi:
            norm_flags[ms] = f"above_norm (>{hi}%)"
        else:
            norm_flags[ms] = "within_norm"

    return {
        "method": result.method,
        "n_topomaps": result.n_topomaps,
        "coverage_pct": {k: round(v, 1) for k, v in result.coverage_pct.items()},
        "mean_duration_ms": {k: round(v, 1) for k, v in result.mean_duration_ms.items()},
        "occurrence_per_sec": {k: round(v, 3) for k, v in result.occurrence_per_sec.items()},
        "dominant_microstate": dominant,
        "transition_probabilities": trans_str,
        "pediatric_norm_flags": norm_flags,
        "pediatric_norm_reference": {
            ms: {"coverage_pct_range": list(rng)}
            for ms, rng in _PEDIATRIC_NORM_COVERAGE.items()
        },
        "disclaimer": _DISCLAIMER,
    }
