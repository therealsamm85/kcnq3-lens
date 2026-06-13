"""Robust re-referencing and bad-channel interpolation — applied lazily.

Why this exists
---------------
The native reference of a recording is whatever the acquisition system used
(often a single physical electrode). A single noisy or drifting reference
electrode contaminates *every* channel, and a single bad channel skews any
averaged metric (the posterior-average PDR, topography). The standard fix is
a robust common-average reference (CAR) computed over the GOOD channels only,
plus interpolation of bad channels from their neighbours.

This module wraps an existing EEGRecording and applies the transform
**per-epoch as the data is read**, so it works on multi-hour recordings that
cannot be held in memory.

Opt-in by design
----------------
Re-referencing changes the signal that every downstream analysis sees, so it
is NOT applied automatically — the caller chooses it explicitly. The load-
bearing PDR (a frequency, not amplitude, measure) is robust to it, but spike
amplitudes and topography do shift, so this stays a deliberate step.
"""

from __future__ import annotations

import copy

import numpy as np

from ..readers.base import EEGRecording

# Minimal 10-20 neighbour map (by scalp adjacency) for interpolation when no
# montage coordinates are available. Each electrode lists its nearest standard
# neighbours; interpolation averages whichever of these are good + present.
_NEIGHBOURS: dict[str, tuple[str, ...]] = {
    "Fp1": ("Fp2", "F3", "F7", "Fz"),
    "Fp2": ("Fp1", "F4", "F8", "Fz"),
    "F7": ("Fp1", "F3", "T3"),
    "F3": ("Fp1", "F7", "Fz", "C3"),
    "Fz": ("Fp1", "Fp2", "F3", "F4", "Cz"),
    "F4": ("Fp2", "F8", "Fz", "C4"),
    "F8": ("Fp2", "F4", "T4"),
    "T3": ("F7", "C3", "T5"),
    "C3": ("F3", "T3", "Cz", "P3"),
    "Cz": ("Fz", "C3", "C4", "Pz"),
    "C4": ("F4", "T4", "Cz", "P4"),
    "T4": ("F8", "C4", "T6"),
    "T5": ("T3", "P3", "O1"),
    "P3": ("C3", "T5", "Pz", "O1"),
    "Pz": ("Cz", "P3", "P4", "O1", "O2"),
    "P4": ("C4", "T6", "Pz", "O2"),
    "T6": ("T4", "P4", "O2"),
    "O1": ("P3", "Pz", "T5"),
    "O2": ("P4", "Pz", "T6"),
}


def apply_common_average_reference(
    rec: EEGRecording,
    bad_channels: list[str] | None = None,
    interpolate_bad: bool = True,
) -> EEGRecording:
    """Return a new EEGRecording that applies CAR + optional interpolation.

    Parameters
    ----------
    rec : EEGRecording
        Source recording (any reader). The original is left untouched.
    bad_channels : list[str], optional
        Channel names to exclude from the average reference and (if
        interpolate_bad) to interpolate from neighbours. If None, no channels
        are treated as bad (plain CAR over all EEG channels).
    interpolate_bad : bool
        If True, bad channels are replaced by the mean of their good
        neighbours (per the 10-20 adjacency map) AFTER re-referencing.

    Returns
    -------
    EEGRecording
        A shallow copy whose epoch reader applies the transform lazily. The
        format_name is suffixed with " (CAR)" so it is visible in reports.

    Notes
    -----
    The CAR is computed over the GOOD EEG channels only (bad channels excluded)
    so a noisy reference/electrode does not leak back into every channel.
    """
    bad_upper = {b.upper() for b in (bad_channels or [])}

    eeg_idx = list(rec.eeg_channel_indices)
    # Good EEG channels = EEG channels not in the bad set.
    good_idx = [i for i in eeg_idx if rec.channel_names[i].upper() not in bad_upper]
    if not good_idx:
        good_idx = eeg_idx  # nothing good — fall back to all EEG (no exclusion)

    # Precompute, for each bad channel index, the file-indices of its good
    # neighbours (for interpolation).
    name_to_idx = {rec.channel_names[i].upper(): i for i in eeg_idx}
    interp_src: dict[int, list[int]] = {}
    if interpolate_bad:
        for i in eeg_idx:
            nm = rec.channel_names[i].upper()
            if nm not in bad_upper:
                continue
            neigh = _NEIGHBOURS.get(rec.channel_names[i], ())
            srcs = [
                name_to_idx[n.upper()] for n in neigh
                if n.upper() in name_to_idx and n.upper() not in bad_upper
            ]
            if srcs:
                interp_src[i] = srcs

    good_idx_arr = np.array(good_idx, dtype=int)
    base_read = rec.read_epoch  # bound method of the source rec

    def _read_epoch_car(_self, ep: int, eps_s: float = 30.0):
        d = base_read(ep, eps_s)
        if d is None:
            return None
        out = d.astype(np.float32, copy=True)
        # Common-average reference over good channels, subtracted from ALL channels.
        car = out[good_idx_arr].mean(axis=0)
        out = out - car
        # Interpolate bad channels from good neighbours (post-reference).
        for bad_i, srcs in interp_src.items():
            out[bad_i] = out[srcs].mean(axis=0)
        return out

    new_rec = copy.copy(rec)
    new_rec._full_data = None          # force the lazy reader path
    new_rec._read_epoch_fn = _read_epoch_car
    new_rec.format_name = f"{rec.format_name} (CAR)"
    return new_rec
