"""Minimal BIDS-EEG export — privacy-preserving by construction.

Why this exists
---------------
Research labs and registries (Lerche/Tübingen, RIKEE, Simons, n-Lorem) ingest
data far more easily in BIDS-EEG layout than as a vendor blob. This writes the
standard BIDS directory structure + sidecar metadata directly (no mne-bids
dependency — the layout is simple enough to emit by hand, which keeps the
dependency footprint at numpy/scipy/mne).

Privacy
-------
This exporter NEVER writes PHI. It takes only de-identified fields — a subject
*code* (not a name), an optional age, sex, and variant. It does not write the
patient name, date of birth, or the exact recording date (BIDS allows omitting
acquisition dates; we do). The signal export is opt-in and also carries no
acquisition date. This mirrors the registry's allowlist-by-construction stance.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..readers.base import EEGRecording

# BIDS labels must be alphanumeric (no PHI, no separators that break the spec).
_LABEL_RE = re.compile(r"[^A-Za-z0-9]")


def _clean_label(s: str) -> str:
    return _LABEL_RE.sub("", str(s)) or "x"


@dataclass
class BidsExportResult:
    dataset_root: str
    subject: str
    files_written: list[str]
    signal_exported: bool
    signal_export_note: str = ""


def export_bids(
    rec: EEGRecording,
    out_dir: str | Path,
    subject_label: str,
    *,
    task: str = "rest",
    session: str | None = None,
    metadata: dict | None = None,
    bad_channels: list[str] | None = None,
    include_signal: bool = False,
    power_line_freq: float = 50.0,   # Europe = 50 Hz
    dataset_name: str = "KCNQ3-Lens EEG export",
) -> BidsExportResult:
    """Write a minimal BIDS-EEG dataset for one recording.

    Parameters
    ----------
    rec : EEGRecording
    out_dir : path to the BIDS dataset root (created if absent).
    subject_label : a de-identified subject CODE (e.g. "reference01"). Sanitised
        to alphanumerics; never write a real name here.
    task : BIDS task label (default "rest").
    session : optional session label (e.g. "2023" or a visit code — NOT an
        exact date).
    metadata : optional dict with de-identified participant fields. Recognised
        keys: age (number), sex ("F"/"M"/"n/a"), variant (str), handedness.
    bad_channels : channel names to mark status="bad" in channels.tsv.
    include_signal : if True, export the EEG signal as EDF via MNE (no
        acquisition date embedded). Default False — metadata-only is the
        privacy-safe default.
    power_line_freq : line frequency for the sidecar (50 Hz EU, 60 Hz US).

    Returns
    -------
    BidsExportResult
    """
    root = Path(out_dir)
    sub = _clean_label(subject_label)
    ses = _clean_label(session) if session else None
    metadata = metadata or {}
    bad_upper = {b.upper() for b in (bad_channels or [])}
    written: list[str] = []

    root.mkdir(parents=True, exist_ok=True)

    # --- dataset_description.json ---
    dd = {
        "Name": dataset_name,
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "GeneratedBy": [{"Name": "KCNQ3-Lens", "Description": "qEEG export"}],
    }
    (root / "dataset_description.json").write_text(json.dumps(dd, indent=2))
    written.append("dataset_description.json")

    # --- README ---
    (root / "README").write_text(
        "De-identified BIDS-EEG export from KCNQ3-Lens.\n"
        "Contains no patient name, date of birth, or exact acquisition date.\n"
    )
    written.append("README")

    # --- participants.tsv (de-identified) ---
    age = metadata.get("age")
    sex = metadata.get("sex", "n/a")
    variant = metadata.get("variant", "n/a")
    hand = metadata.get("handedness", "n/a")
    part_path = root / "participants.tsv"
    header = "participant_id\tage\tsex\thandedness\tvariant\n"
    row = (
        f"sub-{sub}\t{age if age is not None else 'n/a'}\t{sex}\t{hand}\t"
        f"{variant}\n"
    )
    # Append if the participant isn't already listed, else (re)create.
    if part_path.exists():
        existing = part_path.read_text()
        if f"sub-{sub}\t" not in existing:
            part_path.write_text(existing + row)
        else:
            part_path.write_text(existing)
    else:
        part_path.write_text(header + row)
    written.append("participants.tsv")

    # --- subject eeg directory ---
    eeg_dir = root / f"sub-{sub}"
    if ses:
        eeg_dir = eeg_dir / f"ses-{ses}"
    eeg_dir = eeg_dir / "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)

    stem = f"sub-{sub}"
    if ses:
        stem += f"_ses-{ses}"
    stem += f"_task-{_clean_label(task)}"

    eeg_names = [rec.channel_names[i] for i in rec.eeg_channel_indices]

    # --- channels.tsv ---
    ch_lines = ["name\ttype\tunits\tsampling_frequency\tstatus"]
    for nm in eeg_names:
        status = "bad" if nm.upper() in bad_upper else "good"
        ch_lines.append(f"{nm}\tEEG\tuV\t{rec.sfreq:.0f}\t{status}")
    (eeg_dir / f"{stem}_channels.tsv").write_text("\n".join(ch_lines) + "\n")
    written.append(f"{stem}_channels.tsv")

    # --- *_eeg.json sidecar ---
    sidecar = {
        "TaskName": task,
        "SamplingFrequency": float(rec.sfreq),
        "EEGChannelCount": len(eeg_names),
        "EEGReference": metadata.get("reference", "unknown"),
        "PowerLineFrequency": power_line_freq,
        "SoftwareFilters": "n/a",
        "RecordingDuration": round(float(rec.duration_s), 1),
        "RecordingType": "continuous",
        "Manufacturer": rec.format_name,
    }
    (eeg_dir / f"{stem}_eeg.json").write_text(json.dumps(sidecar, indent=2))
    written.append(f"{stem}_eeg.json")

    # --- optional signal export (EDF, no acquisition date) ---
    signal_exported = False
    signal_note = ""
    if include_signal:
        signal_exported, signal_note = _export_signal_edf(
            rec, eeg_dir / f"{stem}_eeg.edf"
        )
        if signal_exported:
            written.append(f"{stem}_eeg.edf")

    return BidsExportResult(
        dataset_root=str(root),
        subject=f"sub-{sub}",
        files_written=written,
        signal_exported=signal_exported,
        signal_export_note=signal_note,
    )


def _export_signal_edf(rec: EEGRecording, out_path: Path) -> tuple[bool, str]:
    """Export EEG channels to EDF via MNE, with NO acquisition date.

    Returns (success, note). The note explains WHY it failed (e.g. the optional
    'edfio' package is not installed) rather than failing silently — EDF export
    is opt-in and keeps the base install light.
    """
    try:
        import edfio  # noqa: F401  (MNE's EDF export backend)
    except ImportError:
        return False, (
            "signal not exported: the optional 'edfio' package is required "
            "for EDF export — install with `pip install edfio`. Metadata "
            "(sidecar, channels.tsv, participants.tsv) was still written."
        )
    try:
        import mne
        import numpy as np
        mne.set_log_level("ERROR")
        eeg_idx = rec.eeg_channel_indices
        names = [rec.channel_names[i] for i in eeg_idx]
        # Assemble full signal (µV → volts for MNE). For very long recordings
        # this is memory-heavy; callers choose include_signal deliberately.
        segs = []
        for _, d in rec.iter_epochs():
            segs.append(d[eeg_idx])
        if not segs:
            return False, "signal not exported: recording yielded no epochs."
        data_uv = np.concatenate(segs, axis=1)
        info = mne.create_info(names, rec.sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data_uv * 1e-6, info, verbose="ERROR")
        raw.set_meas_date(None)  # strip acquisition date (privacy)
        raw.export(str(out_path), fmt="edf", overwrite=True, verbose="ERROR")
        return out_path.exists(), ("" if out_path.exists() else "EDF write failed")
    except Exception as e:
        return False, f"signal not exported: {type(e).__name__}: {e}"
