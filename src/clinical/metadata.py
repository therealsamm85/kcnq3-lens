"""Recording metadata — the clinical context that frames every finding.

A neurologist reading an EEG report wants to know:
- Which medications were active at the time of recording?
- When was it recorded (date, time of day, night-vs-day)?
- What was the indication (routine surveillance / suspected seizure /
  treatment monitoring)?
- Any clinical events during the recording?

Without this context, a "spike rate of 19/min" can mean radically different
things. Same EEG on Sultiam vs no medication is two different reports.

This module defines the metadata schema. The values are user-supplied via
the Streamlit sidebar and embedded into the PDF report header.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as date_t
from typing import Any


@dataclass
class RecordingMetadata:
    """Clinical context for a recording. All fields optional."""

    # Patient
    patient_label: str | None = None        # display name (NOT real PHI — anonymized identifier)
    age_years: float | None = None
    sex: str | None = None                  # 'M' / 'F' / 'X' / None
    variant: str | None = None              # genetic variant if known

    # Recording context
    recording_date: str | None = None       # ISO date 'YYYY-MM-DD'
    recording_time_of_day: str | None = None    # 'morning' / 'afternoon' / 'overnight' / 'all-day'
    recording_indication: str | None = None     # text — why was this done
    technologist_notes: str | None = None       # any observed events during recording

    # Treatment context
    current_medications: list[str] = field(default_factory=list)
    # Each item is a free-text string like "Sultiam 3ml BID" or
    # "Magnesium L-Threonate 100mg evening"
    last_medication_change_date: str | None = None     # ISO date
    days_since_last_seizure: int | None = None         # None if no seizures ever
    treatment_history_summary: str | None = None       # free text

    # Reference recording context (for comparison mode)
    is_post_treatment: bool = False
    comparison_baseline_label: str | None = None


def empty() -> RecordingMetadata:
    return RecordingMetadata()


def to_summary_lines(meta: RecordingMetadata) -> list[tuple[str, str]]:
    """Convert to (label, value) pairs for table rendering. Skips empty fields."""
    out: list[tuple[str, str]] = []

    def add(label: str, val: Any):
        if val is None or val == "" or val == []:
            return
        if isinstance(val, list):
            val = "; ".join(str(v) for v in val)
        out.append((label, str(val)))

    add("Patient", meta.patient_label)
    add("Age (years)", meta.age_years)
    add("Sex", meta.sex)
    add("Variant", meta.variant)
    add("Recording date", meta.recording_date)
    add("Time of day", meta.recording_time_of_day)
    add("Indication", meta.recording_indication)
    add("Current medications", meta.current_medications)
    add("Last medication change", meta.last_medication_change_date)
    add("Days since last seizure", meta.days_since_last_seizure)
    add("Technologist notes", meta.technologist_notes)
    add("Treatment history", meta.treatment_history_summary)
    if meta.is_post_treatment:
        add("Recording type", "Post-treatment "
             f"(baseline: {meta.comparison_baseline_label or 'see prior'})")
    return out


def summarize(meta: RecordingMetadata) -> dict:
    return {
        "patient_label": meta.patient_label,
        "age_years": meta.age_years,
        "sex": meta.sex,
        "variant": meta.variant,
        "recording_date": meta.recording_date,
        "recording_time_of_day": meta.recording_time_of_day,
        "recording_indication": meta.recording_indication,
        "current_medications": list(meta.current_medications),
        "last_medication_change_date": meta.last_medication_change_date,
        "days_since_last_seizure": meta.days_since_last_seizure,
        "technologist_notes": meta.technologist_notes,
        "treatment_history_summary": meta.treatment_history_summary,
        "is_post_treatment": meta.is_post_treatment,
        "comparison_baseline_label": meta.comparison_baseline_label,
    }
