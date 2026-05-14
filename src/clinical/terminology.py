"""ILAE / ACNS standardized terminology mapping.

Pediatric epileptologists use very specific vocabulary. "Sustained
rhythmic burst" is informal; ACNS calls the same thing "Rhythmic Delta
Activity" (RDA) if 0.5–4 Hz, or other categories for other frequencies.
Using the right term makes the report instantly more readable to a
clinician.

This module exposes two helpers:
- `acns_pattern_for_burst(dominant_freq_hz)` — maps a burst's dominant
  frequency to its ACNS category (RDA / RTA / RAA / RBA).
- `ilae_descriptor_for_synchrony(pattern_id)` — translates our
  synchrony categories into ILAE 2017 terminology.

These are mappings, not new analyses — they don't change what's
computed, just what the report calls things.
"""

from __future__ import annotations


# ACNS 2021 critical-care EEG terminology — applied to our bursts
def acns_pattern_for_burst(dominant_freq_hz: float) -> str:
    """Return ACNS pattern name for a rhythmic burst at given dominant frequency.

    R = Rhythmic; Delta/Theta/Alpha/Beta = frequency band; A = Activity.
    """
    if dominant_freq_hz < 0.5:
        return "Sub-delta rhythmic activity (very slow)"
    if dominant_freq_hz < 4:
        return "Rhythmic Delta Activity (RDA, 0.5–4 Hz)"
    if dominant_freq_hz < 8:
        return "Rhythmic Theta Activity (RTA, 4–7 Hz)"
    if dominant_freq_hz < 13:
        return "Rhythmic Alpha Activity (RAA, 8–12 Hz)"
    if dominant_freq_hz < 30:
        return "Rhythmic Beta Activity (RBA, 13–30 Hz)"
    return "High-frequency rhythmic activity (≥30 Hz)"


# ILAE 2017 seizure / discharge classification descriptors — applied to
# our synchrony patterns
ILAE_SYNCHRONY_LABELS = {
    "focal": "Focal — onset limited to a single hemisphere region",
    "regional": "Focal multilobar — onset in adjacent regions of one hemisphere",
    "bilateral_synchronous": (
        "Generalized (or rapidly bilaterally synchronous) — homologous regions "
        "L and R fire simultaneously"
    ),
    "bilateral_asynchronous": (
        "Bilateral independent (multifocal) — both hemispheres but without "
        "synchronous coupling, suggesting two independent foci"
    ),
    "generalized": "Generalized — widespread bihemispheric involvement",
    "no_events": "No qualifying events detected",
}


def ilae_descriptor_for_synchrony(pattern_id: str) -> str:
    return ILAE_SYNCHRONY_LABELS.get(pattern_id, pattern_id)


# Standard EEG report headers (used in PDF restructure if available)
STANDARD_REPORT_SECTIONS = (
    "Background",
    "Sleep",
    "Abnormal findings",
    "Events",
    "Impression",
    "Recommendations",
)
