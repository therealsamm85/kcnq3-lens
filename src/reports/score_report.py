"""D2 — SCORE/IFCN-structured report mapping.  [BUILD]

The doctor PDF is bespoke. SCORE (Standardized Computer-based Organized Reporting
of EEG; IFCN/ILAE consensus) defines a structured vocabulary + section layout
clinicians expect. This maps the tool's existing findings onto SCORE-style
sections (background, sleep, interictal epileptiform, [ictal episodes], summary
impression) so a neurologist reads a familiar structure.

BUILD: the SCORE standard is open terminology but there is no open code library
to borrow — this is a transparent mapping from `findings` to a structured dict +
human-readable Markdown. It re-words, it does not re-analyze.

SCAFFOLD — implemented in wave D2.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScoreReport:
    sections: dict[str, list[str]] = field(default_factory=dict)
    impression: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def build_score_report(findings: dict) -> ScoreReport:
    """Map findings to a SCORE/IFCN-structured report. SCAFFOLD — wave D2."""
    raise NotImplementedError("scaffold — implemented in wave D2")


def render_score_markdown(report: ScoreReport) -> str:
    raise NotImplementedError("scaffold — implemented in wave D2")
