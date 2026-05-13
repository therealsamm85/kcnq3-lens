"""Proactive clinical insights from quantitative EEG findings.

Translates raw numerical findings into:
- Anatomical descriptions (channel → brain region → function)
- Functional network impact estimates
- Clinical pattern matches (KCNQ-spectrum, CSWS, BECTS, etc.)
- Cross-modal observations (combinations that imply more than each alone)

All output is deterministic (no LLM) and intended to support — not replace —
the doctor's interpretation.
"""

from .narrative import build_narrative
from .anatomical import (
    analyze_topography,
    summarize_anatomy,
    CHANNEL_INFO,
    NETWORK_INFO,
)
from .patterns import match_patterns, summarize_patterns, PATTERNS

__all__ = [
    "build_narrative",
    "analyze_topography",
    "summarize_anatomy",
    "match_patterns",
    "summarize_patterns",
    "CHANNEL_INFO",
    "NETWORK_INFO",
    "PATTERNS",
]
