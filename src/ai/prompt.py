"""Shared system prompts and payload builders used by all LLM providers."""

from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """You are an educational assistant helping a parent understand quantitative findings from their child's EEG recording. You are NOT a doctor and you do NOT give medical advice or diagnoses.

Your job is to:
1. Translate the numerical findings into clear, accessible language a parent can understand.
2. Note which findings are notable (above or below age-typical reference ranges) and what they generally mean about brain function.
3. Suggest specific, well-phrased questions the parent can ask their child's treating neurologist about each finding.
4. Be honest about uncertainty. Many features have wide normal ranges; many can be affected by recording conditions.

You MUST NOT:
- Diagnose any condition.
- Recommend or discourage any medication, supplement, or treatment.
- Tell the parent what to do.
- Override anything the doctor has told them.

Always close with the reminder that the tool is meant to support, not replace, the doctor's interpretation.

Structure your response as:
- "What we measured" — short summary of each analysis result
- "What stands out" — features that are notable relative to age-typical ranges
- "Questions worth asking the doctor" — specific questions for each finding of interest

Use markdown headings and bullet points. Aim for clarity over comprehensiveness; under 600 words is fine.
"""


COMPARISON_SYSTEM_PROMPT = """You are an educational assistant helping a parent compare two EEG recordings of their child — typically before and after a treatment change. You are NOT a doctor and you do NOT give medical advice or diagnoses.

Your job is to:
1. Identify which numerical metrics meaningfully changed between the pre and post recordings.
2. Translate those changes into clear, parent-accessible language: what got better, what got worse, what stayed the same.
3. Be honest that not every numerical change reflects a real biological change — recording conditions, sleep architecture on a given night, electrode placement, and many other factors influence the numbers.
4. Suggest specific questions the parent can bring to the doctor about each notable change.
5. End by emphasizing that the doctor's clinical assessment of the child — symptoms, behavior, development — is more important than EEG numbers alone.

You MUST NOT:
- Diagnose any condition.
- Recommend or discourage any medication, supplement, or treatment change.
- Conclude that a treatment is "working" or "not working" — that's the doctor's call.
- Override anything the doctor has told them.

Structure your response as:
- "What clearly changed" — metrics with substantial change in either direction
- "What didn't really change" — metrics that look similar (within noise)
- "What this might mean (with caveats)" — possible interpretations, all hedged appropriately
- "Questions to bring to the doctor"

Use markdown headings and bullet points. Under 700 words.
"""


def build_findings_payload(
    findings: dict[str, Any],
    age_years: float | None = None,
    variant: str | None = None,
) -> dict[str, Any]:
    """Sanitized payload for single-recording interpretation."""
    return {
        "child_age_years": age_years,
        "known_variant": variant,
        "analyses": findings,
    }


def build_comparison_payload(
    comparison: dict[str, Any],
    age_years: float | None = None,
    variant: str | None = None,
    pre_label: str = "pre-treatment",
    post_label: str = "post-treatment",
) -> dict[str, Any]:
    """Sanitized payload for pre/post comparison interpretation."""
    return {
        "child_age_years": age_years,
        "known_variant": variant,
        "pre_label": pre_label,
        "post_label": post_label,
        "deltas": comparison.get("deltas", []),
        "overall_summary": comparison.get("overall", {}),
        "raw_pre_findings": comparison.get("pre_findings", {}),
        "raw_post_findings": comparison.get("post_findings", {}),
    }


def build_user_message(payload: dict[str, Any], task: str = "single") -> str:
    """Format the payload as a user-facing prompt string."""
    if task == "compare":
        intro = (
            "Below are quantitative comparisons between two EEG recordings — "
            "typically before and after a treatment change. Please interpret "
            "what changed and what didn't, following your system prompt's "
            "structure."
        )
    else:
        intro = (
            "Here are quantitative findings from a pediatric EEG analysis. "
            "Please interpret them for the parent following the structure in "
            "your system prompt."
        )
    return f"{intro}\n\n```json\n{_safe_json(payload)}\n```"


def _safe_json(obj: Any) -> str:
    def _default(o):
        if isinstance(o, tuple):
            return list(o)
        return str(o)

    return json.dumps(obj, default=_default, indent=2)


def build_copy_paste_prompt(
    findings: dict[str, Any],
    age_years: float | None = None,
    variant: str | None = None,
    task: str = "single",
    pre_label: str | None = None,
    post_label: str | None = None,
) -> str:
    """Build a complete, self-contained prompt that a family can paste into
    any free chat interface (ChatGPT / Claude / Gemini web) — no API key
    required.

    The returned string includes:
    - The full role / scope / safety instructions
    - The findings JSON payload (no raw EEG, just metrics)
    - A user-facing closing question

    Use task='single' for single-recording interpretation, 'compare' for
    pre/post comparison.
    """
    if task == "compare":
        system = COMPARISON_SYSTEM_PROMPT
        payload = build_comparison_payload(
            findings, age_years=age_years, variant=variant,
            pre_label=pre_label or "pre-treatment",
            post_label=post_label or "post-treatment",
        )
        closing = (
            "Please interpret these comparison findings for a parent "
            "following the structure described above."
        )
    else:
        system = SYSTEM_PROMPT
        payload = build_findings_payload(
            findings, age_years=age_years, variant=variant,
        )
        closing = (
            "Please interpret these findings for a parent following the "
            "structure described above."
        )

    return (
        "# Instructions for the AI\n\n"
        f"{system}\n\n"
        "---\n\n"
        "# Findings to interpret\n\n"
        "```json\n"
        f"{_safe_json(payload)}\n"
        "```\n\n"
        "---\n\n"
        f"{closing}"
    )
