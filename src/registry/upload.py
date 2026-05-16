"""Upload helpers for registry submissions.

The desktop app never holds GitHub credentials. Instead, the
"Contribute" button opens the user's browser to a pre-filled GitHub
issue on the registry repo, containing the submission JSON in a code
fence. The user clicks "Submit issue" — that's the affirmative action.

A maintainer reviews the issue manually and, on approval, appends the
JSON line to `data/registry.jsonl` in a follow-up PR. CI re-validates.

Why issues, not direct PR
-------------------------
- Direct edits to data/registry.jsonl require a GitHub account AND a
  fork. Non-technical families won't do that.
- Issues require only a GitHub account, which is the minimum bar for
  any opt-in contribution path.
- The maintainer's review step adds a final eyes-on-it check before
  PHI hits the public file.
- If a family decides not to submit after seeing the preview, they
  close the browser tab and nothing has happened.

URL length
----------
GitHub accepts query strings up to ~8 KB. A typical submission
serializes to < 2 KB, well under the limit. We do not chunk.
"""

from __future__ import annotations

import json
import os
import urllib.parse
from typing import Any


# Defaults — overridable via env var so a fork or staging registry can
# point the contribute button somewhere else without code changes.
DEFAULT_OWNER = os.environ.get("KCNQ3_REGISTRY_OWNER", "therealsamm85")
DEFAULT_REPO = os.environ.get("KCNQ3_REGISTRY_REPO", "kcnq3-registry")


def build_issue_url(
    submission: dict,
    *,
    owner: str | None = None,
    repo: str | None = None,
) -> str:
    """Return a https://github.com/.../issues/new URL with the
    submission JSON pre-filled in the body."""
    owner = owner or DEFAULT_OWNER
    repo = repo or DEFAULT_REPO

    sub_pretty = json.dumps(submission, indent=2, sort_keys=False)
    body = (
        "Adding one submission to the registry.\n\n"
        "**Variant:** "
        f"`{submission['subject']['variant_gene']} "
        f"{submission['subject']['variant_protein']}` "
        f"({submission['subject']['variant_type']})\n"
        "**Age bucket:** "
        f"`{submission['subject']['age_years_bucket']}`\n"
        f"**Sex:** `{submission['subject']['sex']}`\n"
        f"**Schema version:** {submission['schema_version']}\n"
        f"**Tool version:** {submission['tool_version']}\n\n"
        "<details><summary>Submission JSON</summary>\n\n"
        "```json\n"
        f"{sub_pretty}\n"
        "```\n"
        "</details>\n\n"
        "By opening this issue I affirm the consent text in "
        "[data/consent_v1.md]"
        f"(https://github.com/{owner}/{repo}/blob/main/data/consent_v1.md)."
    )
    title = (
        "Submission: "
        f"{submission['subject']['variant_gene']} "
        f"{submission['subject']['variant_protein']} "
        f"({submission['subject']['age_years_bucket']}, "
        f"{submission['subject']['sex']})"
    )

    params = urllib.parse.urlencode({
        "title": title,
        "body": body,
        "labels": "submission,needs-review",
    })
    return f"https://github.com/{owner}/{repo}/issues/new?{params}"


def submission_summary_md(submission: dict) -> str:
    """Render a human-readable preview of a submission for the UI.

    Used in the consent + review screen BEFORE the user is sent to
    GitHub. Shows exactly what will be uploaded — nothing hidden.
    """
    s = submission
    subj = s["subject"]
    rec = s["recording"]
    findings = s["findings"]

    lines: list[str] = []
    lines.append("**This is everything that will be sent — nothing else.**")
    lines.append("")
    lines.append("### Subject (de-identified)")
    lines.append(f"- Variant: `{subj['variant_gene']} "
                  f"{subj['variant_protein']}` ({subj['variant_type']})")
    lines.append(f"- Age bucket: `{subj['age_years_bucket']}` "
                  f"(exact age **not** shared)")
    lines.append(f"- Sex: `{subj['sex']}`")
    if subj.get("country_region"):
        lines.append(f"- Country: `{subj['country_region']}`")
    lines.append("")
    lines.append("### Recording (de-identified)")
    lines.append(f"- Duration bucket: `{rec['duration_hours_bucket']}` "
                  f"(exact duration **not** shared)")
    lines.append(f"- Channels: {rec['n_channels']}")
    lines.append(f"- Montage: `{rec['montage']}`")
    lines.append(f"- Had sleep: {rec['had_sleep']}")
    lines.append("")
    lines.append("### Quantitative findings")
    if not findings:
        lines.append("_(no quantitative findings to share — submission "
                      "won't add aggregate value, consider cancelling)_")
    else:
        for k, v in sorted(findings.items()):
            lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    if s.get("intervention"):
        it = s["intervention"]
        lines.append("### Intervention")
        lines.append(f"- Type: `{it['type']}` — name: `{it['name']}`")
        lines.append(f"- Record kind: `{it['record_kind']}`")
        if it.get("linked_pre_submission_id"):
            lines.append(
                f"- Linked pre id: `{it['linked_pre_submission_id']}`"
            )
        lines.append("")
    lines.append("### Identifiers")
    lines.append(f"- Submission ID: `{s['submission_id']}` "
                 f"(save this — needed to withdraw)")
    lines.append(f"- Submitted month: `{s['submitted_at_month']}` "
                 f"(no exact day)")
    return "\n".join(lines)


def to_jsonl_line(submission: dict) -> str:
    """Serialize a submission as a single JSONL line (no trailing \\n).

    Used for the 'copy to clipboard' fallback path — power users can
    paste this directly into data/registry.jsonl in their fork."""
    return json.dumps(submission, separators=(",", ":"), sort_keys=False)
