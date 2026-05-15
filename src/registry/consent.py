"""Family consent record — minimum data needed to prove informed opt-in.

We do NOT store an identity. We store:
- Which version of the consent text the family read
- That they affirmed
- When (month only — not exact day, to prevent timestamp correlation)

The actual consent text lives in `data/consent_v1.md` in the registry
repo (referenced by version number). When the consent text changes,
the version increments; previously consented families must re-affirm.

If `given` is False the submission MUST NOT be built. The builder
enforces this.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime


CURRENT_CONSENT_VERSION = 1


@dataclass(frozen=True)
class Consent:
    version: int
    given: bool
    given_at_month: str  # 'YYYY-MM', not exact day

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Consent":
        return cls(
            version=int(d.get("version", 0)),
            given=bool(d.get("given", False)),
            given_at_month=str(d.get("given_at_month", "")),
        )


def make_consent(given: bool) -> Consent:
    """Construct a current-version consent with this month stamped."""
    return Consent(
        version=CURRENT_CONSENT_VERSION,
        given=bool(given),
        given_at_month=datetime.now().strftime("%Y-%m"),
    )
