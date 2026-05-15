"""KCNQ-spectrum federated registry — submission builder + validator.

Public surface:
- `SubmissionInput`, `build_submission`, `BuildError`  (deid.py)
- `Consent`, `make_consent`, `CURRENT_CONSENT_VERSION`  (consent.py)
- `validate_submission`  (validate.py)
- `scan_for_phi`  (phi_check.py)
- `bucket_age_years`, `bucket_duration_hours`  (buckets.py)
- `SCHEMA_VERSION`  (schema.py)

NO module here performs network I/O. The registry is local-first;
the upload path (GitHub PR) is wired in v0.12.3.
"""

from .schema import SCHEMA_VERSION
from .consent import Consent, CURRENT_CONSENT_VERSION, make_consent
from .deid import SubmissionInput, BuildError, build_submission
from .validate import validate_submission
from .phi_check import scan_for_phi, is_clean
from .buckets import (
    bucket_age_years, bucket_duration_hours,
    AGE_BUCKETS, DURATION_BUCKETS,
)

__all__ = [
    "SCHEMA_VERSION",
    "Consent", "CURRENT_CONSENT_VERSION", "make_consent",
    "SubmissionInput", "BuildError", "build_submission",
    "validate_submission",
    "scan_for_phi", "is_clean",
    "bucket_age_years", "bucket_duration_hours",
    "AGE_BUCKETS", "DURATION_BUCKETS",
]
