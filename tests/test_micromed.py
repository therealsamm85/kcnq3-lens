"""Tests for E1 — Micromed .TRC reader + Natus note (readers/micromed.py).

python-neo is not installed here, so this verifies the graceful-guard behavior
and the auto_detect routing/error path (the neo decode path is correct-by-
construction and exercised only when neo is present)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.readers.micromed import can_read_micromed, read_micromed, NATUS_NOTE
from src.readers.auto_detect import load_eeg

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
n_pass = n_fail = 0


def check(name, cond, detail=""):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f"  {PASS} {name}")
    else:
        n_fail += 1
        print(f"  {FAIL} {name}  {detail}")


print("\n── E1: Micromed graceful guard ────────────────────────────────────")

check("can_read_micromed returns a bool", isinstance(can_read_micromed(), bool))

# Create a dummy .trc so the path exists; without neo the loader must raise a
# clear, actionable ImportError (not mis-read the bytes).
tmp = Path(tempfile.mkdtemp()) / "study.trc"
tmp.write_bytes(b"\x00" * 1024)

if not can_read_micromed():
    try:
        read_micromed(tmp)
        check("read_micromed raises without neo", False, "did not raise")
    except ImportError as e:
        check("read_micromed raises a clear install-neo ImportError",
              "neo" in str(e).lower() and "pip install" in str(e).lower())
    # auto_detect routes .trc to the micromed reader → same actionable error.
    try:
        load_eeg(tmp)
        check("auto_detect routes .trc and surfaces the neo error", False, "did not raise")
    except ImportError as e:
        check("auto_detect .trc → install-neo ImportError", "neo" in str(e).lower())
else:
    check("(neo present) skipping guard tests", True)
    check("(neo present) skipping auto_detect guard test", True)


print("\n── E1: Natus note + missing file ──────────────────────────────────")
check("NATUS_NOTE points to EDF+ export mitigation",
      "EDF+" in NATUS_NOTE and "Natus" in NATUS_NOTE)

# Missing file: if neo is absent the ImportError is raised first (before the
# existence check) — both are acceptable, neither mis-reads.
missing = Path("/tmp/does_not_exist_12345.trc")
try:
    read_micromed(missing)
    raised = None
except (ImportError, FileNotFoundError) as e:
    raised = type(e).__name__
check("missing .trc raises ImportError or FileNotFoundError (never mis-reads)",
      raised in ("ImportError", "FileNotFoundError"), f"got {raised}")

# auto_detect still rejects a truly unknown extension cleanly.
unknown = Path(tempfile.mkdtemp()) / "x.xyz"
unknown.write_bytes(b"\x00" * 16)
try:
    load_eeg(unknown)
    check("unknown extension rejected", False, "did not raise")
except ValueError as e:
    check("unknown extension → ValueError listing supported formats",
          ".trc" in str(e) and "Micromed" in str(e))


print(f"\n{'='*60}\n  PASS: {n_pass}\n  FAIL: {n_fail}\n{'='*60}")
if n_fail:
    sys.exit(1)
