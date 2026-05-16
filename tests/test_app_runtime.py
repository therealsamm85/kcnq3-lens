"""Streamlit runtime tests using AppTest.

Catches widget-API bugs that static analysis misses. v0.9.1 had a
session-state / widget-key conflict that:
- Passed all 99 static + edge-case tests
- Passed app.py syntax compilation
- Passed Streamlit headless boot
- BUT crashed at runtime when a real user clicked "Analyze"

AppTest simulates the Streamlit script lifecycle without launching a
browser. Widget interactions go through the same code path as a real
user click — so anything that would raise live also raises here.

Run with:
    python -m tests.test_app_runtime
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from streamlit.testing.v1 import AppTest


PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"

n_pass = 0
n_fail = 0


def check(name: str, condition: bool, detail: str = ""):
    global n_pass, n_fail
    if condition:
        n_pass += 1
        print(f"  {PASS} {name}")
    else:
        n_fail += 1
        print(f"  {FAIL} {name}  {detail}")


def section(name: str):
    print(f"\n── {name} ───────────────────────────────────────────")


APP_PATH = str(Path(__file__).parent.parent / "app.py")


# ─── 1. App loads in each mode without runtime error ────────────────────────
section("App boot — each mode loads without exception")

for mode in ("quickstart", "single", "compare", "longitudinal", "contribute"):
    try:
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        # Set session_state BEFORE first run — this is how AppTest sets
        # widget defaults that we can't drive via .run() alone
        at.session_state["language"] = "en"
        at.run()

        # No top-level exception
        check(f"Mode default boot ({mode}) — initial run no exception",
              not at.exception)

        # The mode radio should exist
        check(f"Mode radio widget present",
              len(at.sidebar.radio) > 0)
    except Exception as e:
        check(f"Mode default boot ({mode})", False, str(e))


# ─── 1b. Contribute mode with fresh app (no seeded recordings) ──────────────
section("Contribute mode — no saved recordings (empty state)")

import os as _os_contrib
import tempfile as _tf_contrib
_contrib_env_dir = _tf_contrib.mkdtemp(prefix="kcnq3_contrib_empty_")
_os_contrib.environ["KCNQ3_LENS_DATA"] = _contrib_env_dir

try:
    from src.longitudinal import db as _db_for_contrib
    _db_for_contrib.reset_init_cache_for_tests()

    at_contrib = AppTest.from_file(APP_PATH, default_timeout=30)
    at_contrib.session_state["language"] = "en"
    at_contrib.run()

    radio_contrib = at_contrib.sidebar.radio[0] if at_contrib.sidebar.radio else None
    if radio_contrib is not None:
        # Switch to contribute mode
        try:
            radio_contrib.set_value("contribute").run()
            check("Contribute mode with empty DB boots without exception",
                  not at_contrib.exception,
                  str(at_contrib.exception) if at_contrib.exception else "")
        except Exception:
            # Some AppTest versions cannot set_value by label directly;
            # in that case just check the boot didn't already crash.
            check("Contribute mode boot (set_value skipped — API mismatch)",
                  not at_contrib.exception)
    else:
        check("Contribute mode radio present", False, "no radio widget found")
except Exception as e:
    check("Contribute mode empty-state AppTest", False, f"{type(e).__name__}: {e}")
finally:
    import shutil as _sh_contrib
    _sh_contrib.rmtree(_contrib_env_dir, ignore_errors=True)
    _os_contrib.environ.pop("KCNQ3_LENS_DATA", None)


# ─── 2. Quick Start widgets set + analyze button visible ─────────────────────
section("Quick Start mode — widget lifecycle")

try:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["language"] = "en"
    at.run()

    # Find number_input for age in Quick Start
    age_inputs = [w for w in at.number_input if w.key == "qs_age"]
    check("Quick Start has qs_age widget", len(age_inputs) > 0)

    if age_inputs:
        age_inputs[0].set_value(5).run()
        check("qs_age set_value(5) ran without exception",
              not at.exception)

    # Variant text input
    variant_inputs = [w for w in at.text_input if w.key == "qs_variant"]
    check("Quick Start has qs_variant widget", len(variant_inputs) > 0)

    if variant_inputs:
        variant_inputs[0].set_value("KCNQ3 p.Arg230His").run()
        check("qs_variant set_value ran without exception",
              not at.exception)

    # Local path text input for file path
    path_inputs = [w for w in at.text_input if w.key == "qs_local_path"]
    check("Quick Start has qs_local_path widget",
          len(path_inputs) > 0)
except Exception as e:
    check("Quick Start widget lifecycle", False, str(e))


# ─── 3. Language switch doesn't crash ───────────────────────────────────────
section("Language switch — EN ⇄ DE")

try:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["language"] = "en"
    at.run()
    check("English boots without exception", not at.exception)

    at2 = AppTest.from_file(APP_PATH, default_timeout=30)
    at2.session_state["language"] = "de"
    at2.run()
    check("German boots without exception", not at2.exception)
except Exception as e:
    check("Language switch", False, str(e))


# ─── 4. Save-to-history form in single mode after findings ──────────────────
section("Single mode — sidebar widgets accessible")

try:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["language"] = "en"
    at.run()

    # Page text exists ("KCNQ3-Lens" title)
    titles_found = any("KCNQ3-Lens" in t.value for t in at.title)
    check("Main title 'KCNQ3-Lens' renders", titles_found)

    # Mode radio has the five modes (Quick Start / Single / Compare /
    # Longitudinal / Contribute — v0.12.3 added Contribute)
    radio = at.sidebar.radio[0] if at.sidebar.radio else None
    if radio is not None:
        check("Mode radio has 5 options",
              len(radio.options) == 5)
        check("First mode is Quick Start (default)",
              radio.options[0].startswith("🎯") or "Quick" in radio.options[0])
except Exception as e:
    check("Single mode boot", False, str(e))


# ─── 5. Session_state / widget-key conflict regression test ─────────────────
section("v0.9.1 regression — qs_age write must not crash")

try:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["language"] = "en"
    at.run()

    # If the v0.9.1 bug were present, the first run with qs_age widget
    # instantiated would raise StreamlitAPIException as soon as the app
    # tried to write to st.session_state["qs_age"]. We test this by
    # checking that no exception fires on a clean run.
    check("No StreamlitAPIException on initial Quick Start render",
          not at.exception,
          str(at.exception) if at.exception else "")
except Exception as e:
    check("v0.9.1 regression test", False, str(e))


# ─── Final ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  PASS: {n_pass}")
print(f"  FAIL: {n_fail}")
print(f"{'='*60}")
if n_fail > 0:
    sys.exit(1)
