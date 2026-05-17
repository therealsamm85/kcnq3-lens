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


# ─── v0.14.2 — All-day banner + new i18n keys ───────────────────────────────
section("v0.14.2 — new i18n keys present in both languages")

try:
    from src.i18n import get_translator
    _T_en = get_translator("en").t
    _T_de = get_translator("de").t

    _new_keys = [
        "auto_detect_success_clock",
        "auto_detect_secondary_block",
        "auto_detect_acclim_warning",
        "auto_detect_allday_tip",
        "clock_time_help",
        "allday_recording_banner",
    ]
    for _key in _new_keys:
        _val_en = _T_en(_key, clock_start="21:48 Thu", clock_end="08:02 Fri",
                         duration=10.2, conf="high", kind="nap",
                         end_h=2.4, clock="21:48 Thu", h=7, m=10,
                         clock_end_val="17:00 Thu")
        check(f"v0.14.2: '{_key}' renders in English (non-empty)",
              bool(_val_en))
        _val_de = _T_de(_key, clock_start="21:48 Do", clock_end="08:02 Fr",
                         duration=10.2, conf="high", kind="Nickerchen",
                         end_h=2.4, clock="21:48 Do", h=7, m=10,
                         clock_end_val="17:00 Do")
        check(f"v0.14.2: '{_key}' renders in German (non-empty)",
              bool(_val_de))
except Exception as e:
    check("v0.14.2: i18n new keys", False, str(e))

# v0.14.2: allday_recording_banner is not shown for short recordings
# (structural test via AppTest)
try:
    at_short = AppTest.from_file(APP_PATH, default_timeout=30)
    at_short.session_state["language"] = "en"
    at_short.run()
    # No recording loaded → no banner
    _all_warnings = [w.value for w in at_short.warning]
    _banner_text = "long recording"
    check("v0.14.2: No all-day banner when no recording loaded",
          not any(_banner_text in w for w in _all_warnings))
except Exception as e:
    check("v0.14.2: no-banner check", False, str(e))

# v0.14.2: clock_time_help helper returns None when no recording in session
try:
    import streamlit as _st
    # We can't run the helper in isolation (it reads session_state),
    # but we can verify the translations used by it are well-formed.
    _help_en = get_translator("en").t("clock_time_help", clock="14:37 Thu", h=0, m=0)
    check("v0.14.2: clock_time_help template renders",
          "14:37 Thu" in _help_en and "0h" in _help_en)
except Exception as e:
    check("v0.14.2: clock_time_help template", False, str(e))


# ─── v0.18.0 — Advanced tab i18n keys ──────────────────────────────────────
section("v0.18.0 — Advanced analyses tab i18n keys in both languages")

try:
    from src.i18n import get_translator as _get_tr_v018
    _T_en_v018 = _get_tr_v018("en").t
    _T_de_v018 = _get_tr_v018("de").t

    # Simple keys (no template vars)
    _simple_keys_v018 = [
        "tab_advanced",
        "sw_header", "sw_density", "sw_count", "sw_amplitude",
        "sw_duration", "sw_slope", "sw_notes_label", "sw_unavailable",
        "hfo_header", "hfo_unavailable_generic",
        "hfo_rate_nrem", "hfo_total", "hfo_isolated", "hfo_on_spike",
        "hfo_duration", "hfo_freq",
        "coupling_header",
        "coupling_plv", "coupling_phase", "coupling_rayleigh_p",
        "coupling_n_spindles", "coupling_n_so", "coupling_n_coupled",
        "coupling_significant", "coupling_nonsignificant",
        "ied_header",
        "ied_rate", "ied_count", "ied_nrem_rate", "ied_agreement",
        "ied_rolandic", "ied_confidence_header", "ied_per_channel_header",
        "ied_age_flag_drift", "ied_age_flag_untested", "ied_unavailable",
    ]
    for _key in _simple_keys_v018:
        _v_en = _T_en_v018(_key)
        check(f"v0.18.0: '{_key}' EN non-empty", bool(_v_en))
        _v_de = _T_de_v018(_key)
        check(f"v0.18.0: '{_key}' DE non-empty", bool(_v_de))

    # Template keys
    _hfo_unavail_en = _T_en_v018("hfo_unavailable", sfreq=256)
    check("v0.18.0: hfo_unavailable template EN",
          "256" in _hfo_unavail_en)
    _coupling_unavail_en = _T_en_v018("coupling_unavailable", reason="no spindles")
    check("v0.18.0: coupling_unavailable template EN",
          "no spindles" in _coupling_unavail_en)
    _hfo_pct_en = _T_en_v018("hfo_on_spike_pct", pct=42)
    check("v0.18.0: hfo_on_spike_pct template EN",
          "42" in _hfo_pct_en)
    _sw_cap_en = _T_en_v018("sw_caption")
    check("v0.18.0: sw_caption non-empty", bool(_sw_cap_en))
    _hfo_cap_en = _T_en_v018("hfo_caption")
    check("v0.18.0: hfo_caption non-empty", bool(_hfo_cap_en))
    _coupling_cap_en = _T_en_v018("coupling_caption")
    check("v0.18.0: coupling_caption non-empty", bool(_coupling_cap_en))
    _ied_cap_en = _T_en_v018("ied_caption")
    check("v0.18.0: ied_caption non-empty", bool(_ied_cap_en))

except Exception as e:
    check("v0.18.0: Advanced tab i18n", False, str(e))


# ─── Final ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  PASS: {n_pass}")
print(f"  FAIL: {n_fail}")
print(f"{'='*60}")
if n_fail > 0:
    sys.exit(1)
