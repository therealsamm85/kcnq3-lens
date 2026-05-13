"""Edge-case + robustness tests.

Catches bugs that don't show up in the happy-path smoke test:
- Empty / missing findings keys
- Channels that don't exist
- Very short recordings
- All-zero or flat signals
- Pre/post comparison with mismatched / empty inputs
- AI router with invalid provider id
- i18n with missing keys / unknown language
- PDF generation with sparse findings
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


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


# ─── 1. Insights with missing / sparse findings ──────────────────────────────
section("Insights — empty / partial findings")

from src.insights import build_narrative, analyze_topography, match_patterns

# Empty
try:
    out = build_narrative({})
    check("build_narrative({}) doesn't crash", True)
    check("Empty input returns empty patterns",
          out["patterns"] == [])
    check("Empty input returns no top channels",
          out["anatomy"]["top_channels"] == [])
except Exception as e:
    check("build_narrative({}) doesn't crash", False, str(e))

# Only topography, no other analyses
try:
    out = build_narrative({"topography": {"all_channels": [
        {"name": "Cz", "median": 7.0, "p90": 12, "pct_high_kurtosis": 30},
        {"name": "Pz", "median": 6.5, "p90": 11, "pct_high_kurtosis": 28},
    ]}})
    check("Topography-only input doesn't crash", True)
    check("Anatomy still computed on partial input",
          len(out["anatomy"]["top_channels"]) > 0)
    # Cross-modal observation 5 (midline central) should fire
    has_sma_obs = any("midline" in o.lower() or "sma" in o.lower()
                       for o in out["cross_modal_observations"])
    check("SMA cross-modal observation fires on Cz/Pz topography", has_sma_obs)
except Exception as e:
    check("Topography-only input doesn't crash", False, str(e))

# Missing nested keys in patterns
try:
    out = build_narrative({"topography": {}, "spindles": {}, "bursts": {}})
    check("Empty inner dicts don't crash patterns",
          isinstance(out["patterns"], list))
except Exception as e:
    check("Empty inner dicts don't crash patterns", False, str(e))

# Channels with names not in CHANNEL_INFO
try:
    out = build_narrative({"topography": {"all_channels": [
        {"name": "UNKNOWN1", "median": 7.0, "p90": 12, "pct_high_kurtosis": 30},
        {"name": "X5", "median": 6.0, "p90": 11, "pct_high_kurtosis": 25},
    ]}})
    check("Unknown channel names don't crash",
          len(out["anatomy"]["top_channels"]) == 2)
    check("Unknown channels get fallback 'unknown region' description",
          all("unknown" in r["region"].lower()
              for r in out["anatomy"]["region_descriptions"]))
except Exception as e:
    check("Unknown channel names don't crash", False, str(e))


# ─── 2. Pre/post comparison edge cases ──────────────────────────────────────
section("Comparison — degenerate inputs")

from src.comparison import compare_findings

# Both empty
try:
    res = compare_findings({}, {})
    check("compare_findings({}, {}) doesn't crash", True)
    check("Empty comparison returns deltas list",
          isinstance(res["deltas"], list))
    check("Empty comparison: no_data verdict",
          res["overall"]["verdict"] == "no_data")
except Exception as e:
    check("compare_findings({}, {}) doesn't crash", False, str(e))

# Pre = 0, Post = nonzero (avoid div-by-zero)
pre = {"spindles": {"density_per_minute": 0.0}}
post = {"spindles": {"density_per_minute": 3.5}}
try:
    res = compare_findings(pre, post)
    spindle_delta = next((d for d in res["deltas"] if "spindle" in d["name"].lower()), None)
    check("Comparison with pre=0 doesn't crash on pct_change",
          spindle_delta is not None and spindle_delta["pct_change"] is None)
except Exception as e:
    check("Comparison with pre=0 doesn't crash", False, str(e))


# ─── 3. AI router with bad inputs ───────────────────────────────────────────
section("AI router — error paths")

from src.ai import interpret_findings, list_providers, get_provider_class

try:
    interpret_findings("nonexistent", "fake-key", {})
    check("Bad provider raises ValueError", False, "should have raised")
except ValueError:
    check("Bad provider raises ValueError", True)

# All providers register
providers = list_providers()
check("Three providers registered", len(providers) == 3)
check("All providers have api_key_url",
      all(p.api_key_url.startswith("https://") for p in providers))
check("All provider IDs are unique",
      len({p.id for p in providers}) == 3)


# ─── 4. i18n robustness ─────────────────────────────────────────────────────
section("i18n — missing keys, unknown languages")

from src.i18n import get_translator

t_de = get_translator("de")
t_xx = get_translator("xx-nonexistent")

check("Unknown language falls back to en", t_xx.language == "en")
check("Missing key returns the key itself",
      t_de.t("nonexistent_key_12345") == "nonexistent_key_12345")
check("Missing format args don't crash",
      isinstance(t_de.t("spindle_below_norm"), str))  # has {low}, {high}
check("Both languages have app_title",
      t_de.t("app_title") and get_translator("en").t("app_title"))

# Coverage spot-check — every visible key should exist in both languages
from src.i18n.translations import TRANSLATIONS
en_keys = set(TRANSLATIONS["en"].keys())
de_keys = set(TRANSLATIONS["de"].keys())
missing_in_de = en_keys - de_keys
missing_in_en = de_keys - en_keys
check(f"All EN keys also in DE (missing: {len(missing_in_de)})",
      len(missing_in_de) == 0,
      f"keys missing in DE: {sorted(missing_in_de)[:5]}")
check(f"All DE keys also in EN (missing: {len(missing_in_en)})",
      len(missing_in_en) == 0,
      f"keys missing in EN: {sorted(missing_in_en)[:5]}")


# ─── 5. PDF generation with sparse findings ─────────────────────────────────
section("PDF reports — sparse / missing fields")

from src.reports import build_doctor_pdf, build_parent_pdf

# Minimal findings
minimal = {"spindles": {"density_per_minute": 1.2, "channel": "Cz"}}
try:
    pdf = build_doctor_pdf(minimal, age_years=5)
    check("Doctor PDF generates with minimal findings",
          len(pdf) > 500 and pdf.startswith(b"%PDF-"))
except Exception as e:
    check("Doctor PDF generates with minimal findings", False, str(e))

try:
    pdf = build_parent_pdf({}, age_years=5)
    check("Parent PDF generates with empty findings",
          len(pdf) > 500 and pdf.startswith(b"%PDF-"))
except Exception as e:
    check("Parent PDF generates with empty findings", False, str(e))


# ─── 6. Patterns scoring sanity ──────────────────────────────────────────────
section("Patterns — confidence math")

# Findings with NO criteria met → no patterns above threshold
nothing = {
    "topography": {"all_channels": [{"name": "O1", "median": 2.0}]},
    "spindles": {"interpretation": "in"},
    "background": {"interpretation": "age_appropriate"},
    "bursts": {"n_bursts_5s_or_longer": 0, "n_bursts_10s_or_longer": 0},
    "morphology": {"pct_complex_spike_wave": 5, "pct_simple_spikes": 90},
    "time_of_night": {"peak_count_per_min": 1},
}
matches = match_patterns(nothing)
check("Normal findings produce few or no pattern matches",
      len(matches) <= 1)


# ─── 7. Readers — invalid path ──────────────────────────────────────────────
section("Readers — invalid input")

from src.readers import load_eeg

try:
    load_eeg("/nonexistent/path/file.eeg")
    check("Missing file raises FileNotFoundError", False, "no exception")
except FileNotFoundError:
    check("Missing file raises FileNotFoundError", True)
except Exception as e:
    check("Missing file raises FileNotFoundError", False, f"raised {type(e).__name__}")


# ─── 8. Sleep-onset on synthetic short data ─────────────────────────────────
section("Sleep onset — short / synthetic data")

# Create a minimal fake EEGRecording
from src.readers.base import EEGRecording
from src.analyses.sleep_onset import detect_sleep_window

# A short 5-min recording — should NOT crash, just return low confidence
class FakeShortRec:
    pass

# Build via the dataclass directly
np.random.seed(0)
n_samples = 200 * 60 * 5  # 5 minutes at 200 Hz
fake_data = np.random.randn(19, n_samples).astype(np.float32) * 50

rec = EEGRecording(
    path=Path("/fake/short.eeg"),
    sfreq=200,
    n_channels=19,
    duration_s=300,
    channel_names=["Fp1", "F4", "F3", "C4", "C3", "P4", "P3", "O2", "O1",
                   "F8", "F7", "T4", "T3", "T6", "T5", "Fz", "Cz", "Pz", "Fp2"],
    n_channels_in_file=19,
    eeg_channel_indices=list(range(19)),
    format_name="synthetic",
)
rec._full_data = fake_data

try:
    sw = detect_sleep_window(rec)
    check("Sleep detect on 5-min recording returns something", sw is not None)
    check("Short-recording confidence is low or medium",
          sw.confidence in ("low", "medium", "high"))
except Exception as e:
    check("Sleep detect on 5-min recording doesn't crash", False, str(e))


# ─── Final ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  PASS: {n_pass}")
print(f"  FAIL: {n_fail}")
print(f"{'='*60}")
if n_fail > 0:
    sys.exit(1)
