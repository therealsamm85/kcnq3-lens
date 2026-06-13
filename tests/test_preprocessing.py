"""Tests for src/preprocessing — re-referencing + interpolation (Wave 2)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.readers.base import EEGRecording
from src.preprocessing.reference import apply_common_average_reference

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


def _make_rec(data: np.ndarray, names: list[str], sfreq: float = 100.0):
    """Build an in-memory EEGRecording from a (n_ch, n_samp) array."""
    n_ch, n_samp = data.shape
    rec = EEGRecording(
        path=Path("/tmp/synthetic.eeg"),
        sfreq=sfreq,
        n_channels=n_ch,
        duration_s=n_samp / sfreq,
        channel_names=names,
        n_channels_in_file=n_ch,
        eeg_channel_indices=list(range(n_ch)),
        format_name="synthetic",
    )
    rec._full_data = data.astype(np.float32)
    return rec


print("\n── Wave 2: common-average reference + interpolation ───────────────")

# Build a 5-channel, 60s synthetic recording. One dead channel (Fz), one
# common-mode signal injected into all (simulating a noisy reference).
rng = np.random.RandomState(0)
sf = 100.0
n = int(60 * sf)
names = ["Fz", "C3", "C4", "Cz", "Pz"]
common = 50.0 * np.sin(2 * np.pi * 0.5 * np.arange(n) / sf)  # shared "reference" drift
data = np.zeros((5, n), dtype=np.float32)
for i in range(5):
    data[i] = 20.0 * rng.randn(n) + common
data[0] = 0.05 * rng.randn(n)  # Fz = dead/flat
rec = _make_rec(data, names, sf)

# --- Plain CAR (no bad channels) ---
car = apply_common_average_reference(rec)
check("CAR returns a lazy recording (no _full_data)", car._full_data is None)
check("CAR format_name marked", car.format_name.endswith("(CAR)"))
ep = car.read_epoch(0, 30.0)
check("CAR epoch has same channel count", ep.shape[0] == 5)
# After CAR the shared common-mode drift should be largely removed → lower
# cross-channel correlation of the slow component than before.
raw_ep = rec.read_epoch(0, 30.0)
# common-mode removal: mean across channels should be ~0 after CAR
car_mean = ep.mean(axis=0)
check("CAR removes common mode (mean across ch ≈ 0)",
      float(np.abs(car_mean).mean()) < 1e-3,
      f"got {float(np.abs(car_mean).mean()):.4f}")

# --- CAR with bad-channel interpolation ---
car_i = apply_common_average_reference(
    rec, bad_channels=["Fz"], interpolate_bad=True
)
ep_i = car_i.read_epoch(0, 30.0)
fz_std_before = float(raw_ep[0].std())
fz_std_after = float(ep_i[0].std())
check("dead Fz was flat before", fz_std_before < 1.0, f"{fz_std_before:.2f}")
check("interpolated Fz is no longer flat",
      fz_std_after > 1.0, f"{fz_std_after:.2f}")
# Interpolated Fz must equal the mean of its present good neighbours. In this
# synthetic montage the only neighbour of Fz that exists is Cz, so the
# interpolated Fz should equal the (post-CAR) Cz trace. (Note: correlating
# against the good-channel mean would be meaningless — by CAR construction the
# mean of the referenced good channels is ~0.)
cz_after = ep_i[3]  # Cz is index 3 in names
check("interpolated Fz == its present neighbour (Cz)",
      np.allclose(ep_i[0], cz_after, atol=1e-4),
      f"max diff={float(np.abs(ep_i[0]-cz_after).max()):.4f}")

# --- Bad channel excluded from the reference ---
# Inject a huge artifact into Pz; with Pz excluded, CAR should not carry it.
data2 = data.copy()
data2[4] += 5000.0 * np.sin(2 * np.pi * 3 * np.arange(n) / sf)  # junk Pz
rec2 = _make_rec(data2, names, sf)
car_excl = apply_common_average_reference(rec2, bad_channels=["Pz"])
ep_excl = car_excl.read_epoch(0, 30.0)
# C3 (a good channel) should not be swamped by the Pz junk because Pz was
# excluded from the reference.
c3_std = float(ep_excl[1].std())
check("good channel not swamped by excluded junk channel (C3 std < 200µV)",
      c3_std < 200.0, f"C3 std={c3_std:.0f}")

# --- Original recording is untouched ---
check("source recording unchanged (still has _full_data)",
      rec._full_data is not None)

# ─── Wave 3: ocular (blink) artifact detection ──────────────────────────────
print("\n── Wave 3: ocular / blink artifact detection ──────────────────────")
from src.preprocessing.ocular import (
    detect_ocular_artifact, clean_epoch_indices, summarize_ocular,
)

sf3 = 100.0
n3 = int(120 * sf3)  # 4 epochs of 30s
names3 = ["Fp1", "Fp2", "C3", "O1", "O2"]
rng3 = np.random.RandomState(1)
d3 = 15.0 * rng3.randn(5, n3).astype(np.float32)
# Inject big slow blinks on Fp1/Fp2 in epochs 0 and 2 only.
def _blink(center, width, amp):
    t = np.arange(n3)
    return amp * np.exp(-0.5 * ((t - center) / width) ** 2)
for c in (300, 1200, 6300, 7200):  # epochs 0 (0-3000) and 2 (6000-9000)
    b = _blink(c, 25, 200.0)
    d3[0] += b
    d3[1] += b
rec3 = _make_rec(d3, names3, sf3)

oc = detect_ocular_artifact(rec3, blink_amplitude_uv=75.0)
check("ocular available with Fp1/Fp2 present", oc.available)
check("blinks detected (>=4)", oc.n_blinks >= 4, f"n={oc.n_blinks}")
check("blink epochs are 0 and 2",
      set(oc.blink_epoch_indices) == {0, 2},
      f"got {oc.blink_epoch_indices}")
clean = clean_epoch_indices(rec3, oc.blink_epoch_indices, 0, 4)
check("clean epochs exclude blink epochs", set(clean) == {1, 3},
      f"got {clean}")
check("summary is JSON-shaped", isinstance(summarize_ocular(oc), dict))

# No frontopolar → unavailable, no crash
rec3b = _make_rec(d3[2:], ["C3", "O1", "O2"], sf3)
oc_b = detect_ocular_artifact(rec3b)
check("no-frontopolar → available=False, no crash", not oc_b.available)

# ─── Wave 4: autoreject-style epoch rejection ───────────────────────────────
print("\n── Wave 4: per-channel epoch rejection ────────────────────────────")
from src.preprocessing.artifact import compute_rejection, summarize_rejection

sf4 = 100.0
n4 = int(300 * sf4)  # 10 epochs of 30s
names4 = ["Fp1", "Fp2", "C3", "C4", "Cz", "O1"]
rng4 = np.random.RandomState(3)
d4 = 30.0 * rng4.randn(6, n4).astype(np.float32)
# Inject a big multi-channel movement artifact into epochs 3 and 7 (all chans).
for ep_bad in (3, 7):
    s = int(ep_bad * 30 * sf4)
    e = s + int(30 * sf4)
    d4[:, s:e] += 2000.0 * rng4.randn(6, e - s).astype(np.float32)
rec4 = _make_rec(d4, names4, sf4)

r4 = compute_rejection(rec4)
check("rejection: 10 epochs scanned", r4.n_epochs == 10, f"{r4.n_epochs}")
check("rejection: the 2 artifact epochs (3,7) are rejected",
      set(r4.rejected_epoch_indices) == {3, 7},
      f"rejected={r4.rejected_epoch_indices}")
check("rejection: the 8 clean epochs are kept",
      len(r4.clean_epoch_indices) == 8, f"{len(r4.clean_epoch_indices)}")
check("rejection: per-channel thresholds present",
      len(r4.per_channel_threshold_uv) == 6)
check("rejection: summary JSON-shaped",
      isinstance(summarize_rejection(r4), dict))

# A single noisy channel must NOT reject otherwise-clean epochs.
d4b = 30.0 * rng4.randn(6, n4).astype(np.float32)
d4b[5] *= 50.0  # O1 is a loud channel throughout
rec4b = _make_rec(d4b, names4, sf4)
r4b = compute_rejection(rec4b)
check("rejection: one loud channel doesn't reject all epochs",
      r4b.pct_rejected_epochs < 20.0, f"{r4b.pct_rejected_epochs}%")

# ─── Wave 5: BIDS-EEG export (privacy-preserving) ───────────────────────────
print("\n── Wave 5: BIDS-EEG export ────────────────────────────────────────")
import json as _json, tempfile as _tf, shutil as _sh
from pathlib import Path as _P
from src.reports.bids import export_bids

_rec5 = _make_rec(20.0 * np.random.RandomState(5).randn(4, int(60*100)),
                  ["Fp1", "Cz", "Pz", "O1"], 100.0)
_tmp = _P(_tf.mkdtemp(prefix="bidstest_"))
try:
    res = export_bids(
        _rec5, _tmp, subject_label="reference 01!",  # sanitised → reference01
        task="rest", session="visitA",
        metadata={"age": 4.9, "sex": "F", "variant": "KCNQ3 p.Arg230His",
                  "reference": "Cz"},
        bad_channels=["Pz"], include_signal=False,
    )
    root = _P(res.dataset_root)
    check("BIDS: subject label sanitised to alphanumerics",
          res.subject == "sub-reference01", res.subject)
    check("BIDS: dataset_description.json valid",
          _json.loads((root / "dataset_description.json").read_text())
          .get("BIDSVersion") == "1.9.0")
    parts = (root / "participants.tsv").read_text()
    check("BIDS: participants.tsv has the de-identified row",
          "sub-reference01" in parts and "KCNQ3 p.Arg230His" in parts)
    ch = list(root.rglob("*_channels.tsv"))[0].read_text()
    check("BIDS: channels.tsv marks Pz bad",
          "Pz\tEEG\tuV" in ch and "\tbad" in ch)
    sc = _json.loads(list(root.rglob("*_eeg.json"))[0].read_text())
    check("BIDS: sidecar carries sfreq + reference",
          sc["SamplingFrequency"] == 100.0 and sc["EEGReference"] == "Cz")
    # PRIVACY: no name / exact-date / source filename anywhere in the metadata.
    blob = " ".join(p.read_text() for p in root.rglob("*")
                    if p.is_file() and p.suffix in (".json", ".tsv", ""))
    leaks = [s for s in ("the reference patient", "2026-", "REDACTED-DOB") if s in blob]
    check("BIDS: no PHI in metadata", not leaks, f"leaked={leaks}")

    # Signal export: works if edfio present, else a clear note (never silent).
    res_sig = export_bids(_rec5, _tmp, subject_label="t2", include_signal=True)
    try:
        import edfio  # noqa: F401
        _have_edfio = True
    except ImportError:
        _have_edfio = False
    if _have_edfio:
        check("BIDS: signal exported to EDF when edfio present",
              res_sig.signal_exported)
        edf = list(root.rglob("sub-t2*_eeg.edf"))
        check("BIDS: EDF file written", len(edf) == 1)
    else:
        check("BIDS: signal-export note explains missing edfio",
              "edfio" in res_sig.signal_export_note)
finally:
    _sh.rmtree(_tmp, ignore_errors=True)

print(f"\n{'='*60}\n  PASS: {n_pass}\n  FAIL: {n_fail}\n{'='*60}")
if n_fail:
    sys.exit(1)
