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

print(f"\n{'='*60}\n  PASS: {n_pass}\n  FAIL: {n_fail}\n{'='*60}")
if n_fail:
    sys.exit(1)
