"""Download a public pediatric EEG sample for first-time users.

Default source: CHB-MIT Scalp EEG Database, chb01_01.edf (PhysioNet).
- 23 EEG channels, 256 Hz, ~1 hour
- Pediatric (ages 1.5–22 across the full dataset; chb01 is age 11 F)
- Open license (PhysioNet ODC-By 1.0)
- ~40 MB

Citation if used (please include in any publication):
  Goldberger A, et al. PhysioBank, PhysioToolkit, and PhysioNet:
  Components of a new research resource for complex physiologic signals.
  Circulation 101(23):e215-e220, 2000.
  Shoeb AH. Application of machine learning to epileptic seizure onset
  detection and treatment. PhD thesis, MIT, 2009.

URL: https://physionet.org/content/chbmit/1.0.0/

Usage:
    python -m scripts.download_sample_data

The file is cached at ~/.kcnq3-lens/samples/chb01_01.edf (or wherever
KCNQ3_LENS_DATA points). Re-running just verifies the cache.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path


SAMPLE_URL = "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_01.edf"
SAMPLE_FILENAME = "chb01_01.edf"
EXPECTED_SIZE_BYTES = 42_399_744   # ~40 MB
EXPECTED_MD5 = None  # Optional integrity check; PhysioNet doesn't publish MD5


def default_sample_dir() -> Path:
    base = Path(os.environ.get("KCNQ3_LENS_DATA",
                                 str(Path.home() / ".kcnq3-lens")))
    return base / "samples"


def sample_path() -> Path:
    return default_sample_dir() / SAMPLE_FILENAME


def is_cached() -> bool:
    """Return True if the sample file is already downloaded and the right size."""
    p = sample_path()
    if not p.exists():
        return False
    try:
        return p.stat().st_size == EXPECTED_SIZE_BYTES
    except OSError:
        return False


def download_sample(force: bool = False, verbose: bool = True) -> Path:
    """Download chb01_01.edf to the sample cache.

    Parameters
    ----------
    force : bool
        If True, re-download even when a cached copy exists.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    Path to the downloaded file.

    Raises
    ------
    RuntimeError if the download fails or size mismatch.
    """
    sample_dir = default_sample_dir()
    sample_dir.mkdir(parents=True, exist_ok=True)
    target = sample_path()

    if not force and is_cached():
        if verbose:
            print(f"✅ Sample already cached: {target}")
        return target

    if verbose:
        print(f"📥 Downloading {SAMPLE_URL}")
        print(f"   → {target}")
        print(f"   (~40 MB; this may take a minute)")

    import urllib.request

    try:
        # Use a streaming download with progress
        with urllib.request.urlopen(SAMPLE_URL, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))
            chunk = 1024 * 64
            downloaded = 0
            with open(target, "wb") as f:
                while True:
                    block = response.read(chunk)
                    if not block:
                        break
                    f.write(block)
                    downloaded += len(block)
                    if verbose and total:
                        pct = 100 * downloaded / total
                        print(f"\r   {downloaded // (1024*1024)}/{total // (1024*1024)} MB ({pct:5.1f}%)",
                              end="", flush=True)
            if verbose:
                print()
    except Exception as e:
        if target.exists():
            target.unlink()
        raise RuntimeError(f"Download failed: {e}") from e

    actual_size = target.stat().st_size
    if actual_size != EXPECTED_SIZE_BYTES:
        target.unlink()
        raise RuntimeError(
            f"Size mismatch: got {actual_size} bytes, "
            f"expected {EXPECTED_SIZE_BYTES}. Download may have been corrupted."
        )

    if verbose:
        print(f"✅ Downloaded {actual_size / (1024*1024):.1f} MB to {target}")
    return target


def sample_description() -> dict:
    """Human-readable description of the sample data — used in the UI."""
    return {
        "name": "CHB-MIT chb01_01.edf",
        "source": "PhysioNet — CHB-MIT Scalp EEG Database",
        "url": "https://physionet.org/content/chbmit/1.0.0/",
        "subject": "Female, age 11, pediatric epilepsy",
        "duration_hours": 1.0,
        "channels": 23,
        "sampling_rate_hz": 256,
        "format": "EDF",
        "size_mb": 40,
        "license": "ODC-By 1.0 (Open Data Commons Attribution)",
        "citation": (
            "Shoeb AH. Application of machine learning to epileptic seizure "
            "onset detection and treatment. PhD thesis, MIT, 2009. "
            "Hosted by PhysioNet: Goldberger A, et al. Circulation 2000."
        ),
        "notes": (
            "Pediatric epilepsy recording (scalp EEG, not intracranial). "
            "KNOWN LIMITATION: this dataset uses a BIPOLAR montage "
            "(channel names like 'FP1-F7', 'F7-T7'). KCNQ3-Lens currently "
            "auto-detects standard monopolar 10-20 channels — bipolar "
            "channels will not be recognised as EEG. Use this sample to "
            "exercise the EDF reader and file pipeline; full analyses "
            "require a monopolar recording. Please cite PhysioNet + Shoeb "
            "if you publish results based on these data."
        ),
        "montage": "bipolar (LIMITATION — see notes)",
    }


if __name__ == "__main__":
    try:
        path = download_sample()
        print()
        info = sample_description()
        print(f"Sample info:")
        print(f"  Subject  : {info['subject']}")
        print(f"  Duration : {info['duration_hours']:.1f} hours")
        print(f"  Channels : {info['channels']}")
        print(f"  Sampling : {info['sampling_rate_hz']} Hz")
        print(f"  Format   : {info['format']}")
        print(f"  License  : {info['license']}")
        print()
        print(f"Path: {path}")
        sys.exit(0)
    except RuntimeError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
