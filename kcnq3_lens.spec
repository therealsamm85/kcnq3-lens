# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the KCNQ3-Lens desktop app.

Build locally:
    pyinstaller kcnq3_lens.spec --clean --noconfirm

Build on CI:
    See .github/workflows/build-releases.yml (cross-platform builds).

Output (per platform):
- macOS:   dist/KCNQ3-Lens.app
- Windows: dist/KCNQ3-Lens/KCNQ3-Lens.exe (+ supporting files)
- Linux:   dist/KCNQ3-Lens/KCNQ3-Lens executable

The output is `onedir` mode (not `onefile`) because Streamlit needs to spawn
subprocesses, which `onefile` mode does not support cleanly.
"""

import sys
import os
from pathlib import Path

# ─── Paths ──────────────────────────────────────────────────────────────────
project_root = os.path.abspath(SPECPATH)
app_py = os.path.join(project_root, "app.py")
launcher = os.path.join(project_root, "scripts", "launch_app.py")

# ─── Data files Streamlit / MNE / YASA need at runtime ──────────────────────
datas = [
    (app_py, "."),
    (os.path.join(project_root, "DISCLAIMER.md"), "."),
    (os.path.join(project_root, "src"), "src"),
    (os.path.join(project_root, "scripts"), "scripts"),
]

# Streamlit static assets
try:
    import streamlit as _st
    st_root = Path(_st.__file__).parent
    datas.append((str(st_root / "static"), "streamlit/static"))
    datas.append((str(st_root / "runtime"), "streamlit/runtime"))
except ImportError:
    pass

# MNE bundled data (montages, head models, channel definitions)
try:
    import mne as _mne
    mne_root = Path(_mne.__file__).parent
    if (mne_root / "channels" / "data").exists():
        datas.append((str(mne_root / "channels" / "data"),
                      "mne/channels/data"))
    if (mne_root / "io" / "nihon" / "tests" / "data").exists():
        # We don't need test data, skip
        pass
except ImportError:
    pass

# YASA pretrained sleep-staging models
try:
    import yasa as _yasa
    yasa_root = Path(_yasa.__file__).parent
    if (yasa_root / "classifiers").exists():
        datas.append((str(yasa_root / "classifiers"), "yasa/classifiers"))
except ImportError:
    pass

# ─── Hidden imports — modules PyInstaller's static analysis misses ──────────
hiddenimports = [
    "streamlit",
    "streamlit.runtime",
    "streamlit.runtime.scriptrunner",
    "streamlit.web.cli",
    "streamlit.testing.v1",
    "mne",
    "mne.io.nihon",
    "mne.io.edf",
    "mne.io.brainvision",
    "yasa",
    "yasa.spindles",
    "yasa.staging",
    "scipy",
    "scipy.signal",
    "scipy.stats",
    "numpy",
    "matplotlib",
    "matplotlib.backends.backend_agg",
    "pandas",
    "reportlab",
    "reportlab.platypus",
    "PIL",
    # AI providers — optional; bundle so the import-check works
    "anthropic",
    "openai",
    # Our packages
    "src",
    "src.readers",
    "src.analyses",
    "src.ai",
    "src.ai.providers",
    "src.clinical",
    "src.longitudinal",
    "src.insights",
    "src.reports",
    "src.utils",
    "src.i18n",
]

# ─── Analysis ───────────────────────────────────────────────────────────────
a = Analysis(
    [launcher],
    pathex=[project_root],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Reduce bundle size — exclude large unused libs
        "tkinter",
        "test",
        "unittest",
        "pytest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="KCNQ3-Lens",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,         # disable UPX (causes false-positive AV on Windows)
    console=True,      # keep console window for now — shows the launcher prints
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="KCNQ3-Lens",
)

# macOS .app bundle wrapper
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="KCNQ3-Lens.app",
        icon=None,  # add a .icns file path here later
        bundle_identifier="org.kcnq3-lens.desktop",
        info_plist={
            "CFBundleShortVersionString": "0.10.1",
            "CFBundleVersion": "0.10.1",
            "NSHighResolutionCapable": True,
            "LSBackgroundOnly": False,
            "NSPrincipalClass": "NSApplication",
            "CFBundleName": "KCNQ3-Lens",
            "CFBundleDisplayName": "KCNQ3-Lens",
            "NSHumanReadableCopyright": (
                "MIT License. Not a medical device. See DISCLAIMER.md."
            ),
        },
    )
