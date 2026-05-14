"""Entry point for the packaged desktop app (Mac .app / Windows .exe).

Starts the Streamlit server on a free local port, opens the user's
default browser to localhost, and keeps the server running until the
browser tab is closed.

PyInstaller bundles this script + all dependencies into a standalone
executable. Users download → double-click → app opens in their browser.
No Python install required.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def find_free_port(default: int = 8501) -> int:
    """Try default port first; if busy, find any free one."""
    for port in (default, 8502, 8503, 8504, 8505, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("No free port available")


def app_path() -> str:
    """Locate app.py in both dev (running from repo) and frozen modes."""
    if getattr(sys, "frozen", False):
        # PyInstaller-frozen: app.py is bundled in _MEIPASS or alongside
        base = Path(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)))
        candidate = base / "app.py"
        if candidate.exists():
            return str(candidate)
    # Dev mode
    return str(Path(__file__).parent.parent / "app.py")


def main() -> int:
    port = find_free_port()
    url = f"http://localhost:{port}"

    print(f"🧠 KCNQ3-Lens starting on {url} ...")
    print("   The app will open in your default browser shortly.")
    print("   Keep this window open while you use the app.")
    print("   Close this window (or press Ctrl+C) to stop the app.")

    # Resolve streamlit CLI path
    if getattr(sys, "frozen", False):
        # In a PyInstaller bundle, streamlit is a sub-module
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            app_path(),
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ]
    else:
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            app_path(),
            "--server.port", str(port),
            "--server.headless", "true",
        ]

    proc = subprocess.Popen(cmd)

    # Give Streamlit a moment to start, then open the browser
    time.sleep(2.5)
    try:
        webbrowser.open(url, new=2)
    except Exception:
        print(f"\n   Open this URL manually in your browser: {url}\n")

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n👋 Stopping KCNQ3-Lens...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    return proc.returncode or 0


if __name__ == "__main__":
    sys.exit(main())
