#!/usr/bin/env python3
"""
DragonShield Launcher
======================
python launch.py            → desktop viewer + web server
python launch.py --viewer   → desktop viewer only
python launch.py --server   → web server only
python launch.py --video X  → viewer preloaded with X
"""

import argparse
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

HERE = Path(__file__).parent


def ensure_deps():
    required = {
        "cv2":        "opencv-python",
        "numpy":      "numpy",
        "PIL":        "pillow",
        "fastapi":    "fastapi",
        "uvicorn":    "uvicorn[standard]",
        "aiofiles":   "aiofiles",
        "scipy":      "scipy",
        "sklearn":    "scikit-learn",
    }
    missing_pkgs = []
    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            missing_pkgs.append(pkg)

    if missing_pkgs:
        print(f"Installing: {', '.join(missing_pkgs)}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--quiet", "--break-system-packages",
        ] + missing_pkgs)
        print("✓ Dependencies ready")


def start_server_thread():
    """Start FastAPI in a daemon thread. Returns quickly."""
    try:
        import uvicorn
        sys.path.insert(0, str(HERE))

        config = uvicorn.Config(
            "src.dashboard.server:app",
            host="0.0.0.0",
            port=8000,
            log_level="error",   # no terminal spam
            access_log=False,
        )
        server = uvicorn.Server(config)

        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        time.sleep(2)
        print("✓ Web server: http://localhost:8000")
        return server
    except Exception as e:
        print(f"Server failed to start: {e}")
        return None


def start_viewer(video_path=None):
    """Launch the Tkinter desktop viewer (blocks until window is closed)."""
    try:
        import tkinter as tk
    except ImportError:
        print()
        print("=" * 60)
        print("  TKINTER NOT INSTALLED")
        print("=" * 60)
        if sys.platform.startswith("win"):
            print("  Reinstall Python from python.org")
            print("  and check 'tcl/tk and IDLE' during install.")
        elif sys.platform == "darwin":
            print("  brew install python-tk")
        else:
            print("  sudo apt install python3-tk")
        print()
        print("  Falling back to web dashboard only.")
        print("  Open: http://localhost:8000")
        return

    sys.path.insert(0, str(HERE))
    from viewer import DragonShieldViewer

    root = tk.Tk()
    root.minsize(920, 640)
    root.title("DragonShield — Video Detection Viewer")

    # Windows: bring to front
    try:
        root.attributes("-topmost", True)
        root.update()
        root.attributes("-topmost", False)
        root.focus_force()
    except Exception:
        pass

    app = DragonShieldViewer(root, initial_video=video_path)
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    print("✓ Desktop viewer window open")
    print("   ▶ DEMO VIDEO  — loads built-in demo footage (auto-loads)")
    print("   📂 OPEN VIDEO  — upload your own MP4/MOV/AVI/WEBM")
    print()

    root.mainloop()   # blocks until window closed


def main():
    parser = argparse.ArgumentParser(description="DragonShield Launcher")
    parser.add_argument("--viewer",    action="store_true",
                        help="Desktop viewer only")
    parser.add_argument("--server",    action="store_true",
                        help="Web server only (no desktop window)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't auto-open web browser")
    parser.add_argument("--video",     default=None,
                        help="Pre-load specific video file in viewer")
    args = parser.parse_args()

    both   = not (args.viewer or args.server)
    do_srv = args.server or both
    do_ui  = args.viewer or both

    ensure_deps()

    if do_srv:
        start_server_thread()
        if not args.no_browser and not args.viewer:
            try:
                webbrowser.open("http://localhost:8000")
            except Exception:
                pass

    if do_ui:
        # Viewer runs on main thread (required by Tkinter on macOS/Windows)
        start_viewer(args.video)
    elif do_srv:
        print("\n✓ Server running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutdown.")


if __name__ == "__main__":
    main()
