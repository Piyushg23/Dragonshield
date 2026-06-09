#!/usr/bin/env python3
"""
DragonShield v2 — One-command launcher.
Run from the project root:  python run.py

Optional env vars:
  PORT=8000    (default 8000)
  CAMERA=0     (OpenCV camera index, default 0)
"""
import subprocess, sys, os, signal

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load .env before anything else so GROQ_API_KEY is available
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass  # will be installed below with REQUIRED

REQUIRED = [
    "fastapi", "uvicorn[standard]", "websockets",
    "opencv-python", "scikit-learn", "numpy", "scipy", "aiofiles",
    "python-dotenv",
]
OPTIONAL = {
    "ultralytics":  "YOLOv8 drone detection (highly recommended)",
    "pyaudio":      "Live microphone acoustic classification",
    "pyrtlsdr":     "RTL-SDR hardware RF detection",
}

print("=" * 60)
print("  DragonShield v2 — Real Multi-Sensor Counter-UAS")
print("=" * 60)
print("\nChecking required dependencies...")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "--quiet"] + REQUIRED
)
print("  ✓ All required dependencies OK")

print("\nOptional hardware dependencies:")
for pkg, desc in OPTIONAL.items():
    try:
        __import__(pkg.split("[")[0].replace("-", "_"))
        print(f"  ✓ {pkg:20s} — {desc}")
    except ImportError:
        print(f"  ○ {pkg:20s} — {desc}")
        print(f"    Install: pip install {pkg}")

port = int(os.environ.get("PORT", 8000))
groq_key = os.environ.get("GROQ_API_KEY", "")
groq_ok = bool(groq_key and groq_key != "gsk_your_key_here")
print(f"\n{'='*60}")
print(f"  Starting server on http://localhost:{port}")
print(f"  Open your browser → http://localhost:{port}")
print(f"  Groq AI analysis: {'ENABLED' if groq_ok else 'DISABLED (set GROQ_API_KEY in .env)'}")
print(f"  Press Ctrl+C to stop")
print(f"{'='*60}\n")

proc = subprocess.Popen([
    sys.executable, "-m", "uvicorn",
    "src.dashboard.server:app",
    "--host", "0.0.0.0",
    "--port", str(port),
    "--log-level", "info",
])

try:
    proc.wait()
except KeyboardInterrupt:
    print("\n  Shutting down DragonShield...")
    # On Windows, Ctrl+C is already forwarded to the child process by the OS.
    # Just wait for it to exit cleanly — don't send another signal.
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("  Stopped.")
    sys.exit(0)
except Exception as e:
    print(f"\n  Error: {e}")
    proc.terminate()
    sys.exit(1)
