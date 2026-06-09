#!/usr/bin/env python3
"""
DragonShield Desktop Viewer v2
================================
- Real MOG2 + Kalman detection on any video
- Improved small-object detection (CLAHE, multi-scale morphology)
- All logging goes to the in-app log panel, not terminal
- Demo video pre-loaded on startup
- Upload any MP4/MOV/AVI/WEBM via file picker or drag-drop

Usage:
    python viewer.py
    python viewer.py --video my_footage.mp4
"""

import argparse, logging, math, os, sys, threading, time
from collections import deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import cv2
import numpy as np
try:
    from PIL import Image, ImageTk
except ImportError:
    print("Run: pip install pillow"); sys.exit(1)

HERE      = Path(__file__).parent
DEMO_VIDEO = HERE / "src" / "dashboard" / "static" / "demo_video.mp4"

# ── suppress all console output — everything goes to the UI log ───────────────
logging.disable(logging.CRITICAL)

# ── palette ───────────────────────────────────────────────────────────────────
C_BG     = "#05080f"
C_PANEL  = "#0b1421"
C_BORDER = "#1a3050"
C_BLUE   = "#00c8ff"
C_GREEN  = "#00ff88"
C_AMBER  = "#ffcc00"
C_ORANGE = "#ff6600"
C_RED    = "#ff1f3a"
C_DIM    = "#4a7090"
C_FAINT  = "#1e3a55"

def score_hex(s):
    if s < 0.10: return C_GREEN
    if s < 0.35: return "#88ff00"
    if s < 0.60: return C_AMBER
    if s < 0.80: return C_ORANGE
    return C_RED

def score_bgr(s):
    if s < 0.10: return (136, 255, 0)
    if s < 0.35: return (0, 255, 136)
    if s < 0.60: return (0, 200, 255)
    if s < 0.80: return (0, 100, 255)
    return (0, 0, 255)

def score_label(s):
    if s < 0.10: return "CLEAR"
    if s < 0.35: return "LOW"
    if s < 0.60: return "MEDIUM"
    if s < 0.80: return "HIGH"
    return "CRITICAL"


# ══════════════════════════════════════════════════════════════════════════════
#  REAL DETECTOR — MOG2 + CLAHE + Multi-scale + Kalman tracking
#  Improved for small/distant objects
# ══════════════════════════════════════════════════════════════════════════════

class KalmanTrack:
    F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
    H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float64)
    Q = np.diag([0.5, 0.5, 2.0, 2.0])
    R = np.eye(2) * 3.0

    def __init__(self, cx, cy):
        self.x = np.array([cx, cy, 0., 0.])
        self.P = np.eye(4) * 8.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, mx, my):
        z = np.array([mx, my])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def pos(self): return float(self.x[0]), float(self.x[1])
    @property
    def vel(self): return float(self.x[2]), float(self.x[3])


class RealDetector:
    """
    MOG2 background subtraction with CLAHE contrast enhancement,
    multi-scale morphological filtering for small objects,
    and Kalman multi-object tracker.
    """

    def __init__(self):
        self._reset_state()
        self._t_buf = deque(maxlen=30)

    def _reset_state(self):
        # Two background models: one sensitive (small objects), one robust
        self.bg_sensitive = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=18, detectShadows=False)
        self.bg_robust = cv2.createBackgroundSubtractorMOG2(
            history=400, varThreshold=45, detectShadows=False)

        # CLAHE for contrast enhancement (helps small/dark objects)
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

        # Morphological kernels at different scales
        self.k_small  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.k_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.k_large  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

        self._tracks: dict = {}
        self._next_id = 0
        self._frame_n = 0
        self._t_buf = deque(maxlen=30)

    def reset(self):
        self._reset_state()

    def process(self, frame: np.ndarray) -> dict:
        self._frame_n += 1
        h, w = frame.shape[:2]

        # ── Preprocessing: CLAHE on luminance ────────────────────────────────
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # ── Dual-scale background subtraction ────────────────────────────────
        fg_sensitive = self.bg_sensitive.apply(enhanced)
        fg_robust    = self.bg_robust.apply(frame)

        # ── Small object detection path (fewer morphological ops) ────────────
        fg_small = cv2.morphologyEx(fg_sensitive, cv2.MORPH_OPEN, self.k_small)
        fg_small = cv2.dilate(fg_small, self.k_small, iterations=1)

        # ── Robust detection path (fewer false positives) ─────────────────────
        fg_big = cv2.morphologyEx(fg_robust, cv2.MORPH_OPEN, self.k_medium)
        fg_big = cv2.dilate(fg_big, self.k_medium, iterations=2)

        # ── Combine: union for recall ─────────────────────────────────────────
        fg_combined = cv2.bitwise_or(fg_small, fg_big)
        # Final cleanup
        fg_combined = cv2.morphologyEx(fg_combined, cv2.MORPH_CLOSE, self.k_medium)

        if self._frame_n <= 12:
            annotated = self._draw_hud(frame.copy(), [], 0.0, w, h, warmup=True)
            return {"tracks": [], "n_tracks": 0, "score": 0.0,
                    "frame": annotated, "fps": 0.0, "warmup": True}

        # ── Contour detection ─────────────────────────────────────────────────
        cnts, _ = cv2.findContours(fg_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in cnts:
            area = cv2.contourArea(c)
            # Accept objects from 35px² (tiny distant drones) to 50k px²
            if area < 35 or area > 50_000:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            asp = bw / max(bh, 1)
            # Relax aspect ratio for small objects (may be circular/square)
            max_asp = 7.0 if area < 300 else 5.0
            if not (0.12 < asp < max_asp):
                continue
            if bw > w * 0.55 or bh > h * 0.55:
                continue
            # Confidence scales with area but is higher for smaller objects
            # when coming from sensitive detector
            in_sensitive = bool(fg_sensitive[int(y+bh/2), int(x+bw/2)] > 0)
            base_conf = min(0.94, area / 6000 * 0.6 + 0.28)
            # Boost small objects detected in sensitive path
            if area < 400 and in_sensitive:
                base_conf = min(0.94, base_conf + 0.15)
            detections.append({
                "cx": x + bw / 2, "cy": y + bh / 2,
                "x1": float(x), "y1": float(y),
                "x2": float(x + bw), "y2": float(y + bh),
                "w": float(bw), "h": float(bh),
                "conf": base_conf,
                "small": area < 400,
            })

        tracks = self._kalman_step(detections)
        score = max((t["threat"] for t in tracks), default=0.0)
        if len(tracks) > 1:
            score = min(1.0, score + (len(tracks) - 1) * 0.04)

        annotated = self._annotate(frame.copy(), tracks, score, w, h)

        now = time.perf_counter()
        self._t_buf.append(now)
        fps = (len(self._t_buf)-1) / max(self._t_buf[-1]-self._t_buf[0], 1e-3) \
              if len(self._t_buf) > 1 else 0.0

        return {"tracks": tracks, "n_tracks": len(tracks), "score": round(score, 4),
                "frame": annotated, "fps": round(fps, 1), "warmup": False}

    def _kalman_step(self, detections):
        for t in self._tracks.values():
            t["kf"].predict()

        tids = list(self._tracks.keys())
        used_t, used_d = set(), set()
        matched = {}
        if tids and detections:
            cost = np.zeros((len(tids), len(detections)))
            for ti, tid in enumerate(tids):
                px, py = self._tracks[tid]["kf"].pos
                for di, d in enumerate(detections):
                    cost[ti, di] = math.hypot(px - d["cx"], py - d["cy"])
            for idx in np.argsort(cost, axis=None):
                ti, di = divmod(int(idx), len(detections))
                if ti in used_t or di in used_d: continue
                if cost[ti, di] > 100: break
                matched[tids[ti]] = detections[di]
                used_t.add(ti); used_d.add(di)

        for tid, det in matched.items():
            kf = self._tracks[tid]["kf"]
            kf.update(det["cx"], det["cy"])
            self._tracks[tid].update({"box": det, "gone": 0})
            self._tracks[tid]["frames"] += 1
            pos = kf.pos
            self._tracks[tid]["traj"].append(pos)
            if len(self._tracks[tid]["traj"]) > 60:
                self._tracks[tid]["traj"].pop(0)

        for di, det in enumerate(detections):
            if di in used_d: continue
            nid = self._next_id; self._next_id += 1
            self._tracks[nid] = {
                "kf": KalmanTrack(det["cx"], det["cy"]),
                "box": det, "gone": 0, "frames": 1,
                "traj": [(det["cx"], det["cy"])],
            }

        for ti, tid in enumerate(tids):
            if ti not in used_t:
                self._tracks[tid]["gone"] += 1

        self._tracks = {tid: t for tid, t in self._tracks.items() if t["gone"] <= 22}

        out = []
        for tid, t in self._tracks.items():
            if t["gone"] > 0: continue
            pos = t["kf"].pos
            vel = t["kf"].vel
            box = t["box"]
            dist_m = max(3.0, (0.3 * 800) / max(box["w"], 1))
            spd = math.hypot(*vel)
            threat = min(1.0, 10.0/max(dist_m, 1) * 0.55 + box["conf"] * 0.45)
            # Small confirmed objects with long track history get threat boost
            if box.get("small") and t["frames"] > 8:
                threat = min(1.0, threat + 0.12)
            heading = (math.degrees(math.atan2(vel[1], vel[0])) + 360) % 360
            out.append({
                "id": tid,
                "x1": int(box["x1"]), "y1": int(box["y1"]),
                "x2": int(box["x2"]), "y2": int(box["y2"]),
                "cx": round(pos[0], 1), "cy": round(pos[1], 1),
                "vx": round(vel[0], 2), "vy": round(vel[1], 2),
                "heading": round(heading, 1),
                "distance_m": round(dist_m, 1),
                "confidence": round(box["conf"], 3),
                "threat": round(threat, 3),
                "frames": t["frames"],
                "small": box.get("small", False),
                "traj": list(t["traj"]),
            })
        return out

    def _annotate(self, frame, tracks, score, w, h):
        for t in tracks:
            c = score_bgr(t["threat"])
            x1, y1, x2, y2 = t["x1"], t["y1"], t["x2"], t["y2"]
            cx, cy = int(t["cx"]), int(t["cy"])
            bw = max(1, x2 - x1)
            is_small = t.get("small", bw < 20)

            # For tiny objects: draw a larger indicator ring around them
            if is_small:
                ring_r = max(14, bw + 10)
                cv2.circle(frame, (cx, cy), ring_r, c, 1)
                cv2.circle(frame, (cx, cy), ring_r + 3, c, 1)
                cv2.drawMarker(frame, (cx, cy), c, cv2.MARKER_CROSS, 10, 1)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), c, 1)

            # Corner brackets (adapt size to object)
            ca = max(5, min(14, bw // 3))
            bx1 = max(0, cx - ring_r - 2) if is_small else x1
            by1 = max(0, cy - ring_r - 2) if is_small else y1
            bx2 = min(w-1, cx + ring_r + 2) if is_small else x2
            by2 = min(h-1, cy + ring_r + 2) if is_small else y2
            for ox, oy, dx, dy in [
                (bx1,by1,ca,0),(bx1,by1,0,ca),(bx2,by1,-ca,0),(bx2,by1,0,ca),
                (bx1,by2,ca,0),(bx1,by2,0,-ca),(bx2,by2,-ca,0),(bx2,by2,0,-ca),
            ]:
                cv2.line(frame, (ox, oy), (ox+dx, oy+dy), c, 2)

            # Threat arc
            arc_r = bw // 2 + (16 if is_small else 14)
            if t["threat"] > 0.02:
                pts = []
                for deg in range(0, int(t["threat"]*360)+1, 5):
                    rd = math.radians(deg - 90)
                    px2 = int(cx + arc_r * math.cos(rd))
                    py2 = int(cy + arc_r * math.sin(rd))
                    if 0 <= px2 < w and 0 <= py2 < h:
                        pts.append((px2, py2))
                if len(pts) > 1:
                    cv2.polylines(frame, [np.array(pts, np.int32)], False, c, 2)

            # Velocity arrow
            spd = math.hypot(t["vx"], t["vy"])
            if spd > 0.4:
                ang = math.atan2(t["vy"], t["vx"])
                r = arc_r + 4
                ax = int(cx + math.cos(ang) * r)
                ay = int(cy + math.sin(ang) * r)
                ex = int(ax + math.cos(ang) * min(spd * 10, 45))
                ey = int(ay + math.sin(ang) * min(spd * 10, 45))
                if 0 < ex < w and 0 < ey < h:
                    cv2.arrowedLine(frame, (ax, ay), (ex, ey), c, 2, tipLength=0.35)

            # Trajectory tail
            traj = t["traj"]
            for i in range(1, len(traj)):
                alpha = i / len(traj)
                pt1 = (int(traj[i-1][0]), int(traj[i-1][1]))
                pt2 = (int(traj[i][0]),   int(traj[i][1]))
                fade = tuple(int(v * alpha * 0.55) for v in c)
                cv2.line(frame, pt1, pt2, fade, 1)

            # Label
            label = f"T-{t['id']}  {int(t['confidence']*100)}%  ~{t['distance_m']:.0f}m"
            sub   = f"{t['heading']:.0f}deg  {int(t['threat']*100)}%thr" + (" [SMALL]" if is_small else "")
            lx = max(0, bx1)
            cv2.putText(frame, label, (lx, max(14, by1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, c, 1, cv2.LINE_AA)
            cv2.putText(frame, sub,   (lx, max(24, by1 - 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, c, 1, cv2.LINE_AA)

        return self._draw_hud(frame, tracks, score, w, h)

    def _draw_hud(self, frame, tracks, score, w, h, warmup=False):
        lvl = score_label(score)
        c   = score_bgr(score)
        ov  = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 28), (3, 8, 14), -1)
        cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)

        top_txt = f"DRAGONSHIELD  [{'WARMUP...' if warmup else lvl}]"
        cv2.putText(frame, top_txt, (10, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, c if not warmup else (60,80,100), 1, cv2.LINE_AA)

        trk_txt = f"{len(tracks)} TRACKS"
        tw = cv2.getTextSize(trk_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0][0]
        cv2.putText(frame, trk_txt, (w - tw - 10, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 140, 180), 1, cv2.LINE_AA)

        score_txt = f"VIS {score:.3f}"
        stw = cv2.getTextSize(score_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0][0]
        cv2.putText(frame, score_txt, (w//2 - stw//2, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, c, 1, cv2.LINE_AA)

        fill = int(w * score)
        cv2.rectangle(frame, (0, h-6), (w, h), (10, 18, 28), -1)
        if fill > 0:
            cv2.rectangle(frame, (0, h-6), (fill, h), c, -1)

        if score > 0.80 and int(time.time() * 3) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 4)

        L = 18
        for ox, oy, sx, sy in [(0,0,1,1),(w,0,-1,1),(0,h,1,-1),(w,h,-1,-1)]:
            cv2.line(frame, (ox,oy), (ox+sx*L, oy),    c, 2)
            cv2.line(frame, (ox,oy), (ox,     oy+sy*L), c, 2)

        return frame


# ══════════════════════════════════════════════════════════════════════════════
#  TKINTER APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class DragonShieldViewer:

    DISPLAY_W = 854
    DISPLAY_H = 480

    def __init__(self, root: tk.Tk, initial_video=None):
        self.root = root
        self.root.title("DragonShield — Video Detection Viewer")
        self.root.configure(bg=C_BG)
        self.root.resizable(True, True)

        self._cap           = None
        self._detector      = RealDetector()
        self._playing       = False
        self._lock          = threading.Lock()
        self._thread        = None
        self._video_path    = None
        self._total_frames  = 0
        self._frame_n       = 0
        self._result        = None
        self._score_history = deque(maxlen=120)
        self._alarm_count   = 0
        self._peak_score    = 0.0
        self._session_start = time.time()
        self._display_w     = self.DISPLAY_W
        self._display_h     = self.DISPLAY_H
        self._seeking       = False
        self._log_entries   = deque(maxlen=200)
        self._total_detected = 0

        self._build_ui()
        self._log("DragonShield v2 ready.", level="INFO")
        self._log("Click DEMO VIDEO or OPEN VIDEO to begin.", level="INFO")

        if initial_video and Path(str(initial_video)).exists():
            self.root.after(300, lambda: self._load_video(str(initial_video)))
        elif DEMO_VIDEO.exists():
            self.root.after(300, lambda: self._load_video(str(DEMO_VIDEO), is_demo=True))
        else:
            self._log("Demo video not found. Use OPEN VIDEO.", level="WARN")

        self.root.after(33, self._ui_tick)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top bar
        top = tk.Frame(self.root, bg=C_BG, height=46)
        top.pack(fill=tk.X, padx=8, pady=(8, 0))

        tk.Label(top, text="⬡  DRAGONSHIELD", bg=C_BG, fg=C_BLUE,
                 font=("Courier", 14, "bold")).pack(side=tk.LEFT, padx=8)
        tk.Label(top, text="COUNTER-UAS · MOG2+KALMAN · REAL DETECTION",
                 bg=C_BG, fg=C_FAINT, font=("Courier", 8)).pack(side=tk.LEFT)

        self._threat_lbl = tk.Label(top, text="▶ STANDBY", bg=C_BG, fg=C_DIM,
                                    font=("Courier", 12, "bold"))
        self._threat_lbl.pack(side=tk.LEFT, padx=24)

        meta = tk.Frame(top, bg=C_BG)
        meta.pack(side=tk.RIGHT, padx=10)
        self._fps_lbl    = tk.Label(meta, text="FPS: —",      bg=C_BG, fg=C_FAINT, font=("Courier", 8))
        self._alarm_lbl  = tk.Label(meta, text="ALARMS: 0",   bg=C_BG, fg=C_FAINT, font=("Courier", 8))
        self._peak_lbl   = tk.Label(meta, text="PEAK: 0%",    bg=C_BG, fg=C_FAINT, font=("Courier", 8))
        self._uptime_lbl = tk.Label(meta, text="UP: 00:00",   bg=C_BG, fg=C_FAINT, font=("Courier", 8))
        for i, lbl in enumerate([self._fps_lbl, self._alarm_lbl, self._peak_lbl, self._uptime_lbl]):
            lbl.grid(row=0, column=i, padx=8)

        # Main area
        main = tk.Frame(self.root, bg=C_BG)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # Video panel
        vid_frame = tk.Frame(main, bg=C_BG,
                             highlightbackground=C_BORDER, highlightthickness=1)
        vid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(vid_frame, bg="#000000",
                                 width=self.DISPLAY_W, height=self.DISPLAY_H,
                                 cursor="crosshair", highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas.bind("<Configure>", lambda e: setattr(self, "_display_w", e.width) or setattr(self, "_display_h", e.height))
        self._canvas.create_text(self.DISPLAY_W//2, self.DISPLAY_H//2,
                                 text="LOAD A VIDEO OR CLICK  DEMO VIDEO",
                                 fill=C_FAINT, font=("Courier", 12), tags="placeholder")

        # Side panel
        side = tk.Frame(main, bg=C_PANEL, width=210,
                        highlightbackground=C_BORDER, highlightthickness=1)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        side.pack_propagate(False)
        self._build_side(side)

        # Controls bar
        ctrl = tk.Frame(self.root, bg=C_PANEL,
                        highlightbackground=C_BORDER, highlightthickness=1)
        ctrl.pack(fill=tk.X, padx=8, pady=(0, 4))

        B = {"bg": C_BG, "activebackground": C_BORDER, "activeforeground": C_BLUE,
             "relief": "flat", "bd": 0, "font": ("Courier", 9, "bold"),
             "cursor": "hand2", "padx": 10, "pady": 6}

        self._btn_demo = tk.Button(ctrl, text="▶  DEMO VIDEO",
                                   fg=C_AMBER, command=self._open_demo, **B)
        self._btn_demo.pack(side=tk.LEFT, padx=4, pady=4)

        self._btn_open = tk.Button(ctrl, text="📂  OPEN VIDEO",
                                   fg=C_BLUE, command=self._open_file, **B)
        self._btn_open.pack(side=tk.LEFT, padx=4, pady=4)

        tk.Frame(ctrl, bg=C_BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y, pady=6)

        self._btn_play = tk.Button(ctrl, text="▶  PLAY", fg=C_GREEN,
                                   command=self._toggle_play,
                                   state=tk.DISABLED, **B)
        self._btn_play.pack(side=tk.LEFT, padx=4, pady=4)

        self._btn_restart = tk.Button(ctrl, text="⏮  RESTART", fg=C_DIM,
                                      command=self._restart,
                                      state=tk.DISABLED, **B)
        self._btn_restart.pack(side=tk.LEFT, padx=4, pady=4)

        self._mode_lbl = tk.Label(ctrl, text="NO VIDEO", bg=C_PANEL,
                                  fg=C_FAINT, font=("Courier", 8))
        self._mode_lbl.pack(side=tk.LEFT, padx=10)

        self._time_lbl = tk.Label(ctrl, text="0:00 / 0:00", bg=C_PANEL,
                                  fg=C_DIM, font=("Courier", 8))
        self._time_lbl.pack(side=tk.RIGHT, padx=8)

        self._progress = ttk.Scale(ctrl, from_=0, to=1000,
                                   orient=tk.HORIZONTAL, command=self._on_seek)
        self._progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=8, pady=4)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Horizontal.TScale", background=C_PANEL,
                        troughcolor=C_BORDER, slidercolor=C_BLUE)

        # Status bar
        status = tk.Frame(self.root, bg=C_BG)
        status.pack(fill=tk.X, padx=8, pady=(0, 6))
        self._status_lbl = tk.Label(status, text="Initialising…",
                                    bg=C_BG, fg=C_FAINT, font=("Courier", 8), anchor="w")
        self._status_lbl.pack(side=tk.LEFT)
        tk.Label(status, text="MOG2 + CLAHE + KALMAN  |  SMALL OBJECT DETECTION ON",
                 bg=C_BG, fg=C_FAINT, font=("Courier", 8)).pack(side=tk.RIGHT, padx=8)

    def _build_side(self, parent):
        def sec(text):
            f = tk.Frame(parent, bg=C_PANEL)
            f.pack(fill=tk.X, padx=6, pady=(8, 0))
            tk.Label(f, text=text, bg=C_PANEL, fg=C_FAINT,
                     font=("Courier", 7, "bold")).pack(anchor="w")
            tk.Frame(f, bg=C_BORDER, height=1).pack(fill=tk.X, pady=2)
            return f

        def kv(parent, key, var):
            row = tk.Frame(parent, bg=C_PANEL)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=key, bg=C_PANEL, fg=C_FAINT,
                     font=("Courier", 7), width=11, anchor="w").pack(side=tk.LEFT)
            tk.Label(row, textvariable=var, bg=C_PANEL, fg=C_DIM,
                     font=("Courier", 8), anchor="e").pack(side=tk.RIGHT)

        # Gauge
        gf = sec("FUSED THREAT")
        self._gauge_cv = tk.Canvas(gf, bg=C_PANEL, height=90, highlightthickness=0)
        self._gauge_cv.pack(fill=tk.X, padx=4, pady=4)
        self._gauge_var = tk.StringVar(value="0%")
        tk.Label(gf, textvariable=self._gauge_var, bg=C_PANEL, fg=C_GREEN,
                 font=("Courier", 18, "bold")).pack()
        tk.Label(gf, text="THREAT INDEX", bg=C_PANEL, fg=C_FAINT,
                 font=("Courier", 7)).pack()

        # Visual stats
        vsf = sec("VISUAL SENSOR")
        self._sv_tracks  = tk.StringVar(value="0")
        self._sv_conf    = tk.StringVar(value="—")
        self._sv_dist    = tk.StringVar(value="—")
        self._sv_fps     = tk.StringVar(value="—")
        self._sv_small   = tk.StringVar(value="0")
        kv(vsf, "TRACKS",    self._sv_tracks)
        kv(vsf, "BEST CONF", self._sv_conf)
        kv(vsf, "CLOSEST",   self._sv_dist)
        kv(vsf, "SMALL OBJ", self._sv_small)
        kv(vsf, "PROC FPS",  self._sv_fps)

        # Session
        sf = sec("SESSION")
        self._ss_frames  = tk.StringVar(value="0")
        self._ss_dets    = tk.StringVar(value="0")
        self._ss_alarms  = tk.StringVar(value="0")
        self._ss_peak    = tk.StringVar(value="0%")
        kv(sf, "FRAMES",     self._ss_frames)
        kv(sf, "DETECTIONS", self._ss_dets)
        kv(sf, "ALARMS",     self._ss_alarms)
        kv(sf, "PEAK THR",   self._ss_peak)

        # History
        hf = sec("SCORE HISTORY")
        self._hist_cv = tk.Canvas(hf, bg="#060d14", height=65, highlightthickness=0)
        self._hist_cv.pack(fill=tk.X, padx=4, pady=4)

        # Tracks list
        tf = sec("ACTIVE TRACKS")
        self._trk_text = tk.Text(tf, bg="#060d14", fg=C_DIM, font=("Courier", 7),
                                 height=7, relief="flat", bd=0, state=tk.DISABLED)
        self._trk_text.pack(fill=tk.X, padx=4, pady=4)

        # Event log — all logging goes here, not terminal
        lf = sec("EVENT LOG")
        log_frame = tk.Frame(lf, bg="#060d14")
        log_frame.pack(fill=tk.X, padx=4, pady=4)
        self._log_text = tk.Text(log_frame, bg="#060d14", fg=C_DIM,
                                 font=("Courier", 7), height=10,
                                 relief="flat", bd=0, state=tk.DISABLED, wrap=tk.CHAR)
        log_sb = tk.Scrollbar(log_frame, command=self._log_text.yview,
                              bg=C_BG, troughcolor=C_BG, width=6)
        self._log_text.configure(yscrollcommand=log_sb.set)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_sb.pack(side=tk.RIGHT, fill=tk.Y)

    # ── video loading ─────────────────────────────────────────────────────────

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open Drone Footage",
            filetypes=[("Video", "*.mp4 *.mov *.avi *.webm *.mkv *.m4v"),
                       ("All", "*.*")]
        )
        if path:
            self._load_video(path)

    def _open_demo(self):
        if not DEMO_VIDEO.exists():
            messagebox.showerror("Demo Missing",
                                 f"Demo video not found:\n{DEMO_VIDEO}")
            return
        self._load_video(str(DEMO_VIDEO), is_demo=True)

    def _load_video(self, path, is_demo=False):
        self._stop_playback()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._log(f"Cannot open: {Path(path).name}", level="ERROR")
            messagebox.showerror("Cannot Open", f"OpenCV failed to open:\n{path}")
            return
        with self._lock:
            if self._cap: self._cap.release()
            self._cap          = cap
            self._video_path   = path
            self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._frame_n      = 0
        self._detector.reset()
        self._score_history.clear()
        self._alarm_count    = 0
        self._peak_score     = 0.0
        self._total_detected = 0
        name = "DEMO" if is_demo else Path(path).name
        self._mode_lbl.config(text=name, fg=C_AMBER if is_demo else C_BLUE)
        self._log(f"Loaded: {name}  ({self._total_frames}fr @ {cap.get(cv2.CAP_PROP_FPS):.0f}fps)", level="INFO")
        self._btn_play.config(state=tk.NORMAL)
        self._btn_restart.config(state=tk.NORMAL)
        self._status_lbl.config(text=f"Ready: {name}")
        self._start_playback()

    # ── playback ──────────────────────────────────────────────────────────────

    def _start_playback(self):
        if self._playing or self._cap is None: return
        self._playing = True
        self._btn_play.config(text="⏸  PAUSE")
        self._thread = threading.Thread(target=self._play_loop, daemon=True)
        self._thread.start()

    def _stop_playback(self):
        self._playing = False
        if self._thread:
            self._thread.join(timeout=1.5)
            self._thread = None

    def _toggle_play(self):
        if self._playing:
            self._playing = False
            self._btn_play.config(text="▶  PLAY")
        else:
            if self._cap is None: return
            with self._lock:
                pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
            if pos >= self._total_frames - 2:
                self._restart()
            else:
                self._start_playback()

    def _restart(self):
        if self._cap is None: return
        self._stop_playback()
        with self._lock:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._frame_n = 0
        self._detector.reset()
        self._score_history.clear()
        self._start_playback()

    def _on_seek(self, val):
        if self._cap is None or self._total_frames == 0 or self._seeking: return
        ratio  = float(val) / 1000.0
        target = int(ratio * self._total_frames)
        was_playing = self._playing
        if was_playing: self._stop_playback()
        with self._lock:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target))
            self._frame_n = target
        self._detector.reset()
        if was_playing: self._start_playback()

    def _play_loop(self):
        while self._playing:
            t0 = time.perf_counter()
            with self._lock:
                if self._cap is None: break
                ret, frame = self._cap.read()
                pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
            if not ret:
                self._playing = False
                self.root.after(0, lambda: self._btn_play.config(text="▶  PLAY"))
                self._log("Playback complete.", level="INFO")
                break
            self._frame_n = pos
            result = self._detector.process(frame)
            with self._lock:
                self._result = result
            score = result["score"]
            if score > 0.60 and (not self._score_history or self._score_history[-1] < 0.60):
                self._alarm_count += 1
            if score > self._peak_score:
                self._peak_score = score
            if result["n_tracks"] > 0:
                self._total_detected += result["n_tracks"]
            self._score_history.append(score)
            with self._lock:
                fps_t = self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 25.0
            fps_t = max(1.0, fps_t)
            sleep = max(0.0, 1.0 / fps_t - (time.perf_counter() - t0))
            time.sleep(sleep)

    # ── UI tick ───────────────────────────────────────────────────────────────

    def _ui_tick(self):
        try: self._update_ui()
        except Exception: pass
        self.root.after(33, self._ui_tick)

    def _update_ui(self):
        # Uptime
        u  = time.time() - self._session_start
        hh, mm, ss = int(u//3600), int((u%3600)//60), int(u%60)
        self._uptime_lbl.config(text=f"UP: {hh:02d}:{mm:02d}:{ss:02d}")

        with self._lock:
            result  = self._result
            cap     = self._cap
            frame_n = self._frame_n
            total   = self._total_frames

        if result is None: return

        score  = result["score"]
        tracks = result["tracks"]
        fps    = result["fps"]
        frame  = result["frame"]
        warmup = result.get("warmup", False)

        # Threat banner
        lvl   = score_label(score)
        color = score_hex(score)
        txt   = "▶ AIRSPACE CLEAR" if lvl == "CLEAR" else f"⚠  {lvl} THREAT"
        self._threat_lbl.config(text=txt, fg=color)
        self._fps_lbl.config(text=f"FPS: {fps:.1f}")
        self._alarm_lbl.config(text=f"ALARMS: {self._alarm_count}")
        self._peak_lbl.config(text=f"PEAK: {int(self._peak_score*100)}%")

        # Gauge + score var
        self._gauge_var.set(f"{int(score*100)}%")
        self._draw_gauge(score, color)

        # Stats
        self._sv_tracks.set(str(len(tracks)))
        self._sv_fps.set(f"{fps:.1f}")
        small_count = sum(1 for t in tracks if t.get("small"))
        self._sv_small.set(str(small_count))
        if tracks:
            self._sv_conf.set(f"{int(max(t['confidence'] for t in tracks)*100)}%")
            self._sv_dist.set(f"{min(t['distance_m'] for t in tracks):.0f}m")
        else:
            self._sv_conf.set("—"); self._sv_dist.set("—")
        self._ss_frames.set(str(frame_n))
        self._ss_dets.set(str(self._total_detected))
        self._ss_alarms.set(str(self._alarm_count))
        self._ss_peak.set(f"{int(self._peak_score*100)}%")

        # Tracks text
        self._trk_text.config(state=tk.NORMAL)
        self._trk_text.delete("1.0", tk.END)
        if warmup:
            self._trk_text.insert(tk.END, "  Building background model…\n")
        elif not tracks:
            self._trk_text.insert(tk.END, "  No active tracks\n")
        else:
            for t in tracks[:7]:
                flag = " [S]" if t.get("small") else ""
                line = (f"  T-{t['id']:02d}  "
                        f"conf:{int(t['confidence']*100)}%  "
                        f"~{t['distance_m']:.0f}m  "
                        f"{int(t['threat']*100)}%thr{flag}\n")
                self._trk_text.insert(tk.END, line)
        self._trk_text.config(state=tk.DISABLED)

        # Alarm events → log panel
        if score > 0.60 and tracks and not warmup:
            ts = time.strftime("%H:%M:%S")
            self._log(f"{ts}  {lvl}: {len(tracks)} track(s), score {score:.3f}", level=lvl)

        # Draw frame
        self._display_frame(frame)
        self._draw_history(color)

        # Progress
        if total > 0 and cap is not None:
            self._seeking = True
            self._progress.set(frame_n / total * 1000)
            self._seeking = False
            fps_v = cap.get(cv2.CAP_PROP_FPS) or 25
            self._time_lbl.config(text=f"{self._fmt(frame_n/fps_v)} / {self._fmt(total/fps_v)}")

    def _display_frame(self, frame: np.ndarray):
        self._canvas.delete("placeholder")
        cw = max(100, self._display_w)
        ch = max(60,  self._display_h)
        fh, fw = frame.shape[:2]
        scale = min(cw/fw, ch/fh)
        nw, nh = int(fw*scale), int(fh*scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img     = ImageTk.PhotoImage(Image.fromarray(rgb))
        ox, oy  = (cw-nw)//2, (ch-nh)//2
        self._canvas.delete("frame")
        self._canvas.create_image(ox, oy, anchor="nw", image=img, tags="frame")
        self._canvas._img = img

    def _draw_gauge(self, score, color):
        c = self._gauge_cv
        c.delete("all")
        cw = max(20, c.winfo_width() or 180)
        ch = max(20, c.winfo_height() or 90)
        cx, cy, r = cw//2, ch-8, min(cw//2-6, ch-14)
        c.create_arc(cx-r, cy-r, cx+r, cy+r, start=0, extent=180,
                     style="arc", outline=C_BORDER, width=8)
        zones = [(0,.10,C_GREEN),(.10,.35,"#88ff00"),(.35,.60,C_AMBER),
                 (.60,.80,C_ORANGE),(.80,1.0,C_RED)]
        for lo, hi, zc in zones:
            c.create_arc(cx-r, cy-r, cx+r, cy+r,
                         start=180-lo*180, extent=-(hi-lo)*180,
                         style="arc", outline=zc+"44", width=8)
        if score > 0:
            c.create_arc(cx-r, cy-r, cx+r, cy+r,
                         start=180, extent=-(score*180),
                         style="arc", outline=color, width=8)
        a = math.radians(180 - score*180)
        nx, ny = cx+(r-12)*math.cos(a), cy-(r-12)*math.sin(a)
        c.create_line(cx, cy, int(nx), int(ny), fill="white", width=2)
        c.create_oval(cx-3, cy-3, cx+3, cy+3, fill="white", outline="")

    def _draw_history(self, color):
        c = self._hist_cv
        c.delete("all")
        cw = max(20, c.winfo_width() or 180)
        ch = max(20, c.winfo_height() or 65)
        pl, pr, pt, pb = 4, 4, 4, 14
        iw, ih = cw-pl-pr, ch-pt-pb
        for v in [.25,.50,.75,1.0]:
            y = pt+ih*(1-v)
            c.create_line(pl, y, pl+iw, y, fill=C_BORDER)
        data = list(self._score_history)
        if len(data) < 2: return
        pts = [(pl+i/(len(data)-1)*iw, pt+ih*(1-v)) for i,v in enumerate(data)]
        poly = [pl, pt+ih] + [coord for p in pts for coord in p] + [pl+iw, pt+ih]
        c.create_polygon(poly, fill=color+"22", outline="")
        flat = [coord for p in pts for coord in p]
        c.create_line(flat, fill=color, width=1.5, smooth=True)
        lx, ly = pts[-1]
        c.create_oval(lx-3, ly-3, lx+3, ly+3, fill=color, outline="")

    # ── log (goes to UI panel ONLY, not terminal) ─────────────────────────────

    def _log(self, msg, level="INFO"):
        colours = {"INFO": C_DIM, "WARN": C_AMBER, "ERROR": C_RED,
                   "CLEAR": C_GREEN, "LOW": "#88ff00", "MEDIUM": C_AMBER,
                   "HIGH": C_ORANGE, "CRITICAL": C_RED}
        col = colours.get(level, C_DIM)
        ts  = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {level:<8}  {msg}"
        self._log_entries.appendleft(entry)
        try:
            self._log_text.config(state=tk.NORMAL)
            self._log_text.insert("1.0", entry + "\n")
            lines = int(self._log_text.index("end-1c").split(".")[0])
            if lines > 120:
                self._log_text.delete(f"{100}.0", tk.END)
            self._log_text.tag_add(f"c{id(entry)}", "1.0", "1.end")
            self._log_text.tag_config(f"c{id(entry)}", foreground=col)
            self._log_text.config(state=tk.DISABLED)
        except Exception:
            pass

    @staticmethod
    def _fmt(secs):
        s = int(max(0, secs))
        return f"{s//60}:{s%60:02d}"

    def on_close(self):
        self._playing = False
        with self._lock:
            if self._cap: self._cap.release()
        self.root.destroy()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None)
    args = parser.parse_args()

    root = tk.Tk()
    root.minsize(920, 640)

    # Remove the default tkinter icon (avoids missing-bitmap errors on Windows)
    try: root.iconbitmap(default="")
    except Exception: pass

    app  = DragonShieldViewer(root, initial_video=args.video)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
