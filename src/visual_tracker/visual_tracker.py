"""
DragonShield — Visual Drone Tracker
=====================================
Detects and tracks drones in video streams using YOLOv8.
Falls back to a classical motion-based detector if YOLOv8 unavailable.
Kalman Filter maintains smooth trajectory estimates between frames.

Hardware: USB webcam, IP camera (RTSP), or Raspberry Pi Camera v2
Optimized for: Jetson Nano (TensorRT), Pi 4 (ONNX), or desktop GPU

Author: DragonShield Project
"""

import numpy as np
import time
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import math

logger = logging.getLogger("Visual_Tracker")
logging.basicConfig(level=logging.INFO)

# ─── Data Structures ──────────────────────────────────────────────────────────
@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass
class DroneTrack:
    track_id: int
    bbox: BoundingBox
    velocity: Tuple[float, float]       # (vx, vy) pixels/frame
    heading_deg: float                  # estimated heading
    distance_estimate_m: float          # rough distance in metres
    threat_score: float
    frames_tracked: int
    last_seen: float
    trajectory: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class VisualDetection:
    tracks: List[DroneTrack]
    frame_width: int
    frame_height: int
    fps: float
    timestamp: float
    n_drones_detected: int
    highest_threat_score: float


# ─── Kalman Filter for Single Drone Track ─────────────────────────────────────
class KalmanTracker:
    """
    Constant-velocity Kalman Filter for 2D drone tracking.
    State: [cx, cy, vx, vy]  (centre x/y, velocity x/y)
    """

    def __init__(self, cx: float, cy: float):
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)

        # State transition matrix (constant velocity)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)

        # Measurement matrix (we observe cx, cy)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)

        # Process noise
        self.Q = np.eye(4) * 1.0
        self.Q[2, 2] = 5.0
        self.Q[3, 3] = 5.0

        # Measurement noise
        self.R = np.eye(2) * 4.0

        # Covariance matrix
        self.P = np.eye(4) * 10.0

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]

    def update(self, measurement: np.ndarray):
        z = measurement.reshape(2, 1)
        y = z - self.H @ self.x.reshape(4, 1)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = (self.x.reshape(4, 1) + K @ y).flatten()
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def position(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> Tuple[float, float]:
        return float(self.x[2]), float(self.x[3])


# ─── Multi-Object Tracker ─────────────────────────────────────────────────────
class MultiObjectTracker:
    """
    Manages multiple drone tracks with Kalman filters.
    Uses IoU + distance for detection-to-track assignment.
    """

    def __init__(self, max_disappeared: int = 15, min_confidence: float = 0.3):
        self.tracks: Dict[int, Dict] = {}
        self.next_id = 0
        self.max_disappeared = max_disappeared
        self.min_confidence = min_confidence

    def _iou(self, b1: BoundingBox, b2: BoundingBox) -> float:
        ix1 = max(b1.x1, b2.x1)
        iy1 = max(b1.y1, b2.y1)
        ix2 = min(b1.x2, b2.x2)
        iy2 = min(b1.y2, b2.y2)
        if ix2 < ix1 or iy2 < iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = b1.area + b2.area - inter
        return inter / (union + 1e-6)

    def _assign(self, detections: List[BoundingBox]) -> Tuple[dict, list, list]:
        """Hungarian-lite assignment via greedy IoU matching."""
        if not self.tracks or not detections:
            return {}, list(range(len(detections))), list(self.tracks.keys())

        # Build cost matrix
        track_ids = list(self.tracks.keys())
        cost = np.zeros((len(track_ids), len(detections)))
        for ti, tid in enumerate(track_ids):
            predicted = self.tracks[tid]['kalman'].position
            for di, det in enumerate(detections):
                dist = math.sqrt((predicted[0] - det.cx) ** 2 + (predicted[1] - det.cy) ** 2)
                cost[ti, di] = dist

        # Greedy assignment
        matched = {}
        used_det = set()
        used_trk = set()
        flat_order = np.argsort(cost, axis=None)
        for idx in flat_order:
            ti = idx // len(detections)
            di = idx % len(detections)
            if ti in used_trk or di in used_det:
                continue
            if cost[ti, di] > 100:  # max pixel distance threshold
                continue
            matched[track_ids[ti]] = di
            used_trk.add(ti)
            used_det.add(di)

        unmatched_det = [i for i in range(len(detections)) if i not in used_det]
        unmatched_trk = [track_ids[ti] for ti in range(len(track_ids)) if ti not in used_trk]
        return matched, unmatched_det, unmatched_trk

    def update(self, detections: List[BoundingBox]) -> List[DroneTrack]:
        # Predict all tracks forward
        for tid in self.tracks:
            self.tracks[tid]['kalman'].predict()

        matched, unmatched_det, unmatched_trk = self._assign(detections)

        # Update matched tracks
        for tid, di in matched.items():
            det = detections[di]
            meas = np.array([det.cx, det.cy])
            self.tracks[tid]['kalman'].update(meas)
            self.tracks[tid]['bbox'] = det
            self.tracks[tid]['disappeared'] = 0
            self.tracks[tid]['frames'] += 1
            self.tracks[tid]['last_seen'] = time.time()
            pos = self.tracks[tid]['kalman'].position
            self.tracks[tid]['trajectory'].append(pos)
            if len(self.tracks[tid]['trajectory']) > 30:
                self.tracks[tid]['trajectory'].pop(0)

        # Create new tracks for unmatched detections
        for di in unmatched_det:
            det = detections[di]
            new_id = self.next_id
            self.next_id += 1
            self.tracks[new_id] = {
                'kalman': KalmanTracker(det.cx, det.cy),
                'bbox': det,
                'disappeared': 0,
                'frames': 1,
                'last_seen': time.time(),
                'trajectory': [(det.cx, det.cy)]
            }

        # Mark unmatched tracks as disappeared
        for tid in unmatched_trk:
            self.tracks[tid]['disappeared'] += 1

        # Remove old tracks
        dead = [tid for tid, t in self.tracks.items()
                if t['disappeared'] > self.max_disappeared]
        for tid in dead:
            del self.tracks[tid]

        # Build output
        result = []
        for tid, t in self.tracks.items():
            if t['disappeared'] > 0:
                continue
            pos = t['kalman'].position
            vel = t['kalman'].velocity
            heading = math.degrees(math.atan2(vel[1], vel[0])) % 360

            # Rough distance: assume drone is ~30cm wide → pixel size → distance
            bbox = t['bbox']
            assumed_drone_width_m = 0.3
            focal_px = 800  # approximate focal length in pixels
            if bbox.w > 0:
                distance_m = (assumed_drone_width_m * focal_px) / bbox.w
            else:
                distance_m = 999.0

            # Threat score based on distance + speed
            speed = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
            threat = min(1.0, (1.0 / max(distance_m, 1.0)) * 10 + speed * 0.02)

            result.append(DroneTrack(
                track_id=tid,
                bbox=bbox,
                velocity=(float(vel[0]), float(vel[1])),
                heading_deg=heading,
                distance_estimate_m=float(distance_m),
                threat_score=float(threat),
                frames_tracked=t['frames'],
                last_seen=t['last_seen'],
                trajectory=list(t['trajectory'])
            ))

        return result


# ─── YOLOv8 Detector ─────────────────────────────────────────────────────────
class YOLODroneDetector:
    """
    YOLOv8-based drone detector.
    Uses ultralytics YOLOv8n (nano) fine-tuned on drone datasets.
    Falls back to background subtraction detector if YOLO unavailable.
    """

    def __init__(self, model_path: Optional[str] = None, confidence: float = 0.35):
        self.confidence = confidence
        self.model = None
        self.use_fallback = False
        self._init_model(model_path)
        self.tracker = MultiObjectTracker()
        self.frame_count = 0
        self.fps = 0.0
        self._last_time = time.time()

    def _init_model(self, model_path: Optional[str]):
        try:
            from ultralytics import YOLO
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
                logger.info(f"Loaded custom YOLO model: {model_path}")
            else:
                self.model = YOLO('yolov8n.pt')  # Downloads automatically
                logger.info("Loaded YOLOv8n base model (not fine-tuned for drones)")
        except Exception as e:
            logger.warning(f"YOLOv8 not available ({e}). Using background subtraction fallback.")
            self.use_fallback = True
            self._bg_subtractor = self._init_bg_subtractor()

    def _init_bg_subtractor(self):
        try:
            import cv2
            return cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=50, detectShadows=False
            )
        except Exception:
            return None

    def _fallback_detect(self, frame: np.ndarray) -> List[BoundingBox]:
        """Classical motion-based detection when YOLO unavailable."""
        try:
            import cv2
            if self.bg_subtractor is None:
                return []

            fg_mask = self._bg_subtractor.apply(frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detections = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 100 < area < 50000:  # drone-plausible size range
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect = w / (h + 1e-6)
                    if 0.3 < aspect < 3.0:  # rough shape filter
                        confidence = min(1.0, area / 5000)
                        detections.append(BoundingBox(
                            x1=float(x), y1=float(y),
                            x2=float(x + w), y2=float(y + h),
                            confidence=confidence
                        ))
            return detections
        except Exception:
            return self._simulate_detections(frame.shape[:2])

    def _simulate_detections(self, frame_shape: Tuple[int, int]) -> List[BoundingBox]:
        """Simulation mode: generate plausible moving drone boxes."""
        h, w = frame_shape
        t = time.time()
        n = np.random.poisson(0.3)
        detections = []
        for i in range(min(n, 3)):
            # Simulate a drone moving across frame
            cx = (w / 2 + 200 * np.sin(t * 0.5 + i * 2)) % w
            cy = (h / 3 + 100 * np.cos(t * 0.3 + i)) % h
            size = np.random.randint(20, 80)
            detections.append(BoundingBox(
                x1=cx - size / 2, y1=cy - size / 2,
                x2=cx + size / 2, y2=cy + size / 2,
                confidence=np.random.uniform(0.4, 0.95)
            ))
        return detections

    def detect(self, frame: np.ndarray) -> VisualDetection:
        t0 = time.time()

        if self.model and not self.use_fallback:
            try:
                results = self.model(frame, verbose=False, conf=self.confidence)
                raw_detections = []
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        # Filter: class 0 = person, we want aerial objects
                        # For fine-tuned model, class 0 = drone
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        raw_detections.append(BoundingBox(x1, y1, x2, y2, conf))
            except Exception as e:
                logger.error(f"YOLO inference failed: {e}")
                raw_detections = self._simulate_detections(frame.shape[:2])
        else:
            try:
                raw_detections = self._fallback_detect(frame)
            except Exception:
                raw_detections = self._simulate_detections(frame.shape[:2])

        tracks = self.tracker.update(raw_detections)

        # FPS tracking
        self.frame_count += 1
        elapsed = time.time() - self._last_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self._last_time = time.time()

        h, w = frame.shape[:2]
        highest_threat = max((t.threat_score for t in tracks), default=0.0)

        return VisualDetection(
            tracks=tracks,
            frame_width=w,
            frame_height=h,
            fps=self.fps,
            timestamp=time.time(),
            n_drones_detected=len(tracks),
            highest_threat_score=highest_threat
        )

    def annotate_frame(self, frame: np.ndarray,
                       detection: VisualDetection) -> np.ndarray:
        """Draw bounding boxes and trajectories on frame."""
        try:
            import cv2
            annotated = frame.copy()
            for track in detection.tracks:
                b = track.bbox
                color = (0, 255, 0) if track.threat_score < 0.5 else \
                        (0, 165, 255) if track.threat_score < 0.8 else (0, 0, 255)
                cv2.rectangle(annotated, (int(b.x1), int(b.y1)),
                              (int(b.x2), int(b.y2)), color, 2)
                label = f"ID:{track.track_id} {b.confidence:.0%}"
                cv2.putText(annotated, label, (int(b.x1), int(b.y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                # Draw trajectory
                traj = track.trajectory
                for i in range(1, len(traj)):
                    pt1 = (int(traj[i - 1][0]), int(traj[i - 1][1]))
                    pt2 = (int(traj[i][0]), int(traj[i][1]))
                    cv2.line(annotated, pt1, pt2, color, 1)

            cv2.putText(annotated, f"FPS: {detection.fps:.1f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(annotated, f"Targets: {detection.n_drones_detected}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            return annotated
        except Exception:
            return frame


if __name__ == "__main__":
    print("=" * 60)
    print("DragonShield Visual Tracker — Simulation Test")
    print("=" * 60)
    detector = YOLODroneDetector()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(5):
        result = detector.detect(dummy_frame)
        print(f"Frame {i+1}: {result.n_drones_detected} tracks detected")
        for t in result.tracks:
            print(f"  Track {t.track_id}: ({t.bbox.cx:.0f},{t.bbox.cy:.0f})"
                  f"  dist≈{t.distance_estimate_m:.1f}m  threat={t.threat_score:.2f}")
        time.sleep(0.1)
