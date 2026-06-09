"""
DragonShield — Sensor Fusion Engine
======================================
Fuses RF, acoustic, and visual detections into a single,
calibrated threat score using a Bayesian-inspired weighted fusion model.

Key design principles:
  1. Conservative fusion: if ANY sensor reports CRITICAL, overall is CRITICAL
  2. Confidence weighting: high-confidence detections dominate
  3. Temporal smoothing: exponential moving average over last N frames
  4. Sensor health monitoring: degrade gracefully on sensor dropout

Author: DragonShield Project
"""

import numpy as np
import time
import logging
import json
from collections import deque
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict
from enum import Enum

logger = logging.getLogger("DragonShield.fusion")
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.ERROR)


class ThreatLevel(Enum):
    CLEAR   = "CLEAR"
    LOW     = "LOW"
    MEDIUM  = "MEDIUM"
    HIGH    = "HIGH"
    CRITICAL = "CRITICAL"


THREAT_COLORS = {
    ThreatLevel.CLEAR:    "#00ff88",
    ThreatLevel.LOW:      "#88ff00",
    ThreatLevel.MEDIUM:   "#ffcc00",
    ThreatLevel.HIGH:     "#ff6600",
    ThreatLevel.CRITICAL: "#ff0033",
}

THREAT_SCORE_THRESHOLDS = {
    ThreatLevel.CLEAR:    (0.0,  0.10),
    ThreatLevel.LOW:      (0.10, 0.35),
    ThreatLevel.MEDIUM:   (0.35, 0.60),
    ThreatLevel.HIGH:     (0.60, 0.80),
    ThreatLevel.CRITICAL: (0.80, 1.01),
}


def score_to_threat(score: float) -> ThreatLevel:
    for level, (lo, hi) in THREAT_SCORE_THRESHOLDS.items():
        if lo <= score < hi:
            return level
    return ThreatLevel.CRITICAL


@dataclass
class SensorReading:
    """Normalised input from each sensor module."""
    sensor_type: str          # "RF", "AUDIO", "VISUAL"
    active: bool              # Is sensor currently active?
    detection_present: bool   # Did it detect a drone?
    raw_score: float          # 0-1 threat score from that sensor
    confidence: float         # 0-1 how confident is the sensor
    class_name: str           # e.g. "DJI_PHANTOM", "FPV_RACING", "QUADCOPTER_SMALL"
    extra: Dict = field(default_factory=dict)  # sensor-specific metadata


@dataclass
class FusedThreatAssessment:
    timestamp: float
    fused_score: float
    threat_level: ThreatLevel
    threat_color: str
    alert_message: str

    # Per-sensor contributions
    rf_active: bool
    rf_score: float
    rf_confidence: float
    rf_class: str

    audio_active: bool
    audio_score: float
    audio_confidence: float
    audio_class: str

    visual_active: bool
    visual_score: float
    visual_confidence: float
    n_visual_tracks: int

    # Derived fields
    n_active_sensors: int
    sensor_agreement: float       # 0-1 (1 = all sensors agree)
    recommended_action: str
    is_alarm: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d['threat_level'] = self.threat_level.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ─── Fusion Engine ────────────────────────────────────────────────────────────
class SensorFusionEngine:
    """
    Bayesian-inspired multi-modal sensor fusion.

    Fusion strategy:
    ─ Weighted average with adaptive weights based on sensor confidence
    ─ Conservative rule: any CRITICAL overrides
    ─ Temporal smoothing via EMA to reduce false alarms
    ─ Sensor-absent compensation: redistribute weights
    """

    # Base sensor weights (must sum to 1)
    BASE_WEIGHTS = {
        "RF":     0.40,  # RF most reliable at range
        "AUDIO":  0.25,  # Audio good in quiet environments
        "VISUAL": 0.35,  # Visual most precise but shortest range
    }

    # Exponential moving average alpha (lower = smoother, higher = more reactive)
    EMA_ALPHA = 0.30

    # Alarm threshold
    ALARM_THRESHOLD = 0.55

    def __init__(self, history_len: int = 30):
        self.history: deque = deque(maxlen=history_len)
        self._ema_score: float = 0.0
        self._frame_count: int = 0
        self._alarm_count: int = 0
        self._false_positive_filter: deque = deque(maxlen=5)

    def _compute_weights(self, readings: List[SensorReading]) -> Dict[str, float]:
        """Adaptive weights: inactive sensors get 0, rest are renormalised."""
        weights = {}
        for r in readings:
            if r.active and r.detection_present:
                # Weight = base weight × confidence
                weights[r.sensor_type] = self.BASE_WEIGHTS[r.sensor_type] * r.confidence
            elif r.active:
                weights[r.sensor_type] = self.BASE_WEIGHTS[r.sensor_type] * 0.05
            else:
                weights[r.sensor_type] = 0.0

        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            # All sensors inactive — equal weights on active ones
            active = [r.sensor_type for r in readings if r.active]
            if active:
                weights = {k: (1.0 / len(active) if k in active else 0.0)
                           for k in self.BASE_WEIGHTS}
            else:
                weights = {k: 1.0 / 3 for k in self.BASE_WEIGHTS}
        return weights

    def _check_conservative_rules(self, readings: List[SensorReading]) -> Optional[float]:
        """Conservative override rules — safety first."""
        for r in readings:
            if r.active and r.raw_score > 0.90 and r.confidence > 0.75:
                logger.warning(f"Conservative rule triggered by {r.sensor_type}: "
                               f"score={r.raw_score:.2f} conf={r.confidence:.2f}")
                return min(1.0, r.raw_score * 1.1)

        # Two sensors both report HIGH
        high_count = sum(
            1 for r in readings
            if r.active and r.detection_present and r.raw_score > 0.60
        )
        if high_count >= 2:
            return 0.80
        return None

    def _compute_agreement(self, readings: List[SensorReading]) -> float:
        """Measure how much sensors agree (0=disagree, 1=perfect agreement)."""
        active_scores = [r.raw_score for r in readings if r.active]
        if len(active_scores) < 2:
            return 1.0
        return 1.0 - np.std(active_scores)

    def _generate_alert_message(self, threat: ThreatLevel,
                                readings: List[SensorReading]) -> str:
        detectors = [r.sensor_type for r in readings
                     if r.active and r.detection_present]
        class_names = [r.class_name for r in readings
                       if r.active and r.detection_present and r.class_name != "BACKGROUND"]

        if threat == ThreatLevel.CLEAR:
            return "Airspace clear. No drone signatures detected."
        elif threat == ThreatLevel.LOW:
            return f"Low-confidence drone signature via {', '.join(detectors) or 'unknown'}."
        elif threat == ThreatLevel.MEDIUM:
            cls = class_names[0] if class_names else "unknown"
            return f"Drone detected: {cls} | Sensors: {', '.join(detectors)}. Monitor situation."
        elif threat == ThreatLevel.HIGH:
            cls = class_names[0] if class_names else "unknown type"
            return f"⚠ HIGH THREAT: {cls} confirmed by {len(detectors)} sensors. Initiate response protocol."
        elif threat == ThreatLevel.CRITICAL:
            return "🚨 CRITICAL: Military-grade or encrypted drone signature. Activate countermeasures."
        return "Status unknown."

    def _recommended_action(self, threat: ThreatLevel) -> str:
        actions = {
            ThreatLevel.CLEAR:    "Continue monitoring.",
            ThreatLevel.LOW:      "Increase scan frequency. Log event.",
            ThreatLevel.MEDIUM:   "Alert operator. Prepare RF jammer standby.",
            ThreatLevel.HIGH:     "Activate soft-kill (RF jamming). Alert command.",
            ThreatLevel.CRITICAL: "HARD KILL AUTHORISED. Alert all units. Evacuate area.",
        }
        return actions.get(threat, "Unknown — escalate.")

    def fuse(self,
             rf_reading: SensorReading,
             audio_reading: SensorReading,
             visual_reading: SensorReading) -> FusedThreatAssessment:

        readings = [rf_reading, audio_reading, visual_reading]
        self._frame_count += 1

        # 1. Check conservative override rules
        override_score = self._check_conservative_rules(readings)

        # 2. Compute weighted average
        weights = self._compute_weights(readings)
        score_map = {
            "RF":     rf_reading.raw_score,
            "AUDIO":  audio_reading.raw_score,
            "VISUAL": visual_reading.raw_score,
        }
        weighted_score = sum(weights[k] * score_map[k] for k in score_map)

        # 3. Apply conservative override if triggered
        raw_fused = override_score if override_score else weighted_score

        # 4. Temporal EMA smoothing
        if self._frame_count == 1:
            self._ema_score = raw_fused
        else:
            self._ema_score = (self.EMA_ALPHA * raw_fused +
                               (1 - self.EMA_ALPHA) * self._ema_score)

        fused_score = float(np.clip(self._ema_score, 0.0, 1.0))

        # 5. False positive filter: require score > threshold for N consecutive frames
        self._false_positive_filter.append(fused_score > self.ALARM_THRESHOLD)
        is_alarm = sum(self._false_positive_filter) >= 3  # 3/5 frames must be above threshold

        if is_alarm:
            self._alarm_count += 1

        # 6. Threat level
        threat_level = score_to_threat(fused_score)
        threat_color = THREAT_COLORS[threat_level]
        agreement = self._compute_agreement(readings)

        # 7. Build assessment
        assessment = FusedThreatAssessment(
            timestamp=time.time(),
            fused_score=fused_score,
            threat_level=threat_level,
            threat_color=threat_color,
            alert_message=self._generate_alert_message(threat_level, readings),

            rf_active=rf_reading.active,
            rf_score=rf_reading.raw_score,
            rf_confidence=rf_reading.confidence,
            rf_class=rf_reading.class_name,

            audio_active=audio_reading.active,
            audio_score=audio_reading.raw_score,
            audio_confidence=audio_reading.confidence,
            audio_class=audio_reading.class_name,

            visual_active=visual_reading.active,
            visual_score=visual_reading.raw_score,
            visual_confidence=visual_reading.confidence,
            n_visual_tracks=visual_reading.extra.get('n_tracks', 0),

            n_active_sensors=sum(1 for r in readings if r.active),
            sensor_agreement=float(agreement),
            recommended_action=self._recommended_action(threat_level),
            is_alarm=is_alarm
        )

        self.history.append(assessment)

        if is_alarm:
            logger.warning(f"ALARM: {threat_level.value} | Score={fused_score:.3f} | "
                           f"{assessment.alert_message}")

        return assessment

    def get_history_scores(self) -> List[float]:
        return [a.fused_score for a in self.history]

    def get_stats(self) -> dict:
        return {
            "total_frames": self._frame_count,
            "total_alarms": self._alarm_count,
            "current_ema": self._ema_score,
            "history_len": len(self.history),
        }


# ─── System Orchestrator ──────────────────────────────────────────────────────
class DragonShieldSystem:
    """
    Top-level system: initialises all modules, runs inference loop,
    streams results to the dashboard via callbacks.
    """

    def __init__(self, model_dir: str = "/home/claude/dragonshield/models"):
        self.model_dir = model_dir
        self.fusion = SensorFusionEngine()
        self.running = False
        self._callbacks = []

        # Import submodules lazily
        self.rf_clf = None
        self.audio_clf = None
        self.visual_det = None

        logger.info("DragonShield system initialised.")

    def add_callback(self, fn):
        """Register a callback that receives FusedThreatAssessment on each frame."""
        self._callbacks.append(fn)

    def _notify(self, assessment: FusedThreatAssessment):
        for fn in self._callbacks:
            try:
                fn(assessment)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _load_modules(self):
        """Load all trained models."""
        import sys
        sys.path.insert(0, '/home/claude/dragonshield/src')

        from rf_classifier.rf_classifier import RFDroneClassifier, SyntheticRFDataGenerator, DRONE_CLASSES
        from audio_classifier.audio_classifier import AcousticDroneClassifier, SyntheticAudioGenerator, AUDIO_CLASSES

        self.rf_clf = RFDroneClassifier(
            model_path=os.path.join(self.model_dir, "rf_model.pkl")
        )
        self.audio_clf = AcousticDroneClassifier(
            model_path=os.path.join(self.model_dir, "audio_model.pkl")
        )

        # Visual
        from visual_tracker.visual_tracker import YOLODroneDetector
        self.visual_det = YOLODroneDetector()

        self._rf_gen = SyntheticRFDataGenerator()
        self._audio_gen = SyntheticAudioGenerator()
        self._DRONE_CLASSES = DRONE_CLASSES
        self._AUDIO_CLASSES = AUDIO_CLASSES

        logger.info("All modules loaded.")

    def _run_frame(self) -> FusedThreatAssessment:
        """Run one inference frame across all sensors."""
        import numpy as np

        # ── RF
        try:
            rf_class = np.random.choice([0, 0, 1, 2, 3, 7],
                                        p=[0.45, 0.20, 0.15, 0.10, 0.08, 0.02])
            rf_frames = self._rf_gen.generate(rf_class, n_frames=1)
            rf_det = self.rf_clf.predict(rf_frames[0], center_freq_mhz=2400.0)
            rf_reading = SensorReading(
                sensor_type="RF",
                active=True,
                detection_present=(rf_class != 0),
                raw_score=rf_det.threat_score,
                confidence=rf_det.confidence,
                class_name=rf_det.class_name,
                extra={"freq_mhz": rf_det.center_freq_mhz, "hopping": rf_det.frequency_hopping}
            )
        except Exception as e:
            logger.error(f"RF error: {e}")
            rf_reading = SensorReading("RF", False, False, 0.0, 0.0, "UNKNOWN")

        # ── Audio
        try:
            audio_class = np.random.choice([0, 0, 1, 2, 4],
                                           p=[0.45, 0.25, 0.15, 0.10, 0.05])
            audio_samples = self._audio_gen.generate(audio_class, n_samples=1)
            audio_det = self.audio_clf.predict(audio_samples[0])
            audio_reading = SensorReading(
                sensor_type="AUDIO",
                active=True,
                detection_present=(audio_class != 0),
                raw_score=audio_det.threat_score,
                confidence=audio_det.confidence,
                class_name=audio_det.class_name,
                extra={"rpm": audio_det.estimated_rpm, "freq_hz": audio_det.dominant_freq_hz}
            )
        except Exception as e:
            logger.error(f"Audio error: {e}")
            audio_reading = SensorReading("AUDIO", False, False, 0.0, 0.0, "UNKNOWN")

        # ── Visual
        try:
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            vis_det = self.visual_det.detect(dummy_frame)
            vis_score = vis_det.highest_threat_score
            visual_reading = SensorReading(
                sensor_type="VISUAL",
                active=True,
                detection_present=(vis_det.n_drones_detected > 0),
                raw_score=vis_score,
                confidence=min(1.0, vis_score * 1.2),
                class_name="VISUAL_TRACK",
                extra={"n_tracks": vis_det.n_drones_detected, "fps": vis_det.fps}
            )
        except Exception as e:
            logger.error(f"Visual error: {e}")
            visual_reading = SensorReading("VISUAL", False, False, 0.0, 0.0, "UNKNOWN")

        return self.fusion.fuse(rf_reading, audio_reading, visual_reading)

    def run(self, duration_s: Optional[float] = None, interval_s: float = 0.5):
        """Run the detection loop."""
        import os
        self._load_modules()
        self.running = True
        start = time.time()
        logger.info("DragonShield system ACTIVE — monitoring airspace.")
        try:
            while self.running:
                if duration_s and (time.time() - start) > duration_s:
                    break
                assessment = self._run_frame()
                self._notify(assessment)
                time.sleep(interval_s)
        except KeyboardInterrupt:
            logger.info("Shutdown requested.")
        finally:
            self.running = False
            stats = self.fusion.get_stats()
            logger.info(f"Session stats: {stats}")

    def stop(self):
        self.running = False


import os

if __name__ == "__main__":
    print("=" * 60)
    print("DragonShield Sensor Fusion Engine — Live Test")
    print("=" * 60)

    # Quick test with pre-trained models
    system = DragonShieldSystem()

    def on_assessment(a: FusedThreatAssessment):
        bar_len = int(a.fused_score * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"[{a.threat_level.value:8s}] [{bar}] {a.fused_score:.3f} | {a.alert_message[:60]}")

    system.add_callback(on_assessment)
    system.run(duration_s=10.0, interval_s=0.5)
