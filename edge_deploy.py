"""
DragonShield — Edge Deployment & Hardware Interface Guide
==========================================================
This script handles:
  1. Model export to ONNX / TensorRT for edge acceleration
  2. Jetson Nano hardware-specific optimisations
  3. Raspberry Pi 4 deployment (CPU-optimised ONNX)
  4. GPIO alert output (buzzer, LED strip, relay for jammer trigger)
  5. System service setup (systemd auto-start on boot)

Run on Jetson Nano:
  python edge_deploy.py --device jetson --export-trt

Run on Raspberry Pi:
  python edge_deploy.py --device rpi --export-onnx

Author: DragonShield Project
"""

import sys
import os
import argparse
import logging
import json
import subprocess
import time
import pickle
import numpy as np

logger = logging.getLogger("EdgeDeploy")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
DEPLOY_DIR = os.path.join(os.path.dirname(__file__), 'deploy')
os.makedirs(DEPLOY_DIR, exist_ok=True)


# ─── ONNX Export ──────────────────────────────────────────────────────────────
def export_sklearn_to_onnx(model_path: str, output_path: str, n_features: int):
    """
    Export scikit-learn models (RF classifier, Audio MLP) to ONNX format.
    ONNX enables hardware-accelerated inference on both Jetson and RPi.
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import onnx

        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        model = data['model']
        scaler = data['scaler']

        # Export scaler
        scaler_onnx = convert_sklearn(scaler, initial_types=[
            ('float_input', FloatTensorType([None, n_features]))
        ])

        # Export classifier
        clf_onnx = convert_sklearn(model, initial_types=[
            ('float_input', FloatTensorType([None, n_features]))
        ])

        scaler_path = output_path.replace('.onnx', '_scaler.onnx')
        clf_path = output_path

        with open(scaler_path, 'wb') as f:
            f.write(scaler_onnx.SerializeToString())
        with open(clf_path, 'wb') as f:
            f.write(clf_onnx.SerializeToString())

        logger.info(f"ONNX exported: {clf_path}")
        return True

    except ImportError:
        logger.warning("skl2onnx not installed. Run: pip install skl2onnx onnx")
        logger.info("Falling back to pickle-based inference (slower but functional)")
        return False
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        return False


def export_yolo_to_onnx(model_path: str, output_path: str):
    """Export YOLOv8 to ONNX for edge deployment."""
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        model.export(format='onnx', dynamic=True, simplify=True)
        logger.info(f"YOLOv8 ONNX exported")
        return True
    except Exception as e:
        logger.error(f"YOLO ONNX export failed: {e}")
        return False


def export_yolo_to_tensorrt(model_path: str):
    """Export YOLOv8 to TensorRT for maximum Jetson Nano performance."""
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        model.export(format='engine', device=0, half=True)  # FP16 for Jetson
        logger.info("YOLOv8 TensorRT engine exported (FP16)")
        return True
    except Exception as e:
        logger.error(f"TensorRT export failed: {e}")
        logger.info("Make sure TensorRT is installed on Jetson Nano.")
        return False


# ─── Jetson Nano Optimisations ────────────────────────────────────────────────
def setup_jetson_performance():
    """
    Configure Jetson Nano for maximum performance mode.
    Run as sudo or within sudoers.
    """
    commands = [
        "sudo nvpmodel -m 0",               # MAXN power mode (10W)
        "sudo jetson_clocks",               # Max CPU/GPU/EMC clocks
        "sudo sh -c 'echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'",
    ]
    logger.info("Setting Jetson to MAXN performance mode...")
    for cmd in commands:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"  ✓ {cmd.split()[-1]}")
            else:
                logger.warning(f"  ✗ {cmd}: {result.stderr.strip()}")
        except Exception as e:
            logger.warning(f"  Skipped: {e}")


def check_jetson_stats() -> dict:
    """Read Jetson resource utilisation."""
    stats = {}
    try:
        # CPU usage
        with open('/proc/loadavg') as f:
            stats['cpu_load'] = float(f.read().split()[0])

        # GPU memory (tegrastats output)
        result = subprocess.run(['tegrastats', '--interval', '1', '--logfile', '/tmp/tg.log'],
                                capture_output=True, timeout=2)
        stats['tegrastats_available'] = True
    except Exception:
        stats['tegrastats_available'] = False

    # GPU temp
    try:
        with open('/sys/devices/virtual/thermal/thermal_zone1/temp') as f:
            stats['gpu_temp_c'] = int(f.read()) / 1000
    except Exception:
        stats['gpu_temp_c'] = None

    return stats


# ─── GPIO Alert Interface ─────────────────────────────────────────────────────
class GPIOAlertInterface:
    """
    Physical alert outputs for DragonShield.
    Requires: Raspberry Pi or Jetson Nano GPIO

    Wiring:
      Pin 18 → Buzzer (active, 5V)
      Pin 23 → Red LED (HIGH THREAT)
      Pin 24 → Amber LED (MEDIUM THREAT)
      Pin 25 → Green LED (CLEAR)
      Pin 16 → Relay trigger (jammer activation — HIGH = activate)
    """

    PINS = {
        'buzzer':       18,
        'led_red':      23,
        'led_amber':    24,
        'led_green':    25,
        'jammer_relay': 16,
    }

    def __init__(self, simulation: bool = False):
        self.simulation = simulation
        self.gpio = None
        self._init_gpio()

    def _init_gpio(self):
        if self.simulation:
            logger.info("GPIO: Simulation mode (no hardware)")
            return
        try:
            import RPi.GPIO as GPIO
            self.gpio = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            for name, pin in self.PINS.items():
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
            logger.info("GPIO initialised successfully.")
        except ImportError:
            logger.warning("RPi.GPIO not available. GPIO alerts disabled.")
            self.simulation = True
        except Exception as e:
            logger.warning(f"GPIO init failed: {e}. Using simulation.")
            self.simulation = True

    def _set_pin(self, pin: int, state: bool):
        if self.simulation:
            return
        if self.gpio:
            self.gpio.output(pin, self.gpio.HIGH if state else self.gpio.LOW)

    def set_threat(self, threat_level: str):
        """Set LED indicators based on threat level."""
        self._set_pin(self.PINS['led_green'],  threat_level == 'CLEAR')
        self._set_pin(self.PINS['led_amber'],  threat_level in ('LOW', 'MEDIUM'))
        self._set_pin(self.PINS['led_red'],    threat_level in ('HIGH', 'CRITICAL'))
        if threat_level == 'CRITICAL':
            self._pulse_buzzer(3)
        elif threat_level == 'HIGH':
            self._pulse_buzzer(1)

        if self.simulation:
            icons = {'CLEAR':'🟢','LOW':'🟡','MEDIUM':'🟠','HIGH':'🔴','CRITICAL':'🚨'}
            print(f"  GPIO: {icons.get(threat_level,'?')} {threat_level}")

    def activate_jammer(self, activate: bool):
        """
        Trigger RF jammer relay.
        WARNING: RF jamming is illegal in most jurisdictions without proper authority.
        This output is for use only by authorised defence personnel.
        """
        self._set_pin(self.PINS['jammer_relay'], activate)
        if activate:
            logger.warning("JAMMER RELAY ACTIVATED — Ensure proper authority.")

    def _pulse_buzzer(self, count: int, duration: float = 0.2):
        for _ in range(count):
            self._set_pin(self.PINS['buzzer'], True)
            time.sleep(duration)
            self._set_pin(self.PINS['buzzer'], False)
            time.sleep(duration)

    def cleanup(self):
        if self.gpio:
            self.gpio.cleanup()


# ─── MAVLink Integration ──────────────────────────────────────────────────────
class MAVLinkInterface:
    """
    Send threat alerts to a GCS (Ground Control Station) via MAVLink.
    Allows DragonShield to integrate with Mission Planner, QGroundControl, etc.

    pip install pymavlink
    """

    def __init__(self, connection_string: str = 'udp:127.0.0.1:14550'):
        self.connection_string = connection_string
        self.conn = None
        self._init_connection()

    def _init_connection(self):
        try:
            from pymavlink import mavutil
            self.conn = mavutil.mavlink_connection(self.connection_string)
            logger.info(f"MAVLink connected: {self.connection_string}")
        except ImportError:
            logger.warning("pymavlink not installed. MAVLink alerts disabled.")
        except Exception as e:
            logger.warning(f"MAVLink connection failed: {e}")

    def send_statustext(self, severity: int, text: str):
        """
        Send text alert to GCS.
        severity: 0=Emergency, 1=Alert, 2=Critical, 3=Error, 4=Warning, 5=Notice, 6=Info
        """
        if not self.conn:
            logger.info(f"MAVLink (sim): [{severity}] {text}")
            return
        try:
            self.conn.mav.statustext_send(severity, text.encode('utf-8')[:50])
        except Exception as e:
            logger.error(f"MAVLink send failed: {e}")

    def threat_to_mavlink(self, threat_level: str, message: str):
        """Convert DragonShield threat level to MAVLink severity."""
        severity_map = {
            'CLEAR':    6,   # INFO
            'LOW':      5,   # NOTICE
            'MEDIUM':   4,   # WARNING
            'HIGH':     2,   # CRITICAL
            'CRITICAL': 0,   # EMERGENCY
        }
        sev = severity_map.get(threat_level, 4)
        prefix = {'CLEAR':'','LOW':'[LOW]','MEDIUM':'[MED]','HIGH':'[HIGH]','CRITICAL':'[!CRIT!]'}
        self.send_statustext(sev, f"{prefix.get(threat_level,'')} {message}")


# ─── Systemd Service Generator ────────────────────────────────────────────────
def generate_systemd_service(user: str = 'pi', project_dir: str = '/home/pi/dragonshield') -> str:
    """Generate systemd service file for auto-start on boot."""
    service = f"""[Unit]
Description=DragonShield Counter-UAS Detection System
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
Restart=always
RestartSec=3
User={user}
WorkingDirectory={project_dir}
ExecStartPre=/bin/sleep 10
ExecStart=/usr/bin/python3 {project_dir}/src/dashboard/server.py
StandardOutput=journal
StandardError=journal
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
"""
    service_path = os.path.join(DEPLOY_DIR, 'dragonshield.service')
    with open(service_path, 'w') as f:
        f.write(service)

    logger.info(f"Systemd service written to {service_path}")
    logger.info("To install:")
    logger.info(f"  sudo cp {service_path} /etc/systemd/system/")
    logger.info("  sudo systemctl daemon-reload")
    logger.info("  sudo systemctl enable dragonshield")
    logger.info("  sudo systemctl start dragonshield")
    return service_path


# ─── Deployment Manifest ──────────────────────────────────────────────────────
def generate_deployment_manifest(device: str) -> dict:
    manifest = {
        "project": "DragonShield Counter-UAS System",
        "version": "1.0.0",
        "deployment_target": device,
        "deployment_date": time.strftime("%Y-%m-%d"),
        "models": {
            "rf_classifier": {
                "file": "models/rf_model.pkl",
                "onnx": f"deploy/rf_model.onnx",
                "features": 20,
                "classes": 8
            },
            "audio_classifier": {
                "file": "models/audio_model.pkl",
                "onnx": f"deploy/audio_model.onnx",
                "features": 71,
                "classes": 7
            },
            "visual_tracker": {
                "file": "yolov8n.pt",
                "trt_engine": "deploy/yolov8n.engine" if device == "jetson" else None,
                "onnx": "deploy/yolov8n.onnx" if device == "rpi" else None
            }
        },
        "hardware": {
            "compute": "NVIDIA Jetson Nano 4GB" if device == "jetson" else "Raspberry Pi 4 4GB",
            "rf": "RTL-SDR Blog V3",
            "audio": "USB omnidirectional microphone",
            "camera": "Raspberry Pi Camera v2 or USB webcam",
            "gpio_alerts": True,
            "mavlink": True
        },
        "performance_targets": {
            "end_to_end_latency_ms": 90 if device == "jetson" else 250,
            "update_rate_hz": 2,
            "false_alarm_rate": "<1 per hour in benign environment"
        },
        "network": {
            "dashboard_port": 8000,
            "mavlink_port": 14550,
            "websocket": "ws://localhost:8000/ws"
        }
    }

    path = os.path.join(DEPLOY_DIR, f'manifest_{device}.json')
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Deployment manifest: {path}")
    return manifest


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='DragonShield Edge Deployment')
    parser.add_argument('--device', choices=['jetson', 'rpi', 'desktop'],
                        default='desktop', help='Target deployment platform')
    parser.add_argument('--export-onnx', action='store_true')
    parser.add_argument('--export-trt', action='store_true')
    parser.add_argument('--generate-service', action='store_true')
    parser.add_argument('--gpio-test', action='store_true')
    parser.add_argument('--mavlink-test', action='store_true')
    args = parser.parse_args()

    logger.info(f"DragonShield Edge Deploy — Target: {args.device.upper()}")

    if args.device == 'jetson':
        setup_jetson_performance()

    if args.export_onnx:
        logger.info("Exporting models to ONNX...")
        export_sklearn_to_onnx(
            os.path.join(MODELS_DIR, 'rf_model.pkl'),
            os.path.join(DEPLOY_DIR, 'rf_model.onnx'),
            n_features=20
        )
        export_sklearn_to_onnx(
            os.path.join(MODELS_DIR, 'audio_model.pkl'),
            os.path.join(DEPLOY_DIR, 'audio_model.onnx'),
            n_features=71
        )
        export_yolo_to_onnx('yolov8n.pt', os.path.join(DEPLOY_DIR, 'yolov8n.onnx'))

    if args.export_trt and args.device == 'jetson':
        logger.info("Exporting YOLOv8 to TensorRT (FP16)...")
        export_yolo_to_tensorrt('yolov8n.pt')

    if args.generate_service:
        generate_systemd_service(
            user='pi' if args.device == 'rpi' else 'dragonshield',
            project_dir=os.path.abspath(os.path.dirname(__file__))
        )

    if args.gpio_test:
        logger.info("GPIO alert test sequence...")
        gpio = GPIOAlertInterface(simulation=True)
        for level in ['CLEAR', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
            logger.info(f"  Testing: {level}")
            gpio.set_threat(level)
            time.sleep(0.5)
        gpio.cleanup()

    if args.mavlink_test:
        mav = MAVLinkInterface('udp:127.0.0.1:14550')
        mav.threat_to_mavlink('HIGH', 'DJI Phantom detected on multiple sensors')

    manifest = generate_deployment_manifest(args.device)
    logger.info(f"\nDeployment manifest generated for {args.device.upper()}")
    logger.info(f"Next: uvicorn src/dashboard/server:app --host 0.0.0.0 --port 8000")


if __name__ == '__main__':
    main()
