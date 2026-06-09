"""
DragonShield — RF Signal Classifier
====================================
Classifies drone types by their RF emission fingerprints using
Software Defined Radio (RTL-SDR dongle) input or simulated IQ data.

Hardware: RTL-SDR Blog V3 (~₹2000) on 2.4 GHz / 5.8 GHz bands
Detects: DJI Phantom/Mini/FPV, generic 2.4GHz RC drones, FPV 5.8GHz video

Author: DragonShield Project
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import time
import logging
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RF_Classifier")

# ─── Drone Class Labels ────────────────────────────────────────────────────────
DRONE_CLASSES = {
    0: "BACKGROUND",           # No drone / ambient RF
    1: "DJI_PHANTOM",          # DJI Phantom series (2.4 GHz OcuSync)
    2: "DJI_MINI",             # DJI Mini series (2.4/5.8 GHz)
    3: "DJI_FPV",              # DJI FPV (5.8 GHz video + 2.4 GHz control)
    4: "GENERIC_RC_2400",      # Generic RC 2.4 GHz (Syma, Eachine, etc.)
    5: "FPV_VIDEO_5800",       # FPV analog video transmitter 5.8 GHz
    6: "WIFI_DRONE",           # WiFi-controlled drone (2.4/5 GHz 802.11)
    7: "MILITARY_SUSPECT",     # Unknown encrypted / frequency-hopping
}

THREAT_LEVELS = {
    0: ("LOW",     0.05),   # Background
    1: ("MEDIUM",  0.55),   # Commercial DJI
    2: ("MEDIUM",  0.60),
    3: ("HIGH",    0.75),   # DJI FPV — agile
    4: ("LOW",     0.20),   # Generic toy
    5: ("HIGH",    0.70),   # FPV analog — likely custom/modified
    6: ("MEDIUM",  0.45),
    7: ("CRITICAL",0.95),   # Unknown — treat as hostile
}


@dataclass
class RFDetection:
    drone_class: int
    class_name: str
    confidence: float
    threat_level: str
    threat_score: float
    center_freq_mhz: float
    bandwidth_mhz: float
    signal_strength_dbm: float
    frequency_hopping: bool
    timestamp: float


# ─── Feature Extraction ────────────────────────────────────────────────────────
class RFFeatureExtractor:
    """
    Extracts discriminative features from IQ sample buffers.
    Features cover spectrum shape, hopping behaviour, modulation hints.
    """

    def __init__(self, sample_rate: float = 2.048e6, fft_size: int = 1024):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.freq_axis = fftfreq(fft_size, 1 / sample_rate)

    def compute_psd(self, iq_samples: np.ndarray) -> np.ndarray:
        """Power Spectral Density via Welch's method."""
        f, psd = signal.welch(
            iq_samples, fs=self.sample_rate,
            nperseg=self.fft_size, return_onesided=False
        )
        return np.abs(psd)

    def extract_features(self, iq_samples: np.ndarray) -> np.ndarray:
        """
        Returns a 1D feature vector:
        [spectral_centroid, spectral_spread, spectral_flatness,
         peak_power, peak_freq_bin, spectral_rolloff,
         energy_bands x8, hopping_score, burst_rate,
         cyclostationary_features x4]
        """
        if len(iq_samples) < self.fft_size:
            iq_samples = np.pad(iq_samples, (0, self.fft_size - len(iq_samples)))

        iq_samples = iq_samples[:self.fft_size]
        psd = self.compute_psd(iq_samples)
        psd_norm = psd / (np.sum(psd) + 1e-10)

        # ── Spectral moments
        freqs_pos = np.abs(self.freq_axis)
        centroid = np.sum(freqs_pos * psd_norm)
        spread = np.sqrt(np.sum(((freqs_pos - centroid) ** 2) * psd_norm))
        flatness = np.exp(np.mean(np.log(psd + 1e-10))) / (np.mean(psd) + 1e-10)

        # ── Peak features
        peak_idx = np.argmax(psd)
        peak_power = 10 * np.log10(psd[peak_idx] + 1e-10)
        peak_freq_norm = peak_idx / self.fft_size

        # ── Spectral rolloff (95% energy)
        cumsum = np.cumsum(psd_norm)
        rolloff_idx = np.searchsorted(cumsum, 0.95)
        rolloff = rolloff_idx / self.fft_size

        # ── Sub-band energy (divide spectrum into 8 bands)
        band_size = self.fft_size // 8
        band_energies = [
            np.sum(psd[i * band_size:(i + 1) * band_size])
            for i in range(8)
        ]
        band_energies = np.array(band_energies)
        band_energies /= (np.sum(band_energies) + 1e-10)

        # ── Frequency hopping score (variance of peak locations across chunks)
        chunk_size = len(iq_samples) // 4
        chunk_peaks = []
        for i in range(4):
            chunk = iq_samples[i * chunk_size:(i + 1) * chunk_size]
            if len(chunk) > 0:
                chunk_fft = np.abs(fft(chunk))
                chunk_peaks.append(np.argmax(chunk_fft))
        hopping_score = np.std(chunk_peaks) / self.fft_size if chunk_peaks else 0.0

        # ── Burst rate (zero-crossing rate as proxy for modulation)
        real_part = np.real(iq_samples)
        zero_crossings = np.sum(np.diff(np.sign(real_part)) != 0)
        burst_rate = zero_crossings / len(real_part)

        # ── Cyclostationary features (autocorrelation at lag offsets)
        autocorr = np.correlate(real_part[:256], real_part[:256], mode='full')
        autocorr_norm = autocorr / (np.max(np.abs(autocorr)) + 1e-10)
        mid = len(autocorr) // 2
        cyclo_features = autocorr_norm[[mid + 10, mid + 20, mid + 40, mid + 80]]

        feature_vector = np.concatenate([
            [centroid, spread, flatness, peak_power, peak_freq_norm, rolloff],
            band_energies,
            [hopping_score, burst_rate],
            cyclo_features
        ])

        return feature_vector.astype(np.float32)


# ─── Synthetic Data Generator (for training without hardware) ──────────────────
class SyntheticRFDataGenerator:
    """
    Generates physically plausible synthetic IQ samples for each drone class.
    Used for training when real RF captures aren't available.
    Replace with real RTL-SDR captures for production accuracy.
    """

    def __init__(self, sample_rate: float = 2.048e6, n_samples: int = 1024):
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.t = np.arange(n_samples) / sample_rate

    def _add_noise(self, signal_arr: np.ndarray, snr_db: float = 15.0) -> np.ndarray:
        snr_linear = 10 ** (snr_db / 10)
        signal_power = np.mean(np.abs(signal_arr) ** 2)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(signal_arr)) + 1j * np.random.randn(len(signal_arr))
        )
        return signal_arr + noise

    def generate(self, drone_class: int, n_frames: int = 200) -> np.ndarray:
        """Returns (n_frames, n_samples) complex IQ array."""
        frames = []
        for _ in range(n_frames):
            if drone_class == 0:  # Background
                iq = 0.01 * (np.random.randn(self.n_samples) + 1j * np.random.randn(self.n_samples))

            elif drone_class == 1:  # DJI Phantom — OcuSync 2.4 GHz OFDM
                f_center = 2.4e6 * np.random.choice([0.8, 0.9, 1.0, 1.1])
                iq = np.exp(2j * np.pi * f_center * self.t)
                # OFDM-like multiple subcarriers
                for sub in np.linspace(-0.3e6, 0.3e6, 12):
                    iq += 0.3 * np.exp(2j * np.pi * sub * self.t + 1j * np.random.rand())
                iq = self._add_noise(iq, snr_db=np.random.uniform(12, 20))

            elif drone_class == 2:  # DJI Mini — narrower BW
                f_center = 2.4e6 * np.random.choice([0.85, 0.95, 1.05])
                iq = np.exp(2j * np.pi * f_center * self.t)
                for sub in np.linspace(-0.15e6, 0.15e6, 8):
                    iq += 0.25 * np.exp(2j * np.pi * sub * self.t + 1j * np.random.rand())
                iq = self._add_noise(iq, snr_db=np.random.uniform(10, 18))

            elif drone_class == 3:  # DJI FPV — frequency hopping
                hop_freqs = np.random.choice(
                    np.linspace(5.7e6, 5.9e6, 20), size=4, replace=False
                )
                iq = np.zeros(self.n_samples, dtype=complex)
                hop_size = self.n_samples // 4
                for i, f in enumerate(hop_freqs):
                    sl = slice(i * hop_size, (i + 1) * hop_size)
                    iq[sl] = np.exp(2j * np.pi * f * self.t[sl])
                iq = self._add_noise(iq, snr_db=np.random.uniform(14, 22))

            elif drone_class == 4:  # Generic RC 2.4 GHz — FHSS
                iq = np.zeros(self.n_samples, dtype=complex)
                n_hops = 8
                hop_size = self.n_samples // n_hops
                for i in range(n_hops):
                    f = np.random.uniform(2.4e6, 2.48e6)
                    sl = slice(i * hop_size, (i + 1) * hop_size)
                    iq[sl] = 0.7 * np.exp(2j * np.pi * f * self.t[sl])
                iq = self._add_noise(iq, snr_db=np.random.uniform(8, 16))

            elif drone_class == 5:  # FPV 5.8 GHz analog video
                f_center = 5.8e6 * np.random.choice([0.96, 0.98, 1.0, 1.02, 1.04])
                iq = np.exp(2j * np.pi * f_center * self.t)
                # Analog FM-like wideband
                fm_mod = np.cumsum(np.random.randn(self.n_samples)) * 0.1e6
                iq *= np.exp(2j * np.pi * fm_mod * self.t)
                iq = self._add_noise(iq, snr_db=np.random.uniform(10, 18))

            elif drone_class == 6:  # WiFi drone
                f_center = 2.437e6
                iq = np.zeros(self.n_samples, dtype=complex)
                # Bursty 802.11 pattern
                burst_starts = np.random.randint(0, self.n_samples - 100, size=5)
                for bs in burst_starts:
                    burst_len = np.random.randint(50, 150)
                    be = min(bs + burst_len, self.n_samples)
                    iq[bs:be] = np.exp(2j * np.pi * f_center * self.t[bs:be])
                iq = self._add_noise(iq, snr_db=np.random.uniform(8, 15))

            elif drone_class == 7:  # Military suspect — encrypted, aperiodic
                iq = np.random.randn(self.n_samples) + 1j * np.random.randn(self.n_samples)
                iq /= np.max(np.abs(iq))
                iq = self._add_noise(iq, snr_db=np.random.uniform(5, 12))

            else:
                iq = np.zeros(self.n_samples, dtype=complex)

            frames.append(iq)

        return np.array(frames)


# ─── Classifier ───────────────────────────────────────────────────────────────
class RFDroneClassifier:
    def __init__(self, model_path: Optional[str] = None):
        self.extractor = RFFeatureExtractor()
        self.scaler = StandardScaler()
        self.model = GradientBoostingClassifier(
            n_estimators=200, max_depth=5,
            learning_rate=0.08, subsample=0.85,
            random_state=42
        )
        self.is_trained = False
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        elif model_path:
            # Auto-train if model file doesn't exist
            logger.warning(f"RF model not found at {model_path}. Auto-training on synthetic data...")
            self.train(n_per_class=200)
            self.save(model_path)

    def _build_dataset(self, n_per_class: int = 300):
        generator = SyntheticRFDataGenerator()
        X, y = [], []
        for cls in DRONE_CLASSES:
            logger.info(f"  Generating class {cls}: {DRONE_CLASSES[cls]}")
            frames = generator.generate(cls, n_frames=n_per_class)
            for frame in frames:
                features = self.extractor.extract_features(frame)
                X.append(features)
                y.append(cls)
        return np.array(X), np.array(y)

    def train(self, n_per_class: int = 300, model_save_path: Optional[str] = None):
        logger.info("Building RF training dataset...")
        X, y = self._build_dataset(n_per_class)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        logger.info("Training Gradient Boosting classifier...")
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred,
                                       target_names=list(DRONE_CLASSES.values()))
        logger.info(f"\nClassification Report:\n{report}")

        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        logger.info(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        if model_save_path:
            self.save(model_save_path)
        return {"accuracy": float(cv_scores.mean()), "report": report}

    def predict(self, iq_samples: np.ndarray,
                center_freq_mhz: float = 2400.0,
                signal_strength_dbm: float = -70.0) -> RFDetection:
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        features = self.extractor.extract_features(iq_samples)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        class_idx = self.model.predict(features_scaled)[0]
        proba = self.model.predict_proba(features_scaled)[0]
        confidence = float(proba[class_idx])

        threat_level, threat_score = THREAT_LEVELS.get(class_idx, ("UNKNOWN", 0.5))

        # Check for frequency hopping
        hopping_score = features[14]  # index in feature vector
        freq_hopping = hopping_score > 0.15

        return RFDetection(
            drone_class=int(class_idx),
            class_name=DRONE_CLASSES[class_idx],
            confidence=confidence,
            threat_level=threat_level,
            threat_score=float(threat_score),
            center_freq_mhz=center_freq_mhz,
            bandwidth_mhz=20.0,
            signal_strength_dbm=signal_strength_dbm,
            frequency_hopping=freq_hopping,
            timestamp=time.time()
        )

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
        logger.info(f"RF model saved to {path}")

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = True
        logger.info(f"RF model loaded from {path}")


# ─── Live SDR Interface ────────────────────────────────────────────────────────
class RTLSDRInterface:
    """
    Interface for RTL-SDR hardware.
    Requires: pip install pyrtlsdr  +  RTL-SDR dongle plugged in USB.
    Falls back to simulation mode if hardware absent.
    """

    def __init__(self, center_freq: float = 2.4e9, sample_rate: float = 2.048e6,
                 gain: float = 40.0):
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.gain = gain
        self.sdr = None
        self._init_hardware()

    def _init_hardware(self):
        try:
            from rtlsdr import RtlSdr
            self.sdr = RtlSdr()
            self.sdr.center_freq = self.center_freq
            self.sdr.sample_rate = self.sample_rate
            self.sdr.gain = self.gain
            logger.info("RTL-SDR hardware initialized successfully.")
        except Exception as e:
            logger.warning(f"RTL-SDR not available ({e}). Running in simulation mode.")
            self.sdr = None

    def read_samples(self, n_samples: int = 1024) -> np.ndarray:
        if self.sdr:
            return self.sdr.read_samples(n_samples)
        # Simulation fallback
        generator = SyntheticRFDataGenerator(self.sample_rate, n_samples)
        fake_class = np.random.choice([0, 0, 0, 1, 2, 4], p=[0.5, 0.15, 0.1, 0.1, 0.1, 0.05])
        frames = generator.generate(fake_class, n_frames=1)
        return frames[0]

    def sweep_bands(self, frequencies_mhz: list = [2400, 2450, 5800, 5850]) -> dict:
        """Sweep multiple frequencies and return signal strengths."""
        results = {}
        for freq_mhz in frequencies_mhz:
            if self.sdr:
                self.sdr.center_freq = freq_mhz * 1e6
                samples = self.sdr.read_samples(1024)
            else:
                samples = self.read_samples(1024)
            power_dbm = 10 * np.log10(np.mean(np.abs(samples) ** 2) + 1e-10) + 30
            results[freq_mhz] = float(power_dbm)
        return results

    def close(self):
        if self.sdr:
            self.sdr.close()


if __name__ == "__main__":
    print("=" * 60)
    print("DragonShield RF Classifier — Training & Test")
    print("=" * 60)

    clf = RFDroneClassifier()
    results = clf.train(n_per_class=200, model_save_path="/home/claude/dragonshield/models/rf_model.pkl")
    print(f"\nFinal CV Accuracy: {results['accuracy']:.1%}")

    # Test with simulated DJI signal
    gen = SyntheticRFDataGenerator()
    test_frame = gen.generate(drone_class=1, n_frames=1)[0]
    detection = clf.predict(test_frame, center_freq_mhz=2400.0, signal_strength_dbm=-65.0)
    print(f"\nTest Detection:")
    print(f"  Class    : {detection.class_name}")
    print(f"  Confidence: {detection.confidence:.1%}")
    print(f"  Threat   : {detection.threat_level} ({detection.threat_score:.2f})")
    print(f"  Hopping  : {detection.frequency_hopping}")
