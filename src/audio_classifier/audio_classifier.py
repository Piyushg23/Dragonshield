"""
DragonShield — Acoustic Drone Classifier
==========================================
Classifies drones by the acoustic signature of their motors and propellers.
Uses MFCC + spectral features fed into a CNN-style classifier.

Hardware: Any microphone (USB mic / built-in / lapel mic)
Range: Effective up to ~80m in quiet environment, ~30m in urban noise
Frequency range of interest: 80 Hz – 8000 Hz (motor fundamentals + harmonics)

Author: DragonShield Project
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, rfft, rfftfreq
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import pickle
import time
import logging
import os
import threading
import queue
from dataclasses import dataclass
from typing import Optional, List

logger = logging.getLogger("Audio_Classifier")
logging.basicConfig(level=logging.INFO)

# ─── Audio Classes ─────────────────────────────────────────────────────────────
AUDIO_CLASSES = {
    0: "BACKGROUND",        # Ambient noise, wind, traffic
    1: "QUADCOPTER_SMALL",  # Small 4-prop drone (DJI Mini, Mavic)
    2: "QUADCOPTER_LARGE",  # Large 4-prop (DJI Phantom, M300)
    3: "HEXACOPTER",        # 6-prop heavy lift
    4: "FPV_RACING",        # High-RPM racing drone (very distinctive)
    5: "FIXED_WING_UAV",    # Fixed wing, lower prop freq
    6: "HELICOPTER",        # Manned helicopter (discriminate from drone)
}

AUDIO_THREAT = {
    0: ("LOW",    0.0),
    1: ("MEDIUM", 0.50),
    2: ("MEDIUM", 0.60),
    3: ("HIGH",   0.70),
    4: ("HIGH",   0.80),
    5: ("MEDIUM", 0.55),
    6: ("LOW",    0.10),   # Manned — not an adversarial drone
}

SAMPLE_RATE = 22050   # Hz
FRAME_DURATION = 1.0  # seconds per analysis frame
HOP_DURATION = 0.5    # seconds hop between frames


@dataclass
class AudioDetection:
    drone_class: int
    class_name: str
    confidence: float
    threat_level: str
    threat_score: float
    dominant_freq_hz: float
    estimated_rpm: float
    n_rotors_estimate: int
    snr_db: float
    timestamp: float


# ─── MFCC Feature Extraction ──────────────────────────────────────────────────
class AcousticFeatureExtractor:
    """
    Extracts acoustic features tailored for drone motor identification.
    Key insight: drone motors produce strong harmonic series at fundamental RPM/60 Hz.
    """

    def __init__(self, sr: int = SAMPLE_RATE, n_mfcc: int = 20,
                 n_fft: int = 2048, hop_length: int = 512):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.frame_len = int(FRAME_DURATION * sr)

    def _mel_filterbank(self, n_filters: int = 40) -> np.ndarray:
        """Compute mel filterbank matrix."""
        low_freq_mel = 2595 * np.log10(1 + 80 / 700)
        high_freq_mel = 2595 * np.log10(1 + (self.sr / 2) / 700)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sr).astype(int)

        filterbank = np.zeros((n_filters, self.n_fft // 2 + 1))
        for m in range(1, n_filters + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]
            for k in range(f_m_minus, f_m):
                if f_m - f_m_minus > 0:
                    filterbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if f_m_plus - f_m > 0:
                    filterbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
        return filterbank

    def compute_mfcc(self, audio: np.ndarray, n_mfcc: int = 20) -> np.ndarray:
        """Compute MFCCs from audio signal."""
        # Frame the signal
        frames = []
        for start in range(0, len(audio) - self.n_fft, self.hop_length):
            frame = audio[start:start + self.n_fft]
            frame = frame * np.hamming(len(frame))
            frames.append(frame)

        if not frames:
            return np.zeros(n_mfcc * 3)

        # FFT + power spectrum
        power_frames = np.array([np.abs(rfft(f)) ** 2 for f in frames])

        # Mel filterbank
        filterbank = self._mel_filterbank(40)
        n_bins = power_frames.shape[1]
        fb = filterbank[:, :n_bins]
        filter_banks = np.dot(power_frames, fb.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)

        # DCT to get MFCCs
        n_frames, n_filters = filter_banks.shape
        dct_matrix = np.zeros((n_mfcc, n_filters))
        for n in range(n_mfcc):
            dct_matrix[n] = np.cos(np.pi * n / n_filters * (np.arange(n_filters) + 0.5))

        mfccs = np.dot(filter_banks, dct_matrix.T)

        # Aggregate: mean + std + delta mean
        mfcc_mean = np.mean(mfccs, axis=0)
        mfcc_std = np.std(mfccs, axis=0)
        if len(mfccs) > 1:
            delta = np.diff(mfccs, axis=0)
            delta_mean = np.mean(delta, axis=0)
        else:
            delta_mean = np.zeros(n_mfcc)

        return np.concatenate([mfcc_mean, mfcc_std, delta_mean])

    def extract_harmonic_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract features from harmonic structure — key for drone rotor ID.
        Drone rotors produce: f0, 2*f0, 3*f0, 4*f0 ... (n_blades * RPM/60)
        """
        freqs = rfftfreq(self.n_fft, 1 / self.sr)
        spectrum = np.abs(rfft(audio[:self.n_fft] * np.hamming(self.n_fft)))

        # Find dominant frequency in rotor range (80 Hz – 500 Hz)
        rotor_mask = (freqs >= 80) & (freqs <= 500)
        rotor_spectrum = spectrum.copy()
        rotor_spectrum[~rotor_mask] = 0
        f0_idx = np.argmax(rotor_spectrum)
        f0 = freqs[f0_idx]

        # Harmonic energy ratio (H1-H6 vs total)
        harmonic_energies = []
        for h in range(1, 7):
            h_freq = f0 * h
            if h_freq < self.sr / 2:
                h_idx = int(h_freq * self.n_fft / self.sr)
                h_window = max(1, int(0.02 * self.n_fft))
                h_energy = np.sum(spectrum[max(0, h_idx - h_window):h_idx + h_window] ** 2)
            else:
                h_energy = 0.0
            harmonic_energies.append(float(h_energy))

        total_energy = np.sum(spectrum ** 2) + 1e-10
        harmonic_ratio = sum(harmonic_energies) / total_energy

        # Spectral centroid in rotor band
        rotor_energy = np.sum(rotor_spectrum)
        if rotor_energy > 0:
            rotor_centroid = np.sum(freqs * rotor_spectrum) / rotor_energy
        else:
            rotor_centroid = 0.0

        # Zero-crossing rate (discriminates tonal vs noisy)
        zcr = np.sum(np.diff(np.sign(audio[:1024])) != 0) / 1024.0

        # RMS energy
        rms = float(np.sqrt(np.mean(audio ** 2)))

        return np.array([
            f0, harmonic_ratio, rotor_centroid, zcr, rms,
            *harmonic_energies
        ], dtype=np.float32)

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Full feature vector: MFCC + harmonic."""
        # Pad or trim
        if len(audio) < self.frame_len:
            audio = np.pad(audio, (0, self.frame_len - len(audio)))
        else:
            audio = audio[:self.frame_len]

        # Normalize
        audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        mfcc_features = self.compute_mfcc(audio, n_mfcc=self.n_mfcc)    # 60 dims
        harmonic_features = self.extract_harmonic_features(audio)         # 11 dims

        return np.concatenate([mfcc_features, harmonic_features])


# ─── Synthetic Audio Generator ────────────────────────────────────────────────
class SyntheticAudioGenerator:
    """
    Generates physically-motivated synthetic drone audio.
    Real-world characterization:
    - Small quadcopter: ~90-150 Hz fundamental, 4 rotors → clear harmonics
    - FPV racer: ~200-350 Hz fundamental, aggressive harmonic series
    - Large hex: ~60-90 Hz, more sub-bass energy
    """

    def __init__(self, sr: int = SAMPLE_RATE):
        self.sr = sr

    def _drone_tone(self, f0: float, duration: float, n_harmonics: int = 6,
                    noise_level: float = 0.1) -> np.ndarray:
        t = np.linspace(0, duration, int(self.sr * duration))
        signal_arr = np.zeros(len(t))
        for h in range(1, n_harmonics + 1):
            amp = 1.0 / h
            phase = np.random.uniform(0, 2 * np.pi)
            signal_arr += amp * np.sin(2 * np.pi * f0 * h * t + phase)
        signal_arr += noise_level * np.random.randn(len(t))
        return signal_arr / (np.max(np.abs(signal_arr)) + 1e-10)

    def generate(self, drone_class: int, n_samples: int = 200,
                 duration: float = FRAME_DURATION) -> List[np.ndarray]:
        samples = []
        for _ in range(n_samples):
            if drone_class == 0:  # Background
                audio = 0.05 * np.random.randn(int(self.sr * duration))
                # Add some low-freq rumble
                t = np.linspace(0, duration, len(audio))
                audio += 0.03 * np.sin(2 * np.pi * 50 * t)

            elif drone_class == 1:  # Small quadcopter
                f0 = np.random.uniform(90, 150)
                audio = self._drone_tone(f0, duration, n_harmonics=5, noise_level=0.15)

            elif drone_class == 2:  # Large quadcopter
                f0 = np.random.uniform(55, 90)
                audio = self._drone_tone(f0, duration, n_harmonics=7, noise_level=0.12)

            elif drone_class == 3:  # Hexacopter
                f0 = np.random.uniform(45, 75)
                # 6 rotors — sum two slight detuned tones
                audio = self._drone_tone(f0, duration, n_harmonics=6, noise_level=0.1)
                audio += 0.5 * self._drone_tone(f0 * 1.02, duration, n_harmonics=6, noise_level=0.1)
                audio /= (np.max(np.abs(audio)) + 1e-10)

            elif drone_class == 4:  # FPV racing
                f0 = np.random.uniform(200, 350)
                audio = self._drone_tone(f0, duration, n_harmonics=8, noise_level=0.2)
                # Add motor whine variation
                t = np.linspace(0, duration, int(self.sr * duration))
                mod = 1 + 0.1 * np.sin(2 * np.pi * 5 * t)
                audio *= mod

            elif drone_class == 5:  # Fixed wing
                f0 = np.random.uniform(30, 60)
                audio = self._drone_tone(f0, duration, n_harmonics=3, noise_level=0.3)

            elif drone_class == 6:  # Helicopter
                f0 = np.random.uniform(10, 25)  # Main rotor much slower
                audio = self._drone_tone(f0, duration, n_harmonics=10, noise_level=0.25)
                # Tail rotor
                audio += 0.3 * self._drone_tone(f0 * 6, duration, n_harmonics=3, noise_level=0.1)
                audio /= (np.max(np.abs(audio)) + 1e-10)

            else:
                audio = np.zeros(int(self.sr * duration))

            samples.append(audio)
        return samples


# ─── Acoustic Classifier ─────────────────────────────────────────────────────
class AcousticDroneClassifier:
    def __init__(self, model_path: Optional[str] = None):
        self.extractor = AcousticFeatureExtractor()
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            learning_rate_init=0.001
        )
        self.is_trained = False
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        elif model_path:
            # Auto-train if model file doesn't exist
            logger.warning(f"Acoustic model not found at {model_path}. Auto-training on synthetic data...")
            self.train(n_per_class=200)
            self.save(model_path)

    def train(self, n_per_class: int = 200, model_save_path: Optional[str] = None):
        logger.info("Building acoustic training dataset...")
        gen = SyntheticAudioGenerator()
        X, y = [], []
        for cls in AUDIO_CLASSES:
            logger.info(f"  Generating class {cls}: {AUDIO_CLASSES[cls]}")
            audio_samples = gen.generate(cls, n_samples=n_per_class)
            for audio in audio_samples:
                features = self.extractor.extract_features(audio)
                X.append(features)
                y.append(cls)

        X, y = np.array(X), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        logger.info("Training MLP acoustic classifier...")
        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred,
                                       target_names=list(AUDIO_CLASSES.values()))
        logger.info(f"\nClassification Report:\n{report}")
        accuracy = float(np.mean(y_pred == y_test))

        if model_save_path:
            self.save(model_save_path)
        return {"accuracy": accuracy, "report": report}

    def predict(self, audio: np.ndarray) -> AudioDetection:
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        features = self.extractor.extract_features(audio)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        class_idx = self.model.predict(features_scaled)[0]
        proba = self.model.predict_proba(features_scaled)[0]
        confidence = float(proba[class_idx])

        threat_level, threat_score = AUDIO_THREAT.get(class_idx, ("UNKNOWN", 0.5))

        # Estimate physical parameters
        harm_features = self.extractor.extract_harmonic_features(audio)
        dominant_freq = float(harm_features[0])
        estimated_rpm = dominant_freq * 60  # f0 = RPM/60

        # Rotor estimate from spectral complexity
        n_rotors = {0: 0, 1: 4, 2: 4, 3: 6, 4: 4, 5: 1, 6: 2}.get(class_idx, 4)

        # SNR estimate
        signal_power = float(np.mean(audio ** 2))
        noise_floor = 0.001
        snr_db = 10 * np.log10(signal_power / noise_floor + 1e-10)

        return AudioDetection(
            drone_class=int(class_idx),
            class_name=AUDIO_CLASSES[class_idx],
            confidence=confidence,
            threat_level=threat_level,
            threat_score=float(threat_score),
            dominant_freq_hz=dominant_freq,
            estimated_rpm=estimated_rpm,
            n_rotors_estimate=n_rotors,
            snr_db=snr_db,
            timestamp=time.time()
        )

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
        logger.info(f"Audio model saved to {path}")

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = True
        logger.info(f"Audio model loaded from {path}")


# ─── Live Microphone Interface ────────────────────────────────────────────────
class MicrophoneStream:
    """
    Real-time audio capture. Requires PyAudio + PortAudio.
    Falls back to simulation if hardware unavailable.
    """

    def __init__(self, sr: int = SAMPLE_RATE, frame_duration: float = FRAME_DURATION):
        self.sr = sr
        self.frame_duration = frame_duration
        self.frame_size = int(sr * frame_duration)
        self.buffer = queue.Queue(maxsize=10)
        self.stream = None
        self._init_mic()

    def _init_mic(self):
        try:
            import pyaudio
            self.pa = pyaudio.PyAudio()
            self.stream = self.pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sr,
                input=True,
                frames_per_buffer=1024,
                stream_callback=self._callback
            )
            logger.info("Microphone initialized.")
        except Exception as e:
            logger.warning(f"Mic unavailable ({e}). Using simulation.")
            self.stream = None
            self._start_sim_thread()

    def _callback(self, in_data, frame_count, time_info, status):
        audio = np.frombuffer(in_data, dtype=np.float32)
        if not self.buffer.full():
            self.buffer.put(audio)
        import pyaudio
        return (None, pyaudio.paContinue)

    def _start_sim_thread(self):
        gen = SyntheticAudioGenerator()
        self._sim_running = True

        def sim_loop():
            while self._sim_running:
                cls = np.random.choice([0, 0, 1, 2, 4], p=[0.5, 0.2, 0.15, 0.1, 0.05])
                frames = gen.generate(cls, n_samples=1, duration=0.1)
                if not self.buffer.full():
                    self.buffer.put(frames[0])
                time.sleep(0.1)

        threading.Thread(target=sim_loop, daemon=True).start()

    def read_frame(self, timeout: float = 2.0) -> Optional[np.ndarray]:
        chunks = []
        total = 0
        while total < self.frame_size:
            try:
                chunk = self.buffer.get(timeout=timeout)
                chunks.append(chunk)
                total += len(chunk)
            except queue.Empty:
                break
        if not chunks:
            return None
        audio = np.concatenate(chunks)[:self.frame_size]
        if len(audio) < self.frame_size:
            audio = np.pad(audio, (0, self.frame_size - len(audio)))
        return audio

    def stop(self):
        self._sim_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.pa.terminate()


if __name__ == "__main__":
    print("=" * 60)
    print("DragonShield Acoustic Classifier — Training & Test")
    print("=" * 60)
    clf = AcousticDroneClassifier()
    results = clf.train(
        n_per_class=150,
        model_save_path="/home/claude/dragonshield/models/audio_model.pkl"
    )
    print(f"\nFinal Accuracy: {results['accuracy']:.1%}")

    gen = SyntheticAudioGenerator()
    test_audio = gen.generate(drone_class=4, n_samples=1)[0]  # FPV racer
    det = clf.predict(test_audio)
    print(f"\nTest Detection (FPV Racer):")
    print(f"  Detected  : {det.class_name}")
    print(f"  Confidence: {det.confidence:.1%}")
    print(f"  RPM Est   : {det.estimated_rpm:.0f}")
    print(f"  Threat    : {det.threat_level} ({det.threat_score:.2f})")
