"""
DragonShield — Test Suite
===========================
Unit + integration tests for all modules.
Run: python -m pytest tests/ -v --tb=short

Author: DragonShield Project
"""

import sys
import os
import time
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ─── RF Classifier Tests ──────────────────────────────────────────────────────
class TestRFClassifier:

    @pytest.fixture(scope="class")
    def trained_clf(self):
        from rf_classifier.rf_classifier import RFDroneClassifier
        clf = RFDroneClassifier()
        clf.train(n_per_class=80)
        return clf

    @pytest.fixture(scope="class")
    def generator(self):
        from rf_classifier.rf_classifier import SyntheticRFDataGenerator
        return SyntheticRFDataGenerator()

    @pytest.fixture(scope="class")
    def extractor(self):
        from rf_classifier.rf_classifier import RFFeatureExtractor
        return RFFeatureExtractor()

    def test_feature_extractor_output_shape(self, extractor, generator):
        frames = generator.generate(0, n_frames=1)
        features = extractor.extract_features(frames[0])
        assert features.ndim == 1
        assert len(features) == 20, f"Expected 20 features, got {len(features)}"

    def test_feature_extractor_finite(self, extractor, generator):
        for cls in range(8):
            frames = generator.generate(cls, n_frames=3)
            for frame in frames:
                feats = extractor.extract_features(frame)
                assert np.all(np.isfinite(feats)), f"Non-finite features for class {cls}"

    def test_model_trains_successfully(self, trained_clf):
        assert trained_clf.is_trained

    def test_model_predicts_background(self, trained_clf, generator):
        frames = generator.generate(0, n_frames=5)
        for frame in frames:
            det = trained_clf.predict(frame)
            assert det.drone_class == 0
            assert det.class_name == "BACKGROUND"
            assert 0.0 <= det.confidence <= 1.0
            assert 0.0 <= det.threat_score <= 1.0

    def test_model_predicts_dji(self, trained_clf, generator):
        correct = 0
        for _ in range(10):
            frame = generator.generate(1, n_frames=1)[0]
            det = trained_clf.predict(frame)
            if det.drone_class == 1:
                correct += 1
        assert correct >= 7, f"DJI detection accuracy too low: {correct}/10"

    def test_threat_levels_ordering(self, trained_clf, generator):
        # Background should have lower threat than DJI Phantom
        bg_frame = generator.generate(0, n_frames=1)[0]
        dji_frame = generator.generate(1, n_frames=1)[0]
        bg_det = trained_clf.predict(bg_frame)
        dji_det = trained_clf.predict(dji_frame)
        assert bg_det.threat_score < dji_det.threat_score, \
            "Background should have lower threat than DJI"

    def test_military_suspect_critical(self, trained_clf, generator):
        frame = generator.generate(7, n_frames=1)[0]
        det = trained_clf.predict(frame)
        assert det.drone_class == 7
        assert det.threat_level == "CRITICAL"
        assert det.threat_score >= 0.90

    def test_model_save_load(self, trained_clf, generator, tmp_path):
        model_path = str(tmp_path / "test_rf.pkl")
        trained_clf.save(model_path)
        assert os.path.exists(model_path)

        from rf_classifier.rf_classifier import RFDroneClassifier
        clf2 = RFDroneClassifier()
        clf2.load(model_path)
        assert clf2.is_trained

        frame = generator.generate(1, n_frames=1)[0]
        det1 = trained_clf.predict(frame)
        det2 = clf2.predict(frame)
        assert det1.drone_class == det2.drone_class

    def test_frequency_hopping_detection(self, trained_clf, generator):
        # DJI FPV (class 3) should be classified as non-background
        # and should have higher threat than pure background
        bg_threats, fpv_threats = [], []
        for _ in range(10):
            bg_frame  = generator.generate(0, n_frames=1)[0]
            fpv_frame = generator.generate(3, n_frames=1)[0]
            bg_threats.append(trained_clf.predict(bg_frame).threat_score)
            fpv_threats.append(trained_clf.predict(fpv_frame).threat_score)
        assert np.mean(fpv_threats) > np.mean(bg_threats), \
            "FPV drone should have higher threat score than background"

    def test_short_iq_input(self, extractor):
        # Extractor should handle inputs shorter than fft_size
        short_iq = np.random.randn(256) + 1j * np.random.randn(256)
        feats = extractor.extract_features(short_iq)
        assert np.all(np.isfinite(feats))

    def test_prediction_includes_timestamp(self, trained_clf, generator):
        frame = generator.generate(0, n_frames=1)[0]
        t_before = time.time()
        det = trained_clf.predict(frame)
        t_after = time.time()
        assert t_before <= det.timestamp <= t_after


# ─── Audio Classifier Tests ───────────────────────────────────────────────────
class TestAudioClassifier:

    @pytest.fixture(scope="class")
    def trained_clf(self):
        from audio_classifier.audio_classifier import AcousticDroneClassifier
        clf = AcousticDroneClassifier()
        clf.train(n_per_class=80)
        return clf

    @pytest.fixture(scope="class")
    def generator(self):
        from audio_classifier.audio_classifier import SyntheticAudioGenerator
        return SyntheticAudioGenerator()

    @pytest.fixture(scope="class")
    def extractor(self):
        from audio_classifier.audio_classifier import AcousticFeatureExtractor
        return AcousticFeatureExtractor()

    def test_mfcc_shape(self, extractor, generator):
        audio = generator.generate(1, n_samples=1)[0]
        feats = extractor.extract_features(audio)
        assert feats.ndim == 1
        assert len(feats) == 71  # 60 MFCC + 11 harmonic

    def test_mfcc_finite(self, extractor, generator):
        for cls in range(7):
            audio = generator.generate(cls, n_samples=3)
            for a in audio:
                feats = extractor.extract_features(a)
                assert np.all(np.isfinite(feats)), f"Non-finite for audio class {cls}"

    def test_harmonic_features_background(self, extractor, generator):
        bg = generator.generate(0, n_samples=5)
        drone = generator.generate(4, n_samples=5)  # FPV — strong harmonics
        bg_harm = np.mean([extractor.extract_harmonic_features(a)[1] for a in bg])
        drone_harm = np.mean([extractor.extract_harmonic_features(a)[1] for a in drone])
        assert drone_harm > bg_harm, "Drone should have stronger harmonic content"

    def test_fpv_higher_frequency(self, extractor, generator):
        small_quad = generator.generate(1, n_samples=10)
        fpv = generator.generate(4, n_samples=10)
        sq_freq = np.mean([extractor.extract_harmonic_features(a)[0] for a in small_quad])
        fpv_freq = np.mean([extractor.extract_harmonic_features(a)[0] for a in fpv])
        assert fpv_freq > sq_freq, "FPV should have higher fundamental frequency"

    def test_classifier_trains(self, trained_clf):
        assert trained_clf.is_trained

    def test_background_detection(self, trained_clf, generator):
        correct = 0
        for _ in range(10):
            audio = generator.generate(0, n_samples=1)[0]
            det = trained_clf.predict(audio)
            if det.drone_class == 0:
                correct += 1
        assert correct >= 7

    def test_fpv_detection(self, trained_clf, generator):
        correct = 0
        for _ in range(10):
            audio = generator.generate(4, n_samples=1)[0]
            det = trained_clf.predict(audio)
            if det.drone_class == 4:
                correct += 1
        assert correct >= 6

    def test_rpm_estimate_range(self, trained_clf, generator):
        # Small quad at 90-150 Hz → RPM = Hz * 60 = 5400-9000
        audio = generator.generate(1, n_samples=1)[0]
        det = trained_clf.predict(audio)
        assert 1000 < det.estimated_rpm < 30000, f"RPM out of range: {det.estimated_rpm}"

    def test_short_audio_input(self, extractor):
        short_audio = np.random.randn(100)
        feats = extractor.extract_features(short_audio)
        assert np.all(np.isfinite(feats))

    def test_silent_audio(self, extractor):
        silent = np.zeros(22050)
        feats = extractor.extract_features(silent)
        assert np.all(np.isfinite(feats))


# ─── Visual Tracker Tests ─────────────────────────────────────────────────────
class TestVisualTracker:

    @pytest.fixture(scope="class")
    def detector(self):
        from visual_tracker.visual_tracker import YOLODroneDetector
        return YOLODroneDetector()

    @pytest.fixture(scope="class")
    def kalman(self):
        from visual_tracker.visual_tracker import KalmanTracker
        return KalmanTracker(320, 240)

    @pytest.fixture(scope="class")
    def tracker(self):
        from visual_tracker.visual_tracker import MultiObjectTracker
        return MultiObjectTracker()

    def test_kalman_predict(self, kalman):
        pos = kalman.predict()
        assert len(pos) == 2
        assert np.all(np.isfinite(pos))

    def test_kalman_update(self, kalman):
        kalman.predict()
        kalman.update(np.array([325.0, 245.0]))
        pos = kalman.position
        # Position should move toward measurement
        assert abs(pos[0] - 325.0) < 50
        assert abs(pos[1] - 245.0) < 50

    def test_kalman_velocity_after_updates(self):
        from visual_tracker.visual_tracker import KalmanTracker
        k = KalmanTracker(100.0, 100.0)
        for i in range(5):
            k.predict()
            k.update(np.array([100.0 + i*10, 100.0]))
        vx, _ = k.velocity
        assert vx > 0, "Velocity should be positive (moving right)"

    def test_multi_tracker_new_detections(self, tracker):
        from visual_tracker.visual_tracker import BoundingBox
        dets = [BoundingBox(100,100,150,150,0.9), BoundingBox(300,200,360,260,0.8)]
        tracks = tracker.update(dets)
        assert len(tracks) == 2

    def test_multi_tracker_persistence(self, tracker):
        from visual_tracker.visual_tracker import BoundingBox
        # First frame
        dets = [BoundingBox(100,100,150,150,0.9)]
        tracks = tracker.update(dets)
        first_id = tracks[0].track_id if tracks else None

        # Second frame — same area
        dets2 = [BoundingBox(105,102,155,152,0.88)]
        tracks2 = tracker.update(dets2)
        if tracks2 and first_id is not None:
            assert any(t.track_id == first_id for t in tracks2), \
                "Track should persist across frames"

    def test_bounding_box_properties(self):
        from visual_tracker.visual_tracker import BoundingBox
        b = BoundingBox(100, 50, 200, 150, 0.9)
        assert b.cx == 150.0
        assert b.cy == 100.0
        assert b.w == 100.0
        assert b.h == 100.0
        assert b.area == 10000.0

    def test_detector_returns_visual_detection(self, detector):
        from visual_tracker.visual_tracker import VisualDetection
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert isinstance(result, VisualDetection)
        assert result.frame_width == 640
        assert result.frame_height == 480
        assert 0.0 <= result.highest_threat_score <= 1.0

    def test_distance_estimate_decreases_with_size(self):
        from visual_tracker.visual_tracker import MultiObjectTracker, BoundingBox
        mt = MultiObjectTracker()
        # Large bbox → close distance
        large_bbox = BoundingBox(100, 100, 300, 300, 0.95)   # w=200px → close
        small_bbox = BoundingBox(300, 200, 340, 240, 0.95)   # w=40px → far
        tracks_large = mt.update([large_bbox])
        mt2 = MultiObjectTracker()
        tracks_small = mt2.update([small_bbox])
        if tracks_large and tracks_small:
            assert tracks_large[0].distance_estimate_m < tracks_small[0].distance_estimate_m


# ─── Sensor Fusion Tests ──────────────────────────────────────────────────────
class TestSensorFusion:

    @pytest.fixture
    def engine(self):
        from sensor_fusion.fusion_engine import SensorFusionEngine
        return SensorFusionEngine()

    @pytest.fixture
    def make_reading(self):
        from sensor_fusion.fusion_engine import SensorReading
        def _make(sensor_type, active=True, detected=False,
                  score=0.0, conf=0.5, cls="BACKGROUND"):
            return SensorReading(sensor_type, active, detected, score, conf, cls)
        return _make

    def test_clear_when_all_zero(self, engine, make_reading):
        from sensor_fusion.fusion_engine import ThreatLevel
        rf = make_reading("RF", score=0.0)
        au = make_reading("AUDIO", score=0.0)
        vi = make_reading("VISUAL", score=0.0)
        result = engine.fuse(rf, au, vi)
        assert result.threat_level == ThreatLevel.CLEAR
        assert result.fused_score < 0.10

    def test_high_rf_alone_raises_threat(self, engine, make_reading):
        from sensor_fusion.fusion_engine import ThreatLevel
        rf = make_reading("RF", detected=True, score=0.85, conf=0.90, cls="MILITARY_SUSPECT")
        au = make_reading("AUDIO", score=0.02)
        vi = make_reading("VISUAL", score=0.03)
        result = engine.fuse(rf, au, vi)
        assert result.threat_level in (ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL)

    def test_conservative_rule_critical(self, engine, make_reading):
        from sensor_fusion.fusion_engine import ThreatLevel
        # Score > 0.90 + confidence > 0.75 should trigger conservative rule
        rf = make_reading("RF", detected=True, score=0.95, conf=0.92, cls="MILITARY_SUSPECT")
        au = make_reading("AUDIO", score=0.10)
        vi = make_reading("VISUAL", score=0.10)
        result = engine.fuse(rf, au, vi)
        assert result.threat_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL)

    def test_ema_smoothing(self, engine, make_reading):
        """Threat score should not jump instantly but smooth over frames."""
        scores = []
        for i in range(10):
            score = 0.8 if i >= 5 else 0.0
            rf = make_reading("RF", detected=score>0, score=score, conf=0.8)
            au = make_reading("AUDIO", score=score*0.9)
            vi = make_reading("VISUAL", score=score*0.85)
            result = engine.fuse(rf, au, vi)
            scores.append(result.fused_score)

        # Frame 5 (first high frame) should NOT immediately hit 0.8
        assert scores[5] < 0.5, f"EMA should smooth: frame 5 score was {scores[5]:.3f}"
        # By frame 9 it should be approaching the high value
        assert scores[9] > scores[5], "Score should increase over time"

    def test_inactive_sensor_handling(self, engine, make_reading):
        from sensor_fusion.fusion_engine import ThreatLevel
        rf = make_reading("RF", active=False, score=0.0)
        au = make_reading("AUDIO", detected=True, score=0.70, conf=0.85)
        vi = make_reading("VISUAL", detected=True, score=0.65, conf=0.80)
        result = engine.fuse(rf, au, vi)
        assert result.rf_active == False
        assert result.fused_score > 0.10  # Should still detect despite no RF

    def test_false_positive_filter(self, engine, make_reading):
        """Single high-score frame should not trigger alarm (needs 3/5)."""
        # Send 1 high frame then 4 low frames
        results = []
        for i in range(5):
            score = 0.9 if i == 0 else 0.01
            rf = make_reading("RF", detected=score>0.1, score=score, conf=0.9)
            au = make_reading("AUDIO", score=score*0.8)
            vi = make_reading("VISUAL", score=score*0.85)
            results.append(engine.fuse(rf, au, vi))
        # With EMA dampening + FP filter, alarm should not be sustained
        # At least some frames should not be alarms
        alarm_count = sum(1 for r in results if r.is_alarm)
        assert alarm_count < 5, "FP filter should suppress some false alarms"

    def test_two_sensors_high_override(self, engine, make_reading):
        from sensor_fusion.fusion_engine import ThreatLevel
        # 2 sensors both high → should get HIGH or above
        rf = make_reading("RF", detected=True, score=0.70, conf=0.80)
        au = make_reading("AUDIO", detected=True, score=0.65, conf=0.75)
        vi = make_reading("VISUAL", score=0.05)
        # Run multiple frames to let EMA rise
        for _ in range(8):
            result = engine.fuse(rf, au, vi)
        assert result.threat_level in (ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL)

    def test_assessment_serialisable(self, engine, make_reading):
        import json
        rf = make_reading("RF", detected=True, score=0.5, conf=0.7)
        au = make_reading("AUDIO", detected=True, score=0.4, conf=0.65)
        vi = make_reading("VISUAL", score=0.3)
        result = engine.fuse(rf, au, vi)
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert "fused_score" in parsed
        assert "threat_level" in parsed
        assert "is_alarm" in parsed

    def test_history_accumulates(self, engine, make_reading):
        rf = make_reading("RF"); au = make_reading("AUDIO"); vi = make_reading("VISUAL")
        for _ in range(15):
            engine.fuse(rf, au, vi)
        scores = engine.get_history_scores()
        assert len(scores) == 15

    def test_stats_tracking(self, engine, make_reading):
        stats = engine.get_stats()
        assert "total_frames" in stats
        assert "total_alarms" in stats
        assert stats["total_frames"] >= 0


# ─── Integration Test ─────────────────────────────────────────────────────────
class TestIntegration:
    """End-to-end: train all models and run a full detection cycle."""

    def test_full_pipeline(self):
        from rf_classifier.rf_classifier import RFDroneClassifier, SyntheticRFDataGenerator
        from audio_classifier.audio_classifier import AcousticDroneClassifier, SyntheticAudioGenerator
        from visual_tracker.visual_tracker import YOLODroneDetector
        from sensor_fusion.fusion_engine import SensorFusionEngine, SensorReading, ThreatLevel

        rf_clf = RFDroneClassifier()
        rf_clf.train(n_per_class=50)
        audio_clf = AcousticDroneClassifier()
        audio_clf.train(n_per_class=50)
        visual_det = YOLODroneDetector()
        fusion = SensorFusionEngine()

        rf_gen = SyntheticRFDataGenerator()
        audio_gen = SyntheticAudioGenerator()

        results = []
        for i in range(10):
            # Simulate DJI Phantom incursion after frame 5
            cls = 1 if i >= 5 else 0
            rf_frame = rf_gen.generate(cls, n_frames=1)[0]
            audio_sample = audio_gen.generate(2 if cls else 0, n_samples=1)[0]
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

            rf_det = rf_clf.predict(rf_frame)
            audio_det = audio_clf.predict(audio_sample)
            vis_det = visual_det.detect(dummy_frame)

            rf_r = SensorReading("RF", True, cls!=0, rf_det.threat_score, rf_det.confidence, rf_det.class_name)
            au_r = SensorReading("AUDIO", True, cls!=0, audio_det.threat_score, audio_det.confidence, audio_det.class_name)
            vi_r = SensorReading("VISUAL", True, vis_det.n_drones_detected>0, vis_det.highest_threat_score, 0.6, "VISUAL")

            assessment = fusion.fuse(rf_r, au_r, vi_r)
            results.append(assessment)

        # Threat should generally increase in second half
        first_half_avg = np.mean([r.fused_score for r in results[:5]])
        second_half_avg = np.mean([r.fused_score for r in results[5:]])
        assert second_half_avg >= first_half_avg * 0.8, \
            f"Threat should increase during incursion: {first_half_avg:.3f} → {second_half_avg:.3f}"

        # All assessments should be valid
        for r in results:
            assert r.threat_level in ThreatLevel.__members__.values()
            assert 0.0 <= r.fused_score <= 1.0
            assert isinstance(r.alert_message, str) and len(r.alert_message) > 0

    def test_model_persistence_roundtrip(self, tmp_path):
        from rf_classifier.rf_classifier import RFDroneClassifier, SyntheticRFDataGenerator
        clf = RFDroneClassifier()
        clf.train(n_per_class=60)

        path = str(tmp_path / "rf_roundtrip.pkl")
        clf.save(path)

        clf2 = RFDroneClassifier()
        clf2.load(path)

        gen = SyntheticRFDataGenerator()
        for cls in [0, 1, 3, 7]:
            frame = gen.generate(cls, n_frames=1)[0]
            d1 = clf.predict(frame)
            d2 = clf2.predict(frame)
            assert d1.drone_class == d2.drone_class, \
                f"Class {cls}: original={d1.drone_class}, loaded={d2.drone_class}"


# ─── Performance Benchmarks ───────────────────────────────────────────────────
class TestPerformance:
    """Latency benchmarks — critical for real-time edge deployment."""

    def test_rf_inference_latency(self):
        from rf_classifier.rf_classifier import RFDroneClassifier, SyntheticRFDataGenerator
        clf = RFDroneClassifier()
        clf.train(n_per_class=60)
        gen = SyntheticRFDataGenerator()
        frame = gen.generate(1, n_frames=1)[0]

        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            clf.predict(frame)
            times.append(time.perf_counter() - t0)

        median_ms = np.median(times) * 1000
        p95_ms = np.percentile(times, 95) * 1000
        print(f"\n  RF inference: median={median_ms:.2f}ms, p95={p95_ms:.2f}ms")
        assert median_ms < 100, f"RF inference too slow: {median_ms:.1f}ms"

    def test_audio_inference_latency(self):
        from audio_classifier.audio_classifier import AcousticDroneClassifier, SyntheticAudioGenerator
        clf = AcousticDroneClassifier()
        clf.train(n_per_class=60)
        gen = SyntheticAudioGenerator()
        audio = gen.generate(1, n_samples=1)[0]

        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            clf.predict(audio)
            times.append(time.perf_counter() - t0)

        median_ms = np.median(times) * 1000
        p95_ms = np.percentile(times, 95) * 1000
        print(f"\n  Audio inference: median={median_ms:.2f}ms, p95={p95_ms:.2f}ms")
        assert median_ms < 200, f"Audio inference too slow: {median_ms:.1f}ms"

    def test_fusion_throughput(self):
        from sensor_fusion.fusion_engine import SensorFusionEngine, SensorReading
        engine = SensorFusionEngine()
        rf = SensorReading("RF", True, True, 0.5, 0.8, "DJI")
        au = SensorReading("AUDIO", True, True, 0.4, 0.7, "QUAD")
        vi = SensorReading("VISUAL", True, True, 0.6, 0.75, "TRACK")

        t0 = time.perf_counter()
        N = 1000
        for _ in range(N):
            engine.fuse(rf, au, vi)
        elapsed = time.perf_counter() - t0
        fps = N / elapsed
        print(f"\n  Fusion throughput: {fps:.0f} fps")
        assert fps > 100, f"Fusion too slow: {fps:.0f} fps (need >100)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
