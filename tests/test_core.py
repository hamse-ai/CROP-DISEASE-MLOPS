"""Unit tests for the pieces that do not need trained artifacts.

These cover the logic most likely to fail silently in production: label
mapping, the promotion gate, the drift trigger and the NumPy reimplementation
of MobileNetV2's preprocessing. All run on synthetic data, so they work on a
fresh clone with no dataset downloaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import build_replay_buffer, evaluate_predictions, train_head
from src.monitoring import ServiceMonitor
from src.preprocessing import (
    background_uniformity,
    colour_statistics,
    is_healthy,
    load_image,
    preprocess_for_backbone,
    split_class_name,
    texture_features,
)
from src.registry import ModelRegistry


# ==========================================================================
# Preprocessing
# ==========================================================================

def _synthetic_image(seed: int = 0, size: int = 224) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)


def test_preprocess_scales_to_minus_one_to_one():
    """The NumPy path must match keras.applications.mobilenet_v2 exactly."""
    image = _synthetic_image()
    out = preprocess_for_backbone(image)

    assert out.shape == (1, 224, 224, 3)
    assert out.dtype == np.float32
    assert -1.0 <= out.min() and out.max() <= 1.0

    # Endpoints are what a rescaling bug would break first.
    assert preprocess_for_backbone(np.zeros((2, 2, 3), np.uint8)).min() == pytest.approx(-1.0)
    assert preprocess_for_backbone(np.full((2, 2, 3), 255, np.uint8)).max() == pytest.approx(1.0)


def test_preprocess_accepts_batch_and_single():
    single = preprocess_for_backbone(_synthetic_image())
    batch = preprocess_for_backbone(np.stack([_synthetic_image(i) for i in range(4)]))
    assert single.shape[0] == 1
    assert batch.shape == (4, 224, 224, 3)


def test_load_image_normalises_mode_and_size(tmp_path):
    """Uploads arrive as RGBA, greyscale and odd sizes; all must normalise."""
    from PIL import Image

    for mode, name in [("RGBA", "a.png"), ("L", "b.png"), ("P", "c.png")]:
        path = tmp_path / name
        Image.new(mode, (97, 143)).save(path)
        out = load_image(path)
        assert out.shape == (224, 224, 3), f"{mode} did not normalise"
        assert out.dtype == np.uint8


def test_load_image_accepts_bytes(tmp_path):
    """The API feeds uploaded bytes straight in without touching disk."""
    from PIL import Image

    path = tmp_path / "leaf.png"
    Image.fromarray(_synthetic_image()).save(path)
    assert load_image(path.read_bytes()).shape == (224, 224, 3)


@pytest.mark.parametrize("raw, crop, condition", [
    ("Tomato___Late_blight", "Tomato", "Late blight"),
    ("Corn_(maize)___Common_rust_", "Corn (maize)", "Common rust"),
    ("Pepper,_bell___healthy", "Pepper bell", "healthy"),
    ("Blueberry___healthy", "Blueberry", "healthy"),
])
def test_split_class_name(raw, crop, condition):
    assert split_class_name(raw) == (crop, condition)


def test_split_class_name_handles_missing_separator():
    assert split_class_name("Unlabelled") == ("Unlabelled", "unknown")


def test_is_healthy():
    assert is_healthy("Tomato___healthy")
    assert not is_healthy("Tomato___Late_blight")


def test_colour_statistics_separate_green_from_brown():
    """The vegetation index must actually order healthy above diseased.

    This is the quantitative claim the EDA section makes, so it is asserted
    rather than assumed.
    """
    green = np.zeros((64, 64, 3), np.uint8)
    green[..., 1] = 200
    brown = np.zeros((64, 64, 3), np.uint8)
    brown[..., 0], brown[..., 1] = 150, 90

    assert colour_statistics(green)["excess_green"] > colour_statistics(brown)["excess_green"]
    assert colour_statistics(green)["redness_index"] < colour_statistics(brown)["redness_index"]


def test_texture_features_detect_lesion_like_structure():
    """A spotted leaf must score higher on edge density than a flat one."""
    flat = np.full((128, 128, 3), 120, np.uint8)
    spotted = flat.copy()
    for y in range(10, 120, 20):
        for x in range(10, 120, 20):
            spotted[y:y + 6, x:x + 6] = 30

    assert texture_features(spotted)["edge_density"] > texture_features(flat)["edge_density"]
    assert texture_features(spotted)["laplacian_var"] > texture_features(flat)["laplacian_var"]


def test_background_uniformity_flags_uniform_backdrop():
    """The lab-background finding underpins the drift trigger, so pin it down."""
    uniform = np.full((128, 128, 3), 200, np.uint8)
    uniform[40:90, 40:90] = [30, 140, 40]           # a leaf on a plain backdrop

    rng = np.random.default_rng(0)
    cluttered = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)

    assert background_uniformity(uniform)["corner_std"] < \
           background_uniformity(cluttered)["corner_std"]
    assert background_uniformity(uniform)["background_fraction"] > 0.5


# ==========================================================================
# Head training and metrics
# ==========================================================================

@pytest.fixture
def separable_embeddings():
    """Linearly separable 5-class embeddings; any sane head should nail these."""
    rng = np.random.default_rng(42)
    n_classes, per_class, dim = 5, 40, 64
    centres = rng.normal(0, 4, (n_classes, dim))
    X = np.concatenate([
        centres[c] + rng.normal(0, 0.5, (per_class, dim)) for c in range(n_classes)
    ]).astype(np.float32)
    y = np.repeat(np.arange(n_classes), per_class)
    return X, y


def test_train_head_learns(separable_embeddings):
    X, y = separable_embeddings
    head = train_head(X, y, kind="logreg")
    assert (head.predict(X) == y).mean() > 0.95


def test_evaluate_predictions_reports_full_metric_suite(separable_embeddings):
    X, y = separable_embeddings
    proba = train_head(X, y, kind="logreg").predict_proba(X)
    metrics = evaluate_predictions(y, proba, [f"class_{i}" for i in range(5)])

    for key in ("accuracy", "f1_macro", "f1_weighted", "precision_macro",
                "recall_macro", "roc_auc_macro", "pr_auc_macro",
                "mean_confidence", "per_class", "confusion_matrix"):
        assert key in metrics, f"missing metric: {key}"

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert np.array(metrics["confusion_matrix"]).shape == (5, 5)


def test_macro_f1_punishes_ignoring_a_rare_class():
    """Why macro-F1 is the headline metric rather than accuracy.

    A model that ignores a rare class entirely still scores high accuracy but
    should be visibly penalised on macro-F1 -- this is the imbalance argument
    the notebook makes, asserted rather than claimed.
    """
    y_true = np.array([0] * 95 + [1] * 5)
    proba = np.zeros((100, 2))
    proba[:, 0] = 1.0                     # always predicts the majority class

    metrics = evaluate_predictions(y_true, proba)
    assert metrics["accuracy"] == pytest.approx(0.95)
    assert metrics["f1_macro"] < 0.5


def test_build_replay_buffer_is_stratified_and_bounded():
    """The buffer must cap per class and keep every class represented."""
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(500), np.ones(50), np.full(300, 2)]).astype(np.int64)
    X = rng.normal(size=(len(y), 32)).astype(np.float32)

    Xb, yb = build_replay_buffer(X, y, per_class=100)

    counts = np.bincount(yb)
    assert counts[0] == 100          # capped
    assert counts[1] == 50           # smaller than the cap, kept whole
    assert counts[2] == 100
    assert Xb.dtype == np.float16    # halves the on-disk buffer
    assert len(Xb) == len(yb)


# ==========================================================================
# Registry
# ==========================================================================

@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(path=tmp_path / "registry.json", model_dir=tmp_path)


def test_registry_versions_increment(registry):
    assert registry.next_version() == "v1"
    registry.register("v1", "head_v1.pkl", {"f1_macro": 0.9})
    assert registry.next_version() == "v2"
    assert registry.active == "v1"


def test_registry_round_trips(registry, tmp_path):
    registry.register("v1", "head_v1.pkl", {"f1_macro": 0.9, "accuracy": 0.95})
    reloaded = ModelRegistry(path=tmp_path / "registry.json", model_dir=tmp_path)

    assert reloaded.active == "v1"
    assert reloaded.versions["v1"].metrics["f1_macro"] == 0.9


def test_promotion_gate_blocks_regression(registry):
    """The check that stops a bad retrain reaching users."""
    registry.register("v1", "head_v1.pkl", {"f1_macro": 0.90})

    ok, reason = registry.should_promote({"f1_macro": 0.85}, max_regression=0.02)
    assert not ok and "regressed" in reason

    ok, _ = registry.should_promote({"f1_macro": 0.895}, max_regression=0.02)
    assert ok, "a drop inside tolerance should still promote"

    ok, _ = registry.should_promote({"f1_macro": 0.95}, max_regression=0.02)
    assert ok


def test_promotion_gate_allows_first_version(registry):
    ok, _ = registry.should_promote({"f1_macro": 0.5}, max_regression=0.02)
    assert ok


def test_activate_rejects_missing_artifact(registry):
    """Rolling back to a version whose file is gone must fail loudly."""
    registry.register("v1", "head_v1.pkl", {})
    with pytest.raises(FileNotFoundError):
        registry.activate("v1")

    with pytest.raises(KeyError):
        registry.activate("v99")


def test_history_marks_active(registry, tmp_path):
    (tmp_path / "head_v1.pkl").write_bytes(b"x")
    (tmp_path / "head_v2.pkl").write_bytes(b"x")
    registry.register("v1", "head_v1.pkl", {})
    registry.register("v2", "head_v2.pkl", {})

    history = registry.history()
    assert sum(h["active"] for h in history) == 1
    assert next(h for h in history if h["active"])["version"] == "v2"


# ==========================================================================
# Monitoring and the drift trigger
# ==========================================================================

def test_monitor_tracks_latency_percentiles(tmp_path):
    monitor = ServiceMonitor(state_path=tmp_path / "state.json")
    for i in range(100):
        monitor.record_prediction(confidence=0.9, latency_ms=float(i))

    stats = monitor.stats()
    assert stats["total_predictions"] == 100
    assert stats["latency_ms"]["p50"] == pytest.approx(50, abs=2)
    assert stats["latency_ms"]["p95"] == pytest.approx(95, abs=2)
    assert stats["latency_ms"]["p99"] <= stats["latency_ms"]["max"]


def test_drift_trigger_needs_a_full_window(tmp_path):
    """A handful of low-confidence predictions must not trigger a retrain.

    Otherwise the service would retrain immediately after every restart, on
    noise.
    """
    monitor = ServiceMonitor(window_size=50, confidence_threshold=0.65,
                             state_path=tmp_path / "state.json")

    for _ in range(10):
        monitor.record_prediction(confidence=0.10, latency_ms=5.0)
    trigger = monitor.check_retrain_trigger(upload_dir=tmp_path / "nonexistent")
    assert not trigger["triggers"]["drift"]["triggered"]
    assert not trigger["triggers"]["drift"]["window_filled"]

    for _ in range(50):
        monitor.record_prediction(confidence=0.10, latency_ms=5.0)
    trigger = monitor.check_retrain_trigger(upload_dir=tmp_path / "nonexistent")
    assert trigger["triggers"]["drift"]["triggered"]
    assert trigger["should_retrain"]
    assert trigger["reasons"]


def test_confident_model_does_not_trigger_drift(tmp_path):
    monitor = ServiceMonitor(window_size=20, confidence_threshold=0.65,
                             state_path=tmp_path / "state.json")
    for _ in range(30):
        monitor.record_prediction(confidence=0.97, latency_ms=5.0)

    trigger = monitor.check_retrain_trigger(upload_dir=tmp_path / "nonexistent")
    assert not trigger["should_retrain"]


def test_volume_trigger_counts_staged_uploads(tmp_path):
    from PIL import Image

    upload_dir = tmp_path / "uploads" / "Tomato___Late_blight"
    upload_dir.mkdir(parents=True)
    monitor = ServiceMonitor(upload_threshold=5, state_path=tmp_path / "state.json")

    for i in range(4):
        Image.fromarray(_synthetic_image(i, 32)).save(upload_dir / f"{i}.png")
    assert not monitor.check_retrain_trigger(tmp_path / "uploads")["should_retrain"]

    for i in range(4, 8):
        Image.fromarray(_synthetic_image(i, 32)).save(upload_dir / f"{i}.png")
    trigger = monitor.check_retrain_trigger(tmp_path / "uploads")
    assert trigger["should_retrain"]
    assert trigger["triggers"]["volume"]["staged_images"] == 8


def test_retrain_clears_stale_drift_window(tmp_path):
    """After retraining, the old model's confidences must not linger.

    Carrying them over would let a stale signal immediately re-trigger.
    """
    monitor = ServiceMonitor(window_size=10, confidence_threshold=0.65,
                             state_path=tmp_path / "state.json")
    for _ in range(20):
        monitor.record_prediction(confidence=0.1, latency_ms=1.0)
    assert monitor.check_retrain_trigger(tmp_path)["triggers"]["drift"]["triggered"]

    monitor.record_retrain()
    trigger = monitor.check_retrain_trigger(tmp_path)
    assert not trigger["triggers"]["drift"]["triggered"]
    assert monitor.retrain_count == 1


def test_monitor_state_survives_restart(tmp_path):
    state = tmp_path / "state.json"
    monitor = ServiceMonitor(state_path=state)
    for _ in range(7):
        monitor.record_prediction(confidence=0.8, latency_ms=10.0, class_name="Tomato___healthy")
    monitor.persist()

    restored = ServiceMonitor(state_path=state)
    restored.restore()
    assert restored.total_predictions == 7
    assert restored.class_counts["Tomato___healthy"] == 7
