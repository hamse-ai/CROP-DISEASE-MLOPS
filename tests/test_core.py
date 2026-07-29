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


def test_error_rate_uses_attempted_requests(tmp_path):
    """Errors without any successful prediction must not report 0%.

    Dividing by successes alone hides the worst case: a service where every
    request fails would report a 0% error rate.
    """
    monitor = ServiceMonitor(state_path=tmp_path / "state.json")

    for _ in range(3):
        monitor.record_error()
    assert monitor.stats()["error_rate"] == 1.0, "all-failures must read as 100%"

    for _ in range(97):
        monitor.record_prediction(confidence=0.9, latency_ms=5.0)
    assert monitor.stats()["error_rate"] == pytest.approx(0.03)


# ==========================================================================
# Retraining safeguards
# ==========================================================================

def test_retrain_inherits_active_head_kind(tmp_path, monkeypatch):
    """A retrain must not silently swap the deployed head architecture.

    run_retraining used to default to "logreg" regardless of what was
    deployed, so retraining an mlp model quietly downgraded it -- costing
    accuracy that the promotion gate then had to absorb as if it were
    forgetting.
    """
    import src.retrain as retrain_module

    registry = ModelRegistry(path=tmp_path / "registry.json", model_dir=tmp_path)
    (tmp_path / "head_v1.pkl").write_bytes(b"x")
    registry.register("v1", "head_v1.pkl", {"f1_macro": 0.96}, head_kind="mlp")

    monkeypatch.setattr(retrain_module, "ModelRegistry",
                        lambda *a, **k: registry)

    captured = {}

    def fake_collect(*_a, **_kw):
        captured["reached_collect"] = True
        raise retrain_module.RetrainError("stop here -- head_kind already resolved")

    monkeypatch.setattr(retrain_module, "collect_uploads", fake_collect)

    # head_kind=None must resolve from the registry before anything else runs.
    with pytest.raises(retrain_module.RetrainError):
        retrain_module.run_retraining(head_kind=None, upload_dir=tmp_path)
    assert captured.get("reached_collect")

    active = registry.active_version()
    assert active.head_kind == "mlp", "registry should still report the mlp head"


def test_build_training_matrix_combines_without_duplicating(tmp_path):
    """The combined matrix must be one allocation, and hold the right rows.

    The naive concatenate held the buffer and the result at once -- roughly
    200 MB for this buffer size, which does not fit beside a loaded model in a
    512 MB container.
    """
    from src.retrain import build_training_matrix

    rng = np.random.default_rng(0)
    X_buffer = rng.normal(size=(500, 64)).astype(np.float16)
    y_buffer = rng.integers(0, 38, 500)
    path = tmp_path / "buffer.npz"
    np.savez_compressed(path, X=X_buffer, y=y_buffer)

    X_new = rng.normal(size=(20, 64)).astype(np.float32)
    y_new = rng.integers(0, 38, 20)

    X, y, n_buffer = build_training_matrix(X_new, y_new, path=path)

    assert n_buffer == 500
    assert X.shape == (520, 64)
    assert X.dtype == np.float32
    assert len(y) == 520
    # New rows must land after the buffer, unmodified.
    np.testing.assert_allclose(X[500:], X_new, rtol=1e-6)
    np.testing.assert_array_equal(y[500:], y_new)


def test_build_training_matrix_requires_a_buffer(tmp_path):
    """Retraining without the replay buffer must refuse, not silently proceed."""
    from src.retrain import RetrainError, build_training_matrix

    with pytest.raises(RetrainError, match="replay buffer missing"):
        build_training_matrix(np.zeros((4, 64), np.float32), np.zeros(4, np.int64),
                              path=tmp_path / "absent.npz")


def _write_job(state, job_id, status, updated_at, **extra):
    import json as _json
    payload = {"jobs": [{"job_id": job_id, "status": status, "progress": 0.55,
                         "message": "loading replay buffer", "submitted_at": "now",
                         "updated_at": updated_at, "result": None, "error": None,
                         **extra}]}
    state.write_text(_json.dumps(payload))


def test_stalled_job_is_reported_as_failed(tmp_path):
    """A job whose worker died must report failure, not vanish.

    Observed for real: the container hit its 512 MB limit during a retrain,
    the kernel killed the uvicorn worker, uvicorn respawned it, /health stayed
    green -- and the job returned "unknown job" with no error recorded. Silent
    failure is worse than a loud one.
    """
    from datetime import datetime, timedelta, timezone

    from src.retrain import RetrainJobManager

    state = tmp_path / "jobs.json"
    stale = (datetime.now(timezone.utc) - timedelta(seconds=600)).isoformat(timespec="seconds")
    _write_job(state, "abc123", "running", stale)

    revived = RetrainJobManager(state_path=state)
    job = revived.get("abc123")

    assert job is not None, "job must not vanish when its worker dies"
    assert job["status"] == "failed"
    assert "memory" in job["error"].lower()


def test_live_job_owned_by_another_worker_is_not_declared_dead(tmp_path):
    """The bug this heartbeat exists to prevent.

    With more than one worker (or replica), the process answering a status
    poll is usually NOT the one running the job. Declaring any "running" job
    dead on sight reported a completed, promoted retrain as "worker exited" --
    a healthy pipeline looking broken.
    """
    from datetime import datetime, timezone

    from src.retrain import RetrainJobManager

    state = tmp_path / "jobs.json"
    fresh = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_job(state, "live1", "running", fresh)

    observer = RetrainJobManager(state_path=state)   # a different worker
    job = observer.get("live1")

    assert job["status"] == "running", "a live job must not be reported failed"
    assert job["error"] is None


def test_completed_jobs_are_readable_by_another_worker(tmp_path):
    """With replicas, the worker polled is rarely the one that ran the job."""
    from src.retrain import RetrainJobManager

    state = tmp_path / "jobs.json"
    owner = RetrainJobManager(state_path=state)
    with owner._lock:
        owner._jobs["done1"] = {
            "job_id": "done1", "status": "completed", "progress": 1.0,
            "message": "promoted", "submitted_at": "now",
            "result": {"version": "v2", "promoted": True}, "error": None,
        }
        owner._order.append("done1")
        owner._persist_locked()

    other = RetrainJobManager(state_path=state)
    job = other.get("done1")
    assert job["status"] == "completed"
    assert job["result"]["version"] == "v2"


def test_update_replay_buffer_stays_float16_and_bounded(tmp_path):
    """The buffer refresh must not inflate to float32.

    This is what actually killed the container worker: the refresh loaded the
    buffer as float32, concatenated (a second copy) and re-subsampled (a third)
    while zlib compressed the result. The peak landed *after* the new model was
    registered, so the retrain looked successful and then the process vanished
    at 90%. The fit itself only adds ~18 MB.
    """
    from src.retrain import update_replay_buffer

    rng = np.random.default_rng(0)
    path = tmp_path / "buffer.npz"
    # 38 classes at the per-class cap, as it ships.
    y_old = np.repeat(np.arange(38), 300)
    X_old = rng.normal(size=(len(y_old), 64)).astype(np.float16)
    np.savez_compressed(path, X=X_old, y=y_old)

    X_new = rng.normal(size=(40, 64)).astype(np.float32)
    y_new = rng.integers(0, 38, 40)

    n, n_classes = update_replay_buffer(X_new, y_new, per_class=300, path=path)

    with np.load(path) as data:
        assert data["X"].dtype == np.float16, "buffer must stay float16 on disk"
        # Bounded: adding images must not grow it past the per-class cap.
        assert len(data["X"]) <= 38 * 300
        assert len(np.unique(data["y"])) == 38, "every class must survive"
    assert n == len(np.load(path)["X"]) and n_classes == 38
