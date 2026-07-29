"""API endpoint tests.

Endpoints that need no trained artifacts are tested unconditionally. Those that
do are marked with `requires_model` and skip cleanly on a fresh clone, so the
suite stays green before the notebook has been run.

The no-model path is tested deliberately rather than skipped: a cold container
is a real state in production, and the service must degrade rather than crash.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from api.main import app
from src.config import BACKBONE_PATH, CLASS_NAMES_PATH, REGISTRY_PATH

ARTIFACTS_PRESENT = (
    Path(BACKBONE_PATH).exists()
    and Path(CLASS_NAMES_PATH).exists()
    and Path(REGISTRY_PATH).exists()
)

requires_model = pytest.mark.skipif(
    not ARTIFACTS_PRESENT,
    reason="trained artifacts absent; run the notebook's export step first",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _image_bytes(seed: int = 0, size: int = 224, fmt: str = "PNG") -> bytes:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=fmt)
    return buffer.getvalue()


# ==========================================================================
# Monitoring -- must work with or without a model
# ==========================================================================

def test_health_always_responds(client):
    """Health must answer even with no model, so orchestrators can tell
    'starting up' from 'dead'."""
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] in {"healthy", "degraded"}
    assert isinstance(body["model_ready"], bool)
    assert body["uptime_seconds"] >= 0
    assert body["hostname"]


def test_health_reports_uptime_increasing(client):
    first = client.get("/health").json()["uptime_seconds"]
    second = client.get("/health").json()["uptime_seconds"]
    assert second >= first


def test_metrics_shape(client):
    response = client.get("/metrics")
    assert response.status_code == 200

    body = response.json()
    for key in ("uptime_seconds", "total_predictions", "latency_ms",
                "confidence", "requests_per_second_1m", "hostname"):
        assert key in body

    for percentile in ("p50", "p95", "p99", "mean", "max"):
        assert percentile in body["latency_ms"]


def test_root(client):
    assert client.get("/").json()["service"] == "crop-disease-classification"


def test_hostname_identifies_replica(client):
    """Load testing across replicas depends on this field being present."""
    assert client.get("/health").json()["hostname"] == client.get("/metrics").json()["hostname"]


# ==========================================================================
# Input validation -- independent of the model
# ==========================================================================

def test_predict_rejects_unsupported_type(client):
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")})
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()


def test_predict_rejects_empty_file(client):
    response = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code == 400


def test_predict_batch_rejects_oversized_batch(client):
    files = [("files", (f"{i}.png", _image_bytes(i, 32), "image/png"))
             for i in range(40)]
    assert client.post("/predict/batch", files=files).status_code == 413


def test_upload_rejects_unknown_class(client):
    """Guessed labels would poison the replay buffer, so they are refused."""
    response = client.post(
        "/upload",
        files={"files": ("leaf.png", _image_bytes(), "image/png")},
        data={"class_name": "Definitely___NotAClass"})
    assert response.status_code in {400, 503}


def test_retrain_status_unknown_job(client):
    assert client.get("/retrain/status/deadbeef").status_code == 404


def test_activate_unknown_version(client):
    assert client.post("/models/activate/v999").status_code == 404


def test_retrain_trigger_endpoint(client):
    body = client.get("/retrain/trigger").json()
    assert "should_retrain" in body
    assert "triggers" in body
    assert {"volume", "drift"} <= set(body["triggers"])


def test_uploads_summary(client):
    body = client.get("/uploads/summary").json()
    assert body["total_images"] >= 0
    assert "retrain_trigger" in body


def test_models_listing(client):
    body = client.get("/models").json()
    assert "versions" in body and isinstance(body["versions"], list)


# ==========================================================================
# Cold-start behaviour
# ==========================================================================

@pytest.mark.skipif(ARTIFACTS_PRESENT, reason="artifacts are present")
def test_predict_returns_503_without_model(client):
    """A model-less container must return 503 (retry me), never 500."""
    response = client.post(
        "/predict", files={"file": ("leaf.png", _image_bytes(), "image/png")})
    assert response.status_code == 503


@pytest.mark.skipif(ARTIFACTS_PRESENT, reason="artifacts are present")
def test_health_degraded_without_model(client):
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["model_ready"] is False


# ==========================================================================
# Full path -- requires trained artifacts
# ==========================================================================

@requires_model
def test_classes_endpoint(client):
    body = client.get("/classes").json()
    assert body["n_classes"] == 38

    entry = body["classes"][0]
    assert {"index", "class_name", "crop", "condition", "healthy"} <= set(entry)


@requires_model
def test_predict_returns_ranked_probabilities(client):
    response = client.post(
        "/predict", files={"file": ("leaf.png", _image_bytes(), "image/png")})
    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["latency_ms"] > 0
    assert body["model_version"]
    assert len(body["top_k"]) == 5

    probabilities = [p["probability"] for p in body["top_k"]]
    assert probabilities == sorted(probabilities, reverse=True)
    assert body["prediction"] == body["top_k"][0]["class_name"]


@requires_model
def test_predict_top_k_is_configurable(client):
    response = client.post(
        "/predict?top_k=3",
        files={"file": ("leaf.png", _image_bytes(), "image/png")})
    assert len(response.json()["top_k"]) == 3


@requires_model
def test_predict_accepts_jpeg_and_odd_sizes(client):
    """Phone photos are JPEG and arbitrarily sized."""
    response = client.post(
        "/predict",
        files={"file": ("photo.jpg", _image_bytes(1, size=640, fmt="JPEG"), "image/jpeg")})
    assert response.status_code == 200


@requires_model
def test_batch_matches_single_prediction(client):
    """Batching must not change the answer -- only the cost per image."""
    payload = _image_bytes(7)

    single = client.post("/predict", files={"file": ("a.png", payload, "image/png")}).json()
    batch = client.post(
        "/predict/batch",
        files=[("files", ("a.png", payload, "image/png"))]).json()

    assert batch["count"] == 1
    assert batch["results"][0]["prediction"] == single["prediction"]
    assert batch["results"][0]["confidence"] == pytest.approx(single["confidence"], abs=1e-5)


@requires_model
def test_metrics_count_predictions(client):
    before = client.get("/metrics").json()["total_predictions"]
    client.post("/predict", files={"file": ("a.png", _image_bytes(3), "image/png")})
    after = client.get("/metrics").json()

    assert after["total_predictions"] == before + 1
    assert after["latency_ms"]["p50"] > 0


@requires_model
def test_upload_accepts_known_class(client, tmp_path):
    import json

    known = json.loads(Path(CLASS_NAMES_PATH).read_text())[0]
    response = client.post(
        "/upload",
        files=[("files", (f"leaf{i}.png", _image_bytes(i), "image/png")) for i in range(3)],
        data={"class_name": known})

    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] == 3
    assert body["staged_total"] >= 3
    assert "retrain_trigger" in body


@requires_model
def test_retrain_refuses_when_conditions_unmet(client):
    """The endpoint enforces the same policy the automatic trigger applies."""
    trigger = client.get("/retrain/trigger").json()
    if not trigger["should_retrain"]:
        assert client.post("/retrain").status_code == 409
