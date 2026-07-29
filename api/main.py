"""FastAPI prediction and retraining service.

Endpoints group into four concerns:

* **Prediction**  ``/predict``, ``/predict/batch``
* **Monitoring**  ``/health``, ``/metrics`` -- what the UI's uptime tab reads
* **Data**        ``/upload``, ``/uploads/summary``
* **Model**       ``/retrain``, ``/retrain/status/{id}``, ``/models``,
                  ``/models/activate/{version}``

Every response carries the container hostname. With several replicas behind the
nginx balancer that is what proves load is actually being distributed during
the Locust runs, rather than one container serving everything.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    BatchPredictionResponse,
    ClassListResponse,
    HealthResponse,
    MetricsResponse,
    ModelListResponse,
    PredictionResponse,
    RetrainJobResponse,
    UploadResponse,
    UploadSummaryResponse,
)
from src.config import (
    CLASS_NAMES_PATH,
    MAX_UPLOAD_MB,
    UPLOAD_DIR,
    ensure_dirs,
)
from src.monitoring import get_monitor
from src.prediction import ModelNotReadyError, get_predictor
from src.preprocessing import is_healthy, split_class_name
from src.registry import ModelRegistry
from src.retrain import get_job_manager, upload_summary

HOSTNAME = socket.gethostname()
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_BATCH_SIZE = 32

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm the model on startup, persist counters on shutdown.

    A cold ONNX session costs a second or two. Under load testing that would
    surface as a misleading p99 outlier, so the cost is paid here instead of by
    the first real request.
    """
    ensure_dirs()
    get_monitor()
    try:
        predictor = get_predictor()
        predictor._ensure_session()
        predictor._ensure_head()
        print(f"[startup] model {predictor.version} ready on {HOSTNAME}")
    except Exception as exc:
        # A container with no artifacts yet is a valid state -- /health reports
        # it as degraded and the service still answers. Crashing here would
        # make the deployment unrecoverable without a rebuild.
        print(f"[startup] model not ready: {exc}")

    yield

    get_monitor().persist()


app = FastAPI(
    title="Crop Disease Classification API",
    description=(
        "Classifies crop leaf disease across 38 classes and 14 crop species. "
        "Serves a frozen MobileNetV2 ONNX backbone with a hot-swappable "
        "scikit-learn head, so retraining runs in-container in seconds."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# The Streamlit UI is served from a different origin (and a different Render
# service in production), so it needs CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================================
# Monitoring
# ==========================================================================

@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health() -> HealthResponse:
    """Liveness plus model readiness. Powers the UI uptime tab."""
    monitor = get_monitor()
    predictor = get_predictor()

    ready = predictor.ready
    return HealthResponse(
        status="healthy" if ready else "degraded",
        model_ready=ready,
        model_version=predictor.version,
        n_classes=len(predictor.class_names),
        uptime_seconds=round(monitor.uptime_seconds, 1),
        uptime_human=monitor.stats()["uptime_human"],
        started_at=monitor.started_at_iso,
        total_predictions=monitor.total_predictions,
        hostname=HOSTNAME,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["monitoring"])
def metrics() -> MetricsResponse:
    """Latency percentiles, throughput and drift state."""
    stats = get_monitor().stats()
    registry = ModelRegistry()
    active = registry.active_version()

    return MetricsResponse(
        **stats,
        hostname=HOSTNAME,
        model_version=registry.active,
        model_metrics={
            k: active.metrics.get(k) for k in
            ("accuracy", "f1_macro", "top5_accuracy", "precision_macro",
             "recall_macro", "roc_auc_macro", "pr_auc_macro")
        } if active and active.metrics else None,
    )


@app.get("/", tags=["monitoring"])
def root() -> dict:
    return {
        "service": "crop-disease-classification",
        "docs": "/docs",
        "health": "/health",
        "hostname": HOSTNAME,
    }


# ==========================================================================
# Prediction
# ==========================================================================

def _validate_image(file: UploadFile, contents: bytes) -> None:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            400, f"unsupported file type {suffix!r}; expected one of "
                 f"{sorted(ALLOWED_SUFFIXES)}")
    if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"file exceeds the {MAX_UPLOAD_MB} MB limit")
    if not contents:
        raise HTTPException(400, "empty file")


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(file: UploadFile = File(...), top_k: int = 5) -> PredictionResponse:
    """Classify a single leaf image."""
    monitor = get_monitor()
    contents = await file.read()
    _validate_image(file, contents)

    try:
        result = get_predictor().predict(contents, top_k=top_k)
    except ModelNotReadyError as exc:
        monitor.record_error()
        # 503, not 500: the caller should retry once artifacts are in place.
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        monitor.record_error()
        raise HTTPException(400, f"could not process image: {exc}") from exc

    monitor.record_prediction(
        confidence=result["confidence"],
        latency_ms=result["latency_ms"],
        class_name=result["prediction"],
    )
    return PredictionResponse(**result, filename=file.filename)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
async def predict_batch(files: list[UploadFile] = File(...),
                        top_k: int = 5) -> BatchPredictionResponse:
    """Classify several images in one backbone pass."""
    if not files:
        raise HTTPException(400, "no files supplied")
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            413, f"batch of {len(files)} exceeds the {MAX_BATCH_SIZE}-image limit")

    monitor = get_monitor()
    payloads, names = [], []
    for file in files:
        contents = await file.read()
        _validate_image(file, contents)
        payloads.append(contents)
        names.append(file.filename)

    try:
        results = get_predictor().predict_batch(payloads, top_k=top_k)
    except ModelNotReadyError as exc:
        monitor.record_error()
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        monitor.record_error()
        raise HTTPException(400, f"could not process batch: {exc}") from exc

    responses = []
    for name, result in zip(names, results):
        monitor.record_prediction(result["confidence"], result["latency_ms"],
                                  result["prediction"])
        responses.append(PredictionResponse(**result, filename=name))

    return BatchPredictionResponse(
        count=len(responses),
        total_latency_ms=round(sum(r.latency_ms for r in responses), 2),
        results=responses,
    )


@app.get("/classes", response_model=ClassListResponse, tags=["prediction"])
def classes() -> ClassListResponse:
    """The label set, split into crop and condition."""
    if not Path(CLASS_NAMES_PATH).exists():
        raise HTTPException(503, "class names not available")

    names = json.loads(Path(CLASS_NAMES_PATH).read_text())
    entries = []
    for index, name in enumerate(names):
        crop, condition = split_class_name(name)
        entries.append({
            "index": index, "class_name": name,
            "crop": crop, "condition": condition,
            "healthy": is_healthy(name),
        })
    return ClassListResponse(n_classes=len(names), classes=entries)


# ==========================================================================
# Data upload
# ==========================================================================

@app.post("/upload", response_model=UploadResponse, tags=["data"])
async def upload(files: list[UploadFile] = File(...),
                 class_name: str = Form(...)) -> UploadResponse:
    """Stage labelled images for the next retrain.

    A class label is required. Guessing labels from filenames would let bad
    data into the replay buffer, where it would corrupt every future retrain --
    so unknown classes are rejected rather than inferred.
    """
    ensure_dirs()

    if not Path(CLASS_NAMES_PATH).exists():
        raise HTTPException(503, "class names not available; cannot validate label")
    known = set(json.loads(Path(CLASS_NAMES_PATH).read_text()))
    if class_name not in known:
        raise HTTPException(
            400, f"unknown class {class_name!r}. See GET /classes for valid labels.")

    destination = Path(UPLOAD_DIR) / class_name
    destination.mkdir(parents=True, exist_ok=True)

    accepted, errors, saved = 0, [], []
    for file in files:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            errors.append(f"{file.filename}: unsupported type {suffix!r}")
            continue

        contents = await file.read()
        if not contents:
            errors.append(f"{file.filename}: empty file")
            continue
        if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
            errors.append(f"{file.filename}: exceeds {MAX_UPLOAD_MB} MB")
            continue

        # Prefix with a short uuid so repeated uploads of the same filename do
        # not silently overwrite one another.
        stored = f"{uuid.uuid4().hex[:8]}_{Path(file.filename).name}"
        (destination / stored).write_bytes(contents)
        saved.append({"filename": file.filename, "class_name": class_name,
                      "saved_as": stored})
        accepted += 1

    summary = upload_summary()
    trigger = get_monitor().check_retrain_trigger()

    return UploadResponse(
        accepted=accepted,
        rejected=len(errors),
        errors=errors,
        files=saved,
        staged_total=summary["total_images"],
        retrain_trigger=trigger,
    )


@app.get("/uploads/summary", response_model=UploadSummaryResponse, tags=["data"])
def uploads_summary() -> UploadSummaryResponse:
    """What is staged and whether that is enough to warrant a retrain."""
    summary = upload_summary()
    return UploadSummaryResponse(
        total_images=summary["total_images"],
        n_classes=summary["n_classes"],
        per_class=summary["per_class"],
        retrain_trigger=get_monitor().check_retrain_trigger(),
    )


# ==========================================================================
# Retraining and model management
# ==========================================================================

@app.post("/retrain", response_model=RetrainJobResponse, tags=["model"])
def retrain(head_kind: str = "logreg",
            force: bool = False,
            keep_uploads: bool = False) -> RetrainJobResponse:
    """Trigger retraining.

    Returns immediately with a job id; poll ``/retrain/status/{job_id}``.
    Holding the request open would time out behind most proxies and would tie
    up a worker that should be serving predictions.

    Without ``force``, the automatic trigger conditions must be satisfied --
    so the endpoint reflects the same policy the pipeline applies on its own.
    """
    monitor = get_monitor()
    trigger = monitor.check_retrain_trigger()

    if not force and not trigger["should_retrain"]:
        raise HTTPException(
            409,
            {
                "message": "retraining conditions not met; pass force=true to override",
                "trigger": trigger,
            },
        )

    job = get_job_manager().submit(
        head_kind=head_kind,
        consume_uploads=not keep_uploads,
        triggered_by="forced" if force else "threshold",
    )
    if job.get("status") == "already_running":
        return RetrainJobResponse(
            job_id=job["job_id"], status="running", progress=0.0,
            message=job["message"], submitted_at="")

    monitor.record_retrain()
    return RetrainJobResponse(**job)


@app.get("/retrain/status/{job_id}", response_model=RetrainJobResponse, tags=["model"])
def retrain_status(job_id: str) -> RetrainJobResponse:
    job = get_job_manager().get(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job {job_id!r}")
    return RetrainJobResponse(**job)


@app.get("/retrain/jobs", tags=["model"])
def retrain_jobs(limit: int = 10) -> dict:
    return {"jobs": get_job_manager().recent(limit)}


@app.get("/retrain/trigger", tags=["model"])
def retrain_trigger() -> dict:
    """Whether the pipeline currently wants to retrain, and why."""
    return get_monitor().check_retrain_trigger()


@app.get("/models", response_model=ModelListResponse, tags=["model"])
def models() -> ModelListResponse:
    """Version history with metrics."""
    registry = ModelRegistry()
    history = registry.history()
    return ModelListResponse(active=registry.active, count=len(history),
                             versions=history)


@app.post("/models/activate/{version}", tags=["model"])
def activate_model(version: str) -> dict:
    """Roll back (or forward) to a specific version."""
    registry = ModelRegistry()
    try:
        entry = registry.activate(version)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc

    get_predictor().reload()
    return {"activated": version, "summary": entry.summary()}


@app.exception_handler(ModelNotReadyError)
def model_not_ready_handler(request, exc: ModelNotReadyError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0",
                port=int(os.getenv("PORT", "8000")), reload=True)
