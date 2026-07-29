"""Pydantic response models for the prediction API.

Defined explicitly rather than returning bare dicts so that FastAPI's generated
OpenAPI docs at /docs are usable as the API contract -- which matters because
the Streamlit UI is written against it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ClassPrediction(BaseModel):
    rank: int
    class_index: int
    class_name: str = Field(description='Raw label, e.g. "Tomato___Late_blight"')
    crop: str = Field(description='Crop species, e.g. "Tomato"')
    condition: str = Field(description='Health condition, e.g. "Late blight"')
    healthy: bool
    probability: float


class PredictionResponse(BaseModel):
    prediction: str
    crop: str
    condition: str
    healthy: bool
    confidence: float
    top_k: list[ClassPrediction]
    model_version: str | None
    latency_ms: float
    filename: str | None = None


class BatchPredictionResponse(BaseModel):
    count: int
    total_latency_ms: float
    results: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str = Field(description='"healthy", "degraded" or "starting"')
    model_ready: bool
    model_version: str | None
    n_classes: int
    uptime_seconds: float
    uptime_human: str
    started_at: str
    total_predictions: int
    hostname: str = Field(
        description="Container hostname; distinguishes replicas during load testing")


class LatencyStats(BaseModel):
    count: int
    mean: float
    p50: float
    p95: float
    p99: float
    max: float


class ConfidenceStats(BaseModel):
    window_size: int
    rolling_mean: float | None
    threshold: float


class ClassCount(BaseModel):
    class_name: str
    count: int


class MetricsResponse(BaseModel):
    uptime_seconds: float
    uptime_human: str
    started_at: str
    total_predictions: int
    total_errors: int
    error_rate: float
    requests_per_second_1m: float
    latency_ms: LatencyStats
    confidence: ConfidenceStats
    top_predicted_classes: list[ClassCount]
    last_prediction_at: str | None
    last_retrain_at: str | None
    retrain_count: int
    hostname: str
    model_version: str | None
    model_metrics: dict[str, Any] | None = None


class UploadedFile(BaseModel):
    filename: str
    class_name: str
    saved_as: str


class UploadResponse(BaseModel):
    accepted: int
    rejected: int
    errors: list[str]
    files: list[UploadedFile]
    staged_total: int
    retrain_trigger: dict[str, Any]


class UploadSummaryResponse(BaseModel):
    total_images: int
    n_classes: int
    per_class: list[ClassCount]
    retrain_trigger: dict[str, Any]


class RetrainJobResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    submitted_at: str
    result: dict[str, Any] | None = None
    error: str | None = None


class ModelVersionSummary(BaseModel):
    version: str
    created_at: str
    source: str
    head_kind: str
    n_train_samples: int
    n_new_samples: int
    notes: str
    accuracy: float | None
    f1_macro: float | None
    top5_accuracy: float | None
    mean_confidence: float | None
    active: bool


class ModelListResponse(BaseModel):
    active: str | None
    count: int
    versions: list[ModelVersionSummary]


class ClassListResponse(BaseModel):
    n_classes: int
    classes: list[dict[str, Any]]
