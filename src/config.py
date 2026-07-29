"""Central configuration: paths, dataset constants and model hyper-parameters.

Every other module imports from here so that paths are defined exactly once.
Anything that differs between local development and the deployed container is
read from an environment variable with a sensible local default.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Resolved from this file so the package works regardless of the caller's cwd
# (the notebook runs from notebook/, the API from the container root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"
RAW_DIR = DATA_DIR / "raw"

MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
BACKBONE_PATH = MODEL_DIR / "backbone_mobilenetv2.onnx"
REPLAY_BUFFER_PATH = MODEL_DIR / "train_embeddings.npz"
# Held-out embeddings shipped with the model so the container can score a
# retrained head without needing the image dataset present.
EVAL_EMBEDDINGS_PATH = MODEL_DIR / "eval_embeddings.npz"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
REGISTRY_PATH = MODEL_DIR / "registry.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOAD_TEST_DIR = REPORTS_DIR / "load_test"

# Runtime state. Kept out of MODEL_DIR so a model-directory sync never clobbers it.
STATE_DIR = Path(os.getenv("STATE_DIR", PROJECT_ROOT / "state"))
PREDICTION_LOG_PATH = STATE_DIR / "prediction_log.jsonl"
MONITOR_STATE_PATH = STATE_DIR / "monitoring_state.json"
# Retraining job state, mirrored to disk so a job survives the worker that ran
# it. Retraining is the container's peak-memory operation; a worker killed
# mid-job would otherwise take its job table with it and the job would simply
# vanish, with health checks still green.
RETRAIN_JOBS_PATH = STATE_DIR / "retrain_jobs.json"

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
DATASET_REPO = "https://github.com/spMohanty/PlantVillage-Dataset.git"
# Only the 256x256 colour variant. The repo also ships `grayscale` and
# `segmented` copies of every image; a sparse checkout avoids downloading
# roughly three times more data than we need.
DATASET_SUBDIR = "raw/color"

N_CLASSES = 38
# PlantVillage class directories are named "<Crop>___<Condition>".
CLASS_SEPARATOR = "___"

# Stratified split, applied per class so every class keeps these proportions.
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
RANDOM_SEED = 42

# Images committed to git so the repo is runnable without the full download.
SAMPLE_PREFIX = "SAMPLE_"
SAMPLES_PER_CLASS = 5

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)          # MobileNetV2's native input resolution
EMBEDDING_DIM = 1280             # MobileNetV2 output width after global pooling
BATCH_SIZE = 32

# The replay buffer guards against catastrophic forgetting during retraining:
# a new head is always fitted on (buffer + newly uploaded images), never on the
# uploads alone. See src/retrain.py.
#
# Size measured against the full training set rather than guessed, because it
# trades accuracy for container memory (macro-F1 cost vs the full 38k split):
#
#     150/class   5,661 rows   14 MB   -0.0360
#     300/class  10,994 rows   28 MB   -0.0192   <- shipped
#     600/class  19,768 rows   38 MB   -0.0078
#    1000/class  27,480 rows   70 MB   -0.0008   diminishing returns
#
# 600 was tried first and OOM-killed the container: scikit-learn upcasts the
# buffer to float64 while fitting, so a 19,768x1280 matrix needs ~200 MB
# transient on top of a ~272 MB idle footprint, against a 512 MB limit.
# 300/class halves that and lands around 390 MB.
#
# The -0.0192 cost is NOT absorbed by the promotion tolerance below. It is a
# constant offset from training on a subset, not a regression caused by new
# data, so the registry compares candidates against a baseline fitted on this
# same buffer (`buffer_baseline_f1`). See ModelRegistry.should_promote.
REPLAY_SAMPLES_PER_CLASS = 300
# A smaller held-out sample used only to score retrained heads in-container.
EVAL_SAMPLES_PER_CLASS = 100

# --------------------------------------------------------------------------
# Retraining triggers
# --------------------------------------------------------------------------
# Fire a retrain when enough new labelled images have been staged...
RETRAIN_UPLOAD_THRESHOLD = 50
# ...or when the model's rolling mean confidence decays, which is our proxy for
# input drift. PlantVillage is lab-photographed on uniform backgrounds, so field
# photos are genuinely out-of-distribution -- see the EDA section of the notebook.
DRIFT_CONFIDENCE_THRESHOLD = 0.65
DRIFT_WINDOW_SIZE = 100

# A retrained head is only promoted if it does not regress macro-F1 on the
# held-out test set by more than this margin.
RETRAIN_MAX_F1_REGRESSION = 0.02

# --------------------------------------------------------------------------
# Service
# --------------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))


def ensure_dirs() -> None:
    """Create every directory the runtime writes to.

    Safe to call repeatedly; used by the API on startup because the deployed
    container starts from a fresh filesystem.
    """
    for path in (DATA_DIR, TRAIN_DIR, TEST_DIR, UPLOAD_DIR, MODEL_DIR,
                 REPORTS_DIR, FIGURES_DIR, LOAD_TEST_DIR, STATE_DIR):
        path.mkdir(parents=True, exist_ok=True)
