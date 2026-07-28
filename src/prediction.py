"""Inference: ONNX backbone + scikit-learn head.

This is the module the deployed API runs on, so it is deliberately austere --
`onnxruntime`, `numpy`, `scikit-learn` and `pillow`, nothing else. No
TensorFlow is imported anywhere in this file or in what it imports.

The backbone is loaded once and shared; the head is hot-swappable, because
retraining produces a new head that must take effect without restarting the
service.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.config import (
    BACKBONE_PATH,
    CLASS_NAMES_PATH,
    EMBEDDING_DIM,
    IMAGE_SIZE,
)
from src.preprocessing import load_image, preprocess_for_backbone, split_class_name
from src.registry import ModelRegistry


class ModelNotReadyError(RuntimeError):
    """Raised when artifacts are missing -- surfaced as HTTP 503, not 500.

    A cold container with no model yet is an expected state, not a bug, and the
    caller should retry rather than treat it as a failure.
    """


class Predictor:
    """Loads the frozen backbone and the active head, and serves predictions."""

    def __init__(self,
                 backbone_path: str | Path = BACKBONE_PATH,
                 class_names_path: str | Path = CLASS_NAMES_PATH,
                 registry: ModelRegistry | None = None,
                 intra_op_threads: int = 1):
        self.backbone_path = Path(backbone_path)
        self.class_names_path = Path(class_names_path)
        self.registry = registry or ModelRegistry()

        # Under Locust the container is already saturated by concurrent
        # requests, so extra intra-op threads only add contention. One thread
        # per session, N concurrent requests, is the better trade here.
        self.intra_op_threads = intra_op_threads

        self._lock = threading.Lock()
        self._session = None
        self._input_name: str | None = None
        self._head = None
        self._head_version: str | None = None
        self.class_names: list[str] = []

        self._load_class_names()

    # -- loading -----------------------------------------------------------

    def _load_class_names(self) -> None:
        if self.class_names_path.exists():
            self.class_names = json.loads(self.class_names_path.read_text())

    def _ensure_session(self):
        """Lazily create the ONNX session, once, under a lock."""
        if self._session is not None:
            return self._session

        with self._lock:
            if self._session is not None:
                return self._session
            if not self.backbone_path.exists():
                raise ModelNotReadyError(
                    f"backbone not found at {self.backbone_path}. "
                    "Run the notebook's export step first."
                )

            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = self.intra_op_threads
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._session = ort.InferenceSession(
                str(self.backbone_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            return self._session

    def _ensure_head(self):
        if self._head is not None:
            return self._head

        with self._lock:
            if self._head is not None:
                return self._head

            self.registry.load()
            head_path = self.registry.head_path()
            if head_path is None:
                raise ModelNotReadyError(
                    "no active classifier head registered. "
                    "Run the notebook's export step, or trigger a retrain."
                )

            import joblib

            self._head = joblib.load(head_path)
            self._head_version = self.registry.active
            return self._head

    def reload(self) -> str | None:
        """Swap in the newly-active head without restarting the service.

        The backbone is intentionally left alone -- it never changes, and
        rebuilding the ONNX session would add a multi-second stall to the
        first request after every retrain.
        """
        with self._lock:
            self._head = None
            self._head_version = None
        self._load_class_names()
        self.registry.load()
        self._ensure_head()
        return self._head_version

    @property
    def ready(self) -> bool:
        try:
            self._ensure_session()
            self._ensure_head()
            return True
        except (ModelNotReadyError, Exception):
            return False

    @property
    def version(self) -> str | None:
        return self._head_version or self.registry.active

    # -- inference ---------------------------------------------------------

    def embed(self, batch: np.ndarray) -> np.ndarray:
        """Run a preprocessed NHWC batch through the backbone."""
        session = self._ensure_session()
        return session.run(None, {self._input_name: batch.astype(np.float32)})[0]

    def embed_images(self, sources: Sequence[str | Path | bytes],
                     batch_size: int = 32) -> np.ndarray:
        """Load, preprocess and embed images, in chunks.

        Chunking is what keeps retraining inside the memory budget: embedding
        several thousand uploaded images at once would allocate far more than
        the container has.
        """
        if not sources:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        chunks = []
        for start in range(0, len(sources), batch_size):
            window = sources[start:start + batch_size]
            batch = preprocess_for_backbone(
                np.stack([load_image(s, IMAGE_SIZE) for s in window]))
            chunks.append(self.embed(batch))
        return np.concatenate(chunks).astype(np.float32)

    def predict(self, source: str | Path | bytes, top_k: int = 5) -> dict[str, Any]:
        """Classify a single image. This is the user-facing prediction path."""
        started = time.perf_counter()

        image = load_image(source, IMAGE_SIZE)
        batch = preprocess_for_backbone(image)
        embedding = self.embed(batch)

        head = self._ensure_head()
        proba = head.predict_proba(embedding)[0]

        result = self._format(proba, top_k)
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return result

    def predict_batch(self, sources: Sequence[str | Path | bytes],
                      top_k: int = 5) -> list[dict[str, Any]]:
        if not sources:
            return []

        started = time.perf_counter()
        batch = preprocess_for_backbone(
            np.stack([load_image(s, IMAGE_SIZE) for s in sources]))
        embeddings = self.embed(batch)

        head = self._ensure_head()
        probabilities = head.predict_proba(embeddings)

        elapsed_ms = (time.perf_counter() - started) * 1000
        per_image = round(elapsed_ms / len(sources), 2)

        results = []
        for proba in probabilities:
            item = self._format(proba, top_k)
            item["latency_ms"] = per_image
            results.append(item)
        return results

    def _format(self, proba: np.ndarray, top_k: int) -> dict[str, Any]:
        """Turn a probability vector into a response.

        The crop/condition split is done here rather than in the UI because it
        is what makes the answer actionable -- "Tomato / Late blight" is a
        finding a loan officer can act on; "Tomato___Late_blight" is a label.
        """
        head = self._head
        # LogisticRegression drops classes absent from its training data, so
        # `classes_` is the authority on which column means what -- never assume
        # column i corresponds to class i.
        class_indices = getattr(head, "classes_", np.arange(len(proba)))

        order = np.argsort(proba)[::-1][:top_k]
        predictions = []
        for rank, position in enumerate(order, 1):
            label_index = int(class_indices[position])
            name = (self.class_names[label_index]
                    if label_index < len(self.class_names) else str(label_index))
            crop, condition = split_class_name(name)
            predictions.append({
                "rank": rank,
                "class_index": label_index,
                "class_name": name,
                "crop": crop,
                "condition": condition,
                "healthy": "healthy" in condition.lower(),
                "probability": round(float(proba[position]), 6),
            })

        best = predictions[0]
        return {
            "prediction": best["class_name"],
            "crop": best["crop"],
            "condition": best["condition"],
            "healthy": best["healthy"],
            "confidence": best["probability"],
            "top_k": predictions,
            "model_version": self.version,
        }

    def info(self) -> dict[str, Any]:
        active = self.registry.active_version()
        return {
            "model_version": self.version,
            "n_classes": len(self.class_names),
            "embedding_dim": EMBEDDING_DIM,
            "image_size": list(IMAGE_SIZE),
            "backbone": self.backbone_path.name,
            "backbone_mb": round(self.backbone_path.stat().st_size / 1e6, 2)
            if self.backbone_path.exists() else None,
            "metrics": active.metrics.get("accuracy") if active else None,
        }


# --------------------------------------------------------------------------
# Process-wide singleton
#
# The API creates one Predictor and reuses it. Loading the ONNX session per
# request would dominate latency and multiply memory use under load.
# --------------------------------------------------------------------------

_predictor: Predictor | None = None
_predictor_lock = threading.Lock()


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                _predictor = Predictor()
    return _predictor


def predict_image(source: str | Path | bytes, top_k: int = 5) -> dict[str, Any]:
    """Convenience wrapper: classify one image with the shared predictor."""
    return get_predictor().predict(source, top_k=top_k)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify a leaf image.")
    parser.add_argument("image", help="path to an image file")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    outcome = predict_image(args.image, top_k=args.top_k)
    print(f"\n  {outcome['crop']} -- {outcome['condition']}")
    print(f"  confidence {outcome['confidence']:.1%}   "
          f"({outcome['latency_ms']} ms, model {outcome['model_version']})\n")
    for item in outcome["top_k"]:
        print(f"    {item['rank']}. {item['class_name']:<45} {item['probability']:.4f}")
