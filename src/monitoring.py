"""Service monitoring: uptime, latency, throughput and the drift trigger.

Two jobs. First, supply the numbers the UI's uptime tab shows and the load test
reports -- request counts, latency percentiles, requests per second. Second,
decide *when the model needs retraining*, which is the part the assignment
cares about: a pipeline that only retrains when a human remembers to press a
button is not really a pipeline.

Two independent conditions fire a retrain:

* **Volume** -- enough newly labelled images have been staged to be worth
  learning from.
* **Confidence drift** -- the model's rolling mean confidence has decayed.
  This one follows directly from the EDA: PlantVillage is photographed on
  uniform laboratory backgrounds, so real field photos are out-of-distribution
  and the model should be expected to grow less certain over time. Falling
  confidence is the earliest signal available without ground-truth labels.

Confidence drift is a *proxy*, not proof -- a confidently wrong model will not
trip it. It is used to schedule retraining, never to claim the model is fine.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import (
    DRIFT_CONFIDENCE_THRESHOLD,
    DRIFT_WINDOW_SIZE,
    MONITOR_STATE_PATH,
    RETRAIN_UPLOAD_THRESHOLD,
    UPLOAD_DIR,
)


class ServiceMonitor:
    """Thread-safe in-process metrics.

    Every counter is guarded by a lock because uvicorn serves requests
    concurrently, and the load test exists specifically to hammer this path.
    """

    def __init__(self,
                 window_size: int = DRIFT_WINDOW_SIZE,
                 confidence_threshold: float = DRIFT_CONFIDENCE_THRESHOLD,
                 upload_threshold: int = RETRAIN_UPLOAD_THRESHOLD,
                 state_path: str | Path = MONITOR_STATE_PATH):
        self.started_at = time.time()
        self.started_at_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.upload_threshold = upload_threshold
        self.state_path = Path(state_path)

        self._lock = threading.Lock()

        # Bounded deques: the service must run indefinitely without growing.
        self._latencies: deque[float] = deque(maxlen=2000)
        self._confidences: deque[float] = deque(maxlen=window_size)
        self._recent_timestamps: deque[float] = deque(maxlen=1000)

        self.total_predictions = 0
        self.total_errors = 0
        self.class_counts: Counter[str] = Counter()
        self.last_prediction_at: str | None = None
        self.last_retrain_at: str | None = None
        self.retrain_count = 0

    # -- recording ---------------------------------------------------------

    def record_prediction(self, confidence: float, latency_ms: float,
                          class_name: str | None = None) -> None:
        with self._lock:
            self.total_predictions += 1
            self._latencies.append(latency_ms)
            self._confidences.append(confidence)
            self._recent_timestamps.append(time.time())
            if class_name:
                self.class_counts[class_name] += 1
            self.last_prediction_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def record_error(self) -> None:
        with self._lock:
            self.total_errors += 1

    def record_retrain(self) -> None:
        with self._lock:
            self.retrain_count += 1
            self.last_retrain_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            # The drift window described the *previous* model; carrying it over
            # would let a stale signal immediately re-trigger a retrain.
            self._confidences.clear()

    # -- derived stats -----------------------------------------------------

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.started_at

    @staticmethod
    def _percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(int(round((q / 100.0) * (len(ordered) - 1))), len(ordered) - 1)
        return round(ordered[index], 2)

    def _requests_per_second(self, window: float = 60.0) -> float:
        cutoff = time.time() - window
        recent = [t for t in self._recent_timestamps if t >= cutoff]
        if len(recent) < 2:
            return 0.0
        return round(len(recent) / window, 3)

    def stats(self) -> dict[str, Any]:
        """Everything the /metrics endpoint returns."""
        with self._lock:
            latencies = list(self._latencies)
            confidences = list(self._confidences)
            top_classes = self.class_counts.most_common(10)
            total, errors = self.total_predictions, self.total_errors
            rps = self._requests_per_second()

        mean_confidence = (sum(confidences) / len(confidences)) if confidences else None

        return {
            "uptime_seconds": round(self.uptime_seconds, 1),
            "uptime_human": _format_duration(self.uptime_seconds),
            "started_at": self.started_at_iso,
            "total_predictions": total,
            "total_errors": errors,
            "error_rate": round(errors / total, 4) if total else 0.0,
            "requests_per_second_1m": rps,
            "latency_ms": {
                "count": len(latencies),
                "mean": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
                "p50": self._percentile(latencies, 50),
                "p95": self._percentile(latencies, 95),
                "p99": self._percentile(latencies, 99),
                "max": round(max(latencies), 2) if latencies else 0.0,
            },
            "confidence": {
                "window_size": len(confidences),
                "rolling_mean": round(mean_confidence, 4) if mean_confidence is not None else None,
                "threshold": self.confidence_threshold,
            },
            "top_predicted_classes": [
                {"class_name": name, "count": count} for name, count in top_classes
            ],
            "last_prediction_at": self.last_prediction_at,
            "last_retrain_at": self.last_retrain_at,
            "retrain_count": self.retrain_count,
        }

    # -- the retraining trigger -------------------------------------------

    def count_staged_uploads(self, upload_dir: str | Path = UPLOAD_DIR) -> int:
        upload_dir = Path(upload_dir)
        if not upload_dir.is_dir():
            return 0
        return sum(
            1 for p in upload_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    def check_retrain_trigger(self, upload_dir: str | Path = UPLOAD_DIR) -> dict[str, Any]:
        """Evaluate both retraining conditions and report why.

        Returns the reasoning, not just a boolean, so the UI can explain to the
        user *why* the system wants to retrain rather than presenting an
        unexplained button.
        """
        staged = self.count_staged_uploads(upload_dir)

        with self._lock:
            confidences = list(self._confidences)

        volume_triggered = staged >= self.upload_threshold

        # Only judge drift on a full window: the mean of three predictions is
        # noise, and acting on it would retrain constantly after every restart.
        rolling_mean = (sum(confidences) / len(confidences)) if confidences else None
        window_full = len(confidences) >= self.window_size
        drift_triggered = bool(
            window_full and rolling_mean is not None
            and rolling_mean < self.confidence_threshold
        )

        reasons = []
        if volume_triggered:
            reasons.append(
                f"{staged} staged images reached the {self.upload_threshold}-image threshold")
        if drift_triggered:
            reasons.append(
                f"rolling mean confidence {rolling_mean:.3f} fell below "
                f"{self.confidence_threshold:.2f} over the last {len(confidences)} predictions")

        return {
            "should_retrain": volume_triggered or drift_triggered,
            "reasons": reasons,
            "triggers": {
                "volume": {
                    "triggered": volume_triggered,
                    "staged_images": staged,
                    "threshold": self.upload_threshold,
                },
                "drift": {
                    "triggered": drift_triggered,
                    "rolling_mean_confidence": round(rolling_mean, 4) if rolling_mean is not None else None,
                    "threshold": self.confidence_threshold,
                    "window_filled": window_full,
                    "window_size": len(confidences),
                },
            },
        }

    # -- persistence -------------------------------------------------------

    def persist(self) -> None:
        """Save cumulative counters so a restart does not zero the history.

        Best-effort: monitoring must never take the prediction service down,
        so any failure here is swallowed.
        """
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps({
                "total_predictions": self.total_predictions,
                "total_errors": self.total_errors,
                "retrain_count": self.retrain_count,
                "last_retrain_at": self.last_retrain_at,
                "class_counts": dict(self.class_counts),
                "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }, indent=2))
        except OSError:
            pass

    def restore(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text())
        except (OSError, json.JSONDecodeError):
            return

        with self._lock:
            self.total_predictions = data.get("total_predictions", 0)
            self.total_errors = data.get("total_errors", 0)
            self.retrain_count = data.get("retrain_count", 0)
            self.last_retrain_at = data.get("last_retrain_at")
            self.class_counts = Counter(data.get("class_counts", {}))


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


# Process-wide singleton, mirroring the predictor.
_monitor: ServiceMonitor | None = None
_monitor_lock = threading.Lock()


def get_monitor() -> ServiceMonitor:
    global _monitor
    if _monitor is None:
        with _monitor_lock:
            if _monitor is None:
                _monitor = ServiceMonitor()
                _monitor.restore()
    return _monitor
