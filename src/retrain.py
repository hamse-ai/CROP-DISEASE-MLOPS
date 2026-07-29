"""Retraining: embed staged uploads, refit the head, gate on macro-F1.

This runs *inside the deployed container*, which is the whole reason the model
was split into a frozen backbone and a light head. Retraining here never
touches the backbone; it embeds the newly uploaded images through it and fits a
fresh classifier on those embeddings combined with the replay buffer.

The sequence:

1. Collect staged uploads from ``data/uploads/<class_name>/``.
2. Embed them through the frozen ONNX backbone, in chunks.
3. Concatenate with the replay buffer -- *never* train on uploads alone.
4. Fit a new head.
5. Score it on the held-out evaluation embeddings shipped with the model.
6. Promote only if macro-F1 has not regressed beyond tolerance.
7. Move consumed uploads out of the staging area.

Step 6 is the important one. An automated retrain is the easiest way for a
pipeline to quietly make a model worse, so a new head has to earn promotion.
A rejected head is still registered, just not activated -- keeping the evidence
of what went wrong.
"""

from __future__ import annotations

import gc
import json
import shutil
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Importable as `src.retrain` by the API, and runnable directly as
# `python src/retrain.py` -- the latter needs the project root on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CLASS_NAMES_PATH,
    RETRAIN_JOBS_PATH,
    EVAL_EMBEDDINGS_PATH,
    MODEL_DIR,
    RANDOM_SEED,
    REPLAY_BUFFER_PATH,
    REPLAY_SAMPLES_PER_CLASS,
    RETRAIN_MAX_F1_REGRESSION,
    TRAIN_DIR,
    UPLOAD_DIR,
)
from src.model import build_replay_buffer, evaluate_predictions, save_head, train_head
from src.preprocessing import list_images
from src.registry import ModelRegistry

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class RetrainError(RuntimeError):
    """Retraining could not proceed (no data, missing artifacts)."""


# ==========================================================================
# Staged uploads
# ==========================================================================

def collect_uploads(upload_dir: str | Path = UPLOAD_DIR,
                    class_names: list[str] | None = None
                    ) -> dict[str, list[Path]]:
    """Map class name -> staged image paths.

    Uploads are organised by directory name, which must match a known class.
    Anything in an unrecognised directory is ignored rather than guessed at:
    silently mislabelling user data would poison the replay buffer for every
    future retrain.
    """
    upload_dir = Path(upload_dir)
    if not upload_dir.is_dir():
        return {}

    known = set(class_names) if class_names else None
    staged: dict[str, list[Path]] = {}

    for class_dir in sorted(p for p in upload_dir.iterdir() if p.is_dir()):
        if known is not None and class_dir.name not in known:
            print(f"  ignoring upload directory with unknown class: {class_dir.name}")
            continue
        images = [p for p in sorted(class_dir.iterdir())
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        if images:
            staged[class_dir.name] = images

    return staged


def upload_summary(upload_dir: str | Path = UPLOAD_DIR) -> dict[str, Any]:
    """What is currently staged, for the UI and the trigger check."""
    staged = collect_uploads(upload_dir)
    per_class = [{"class_name": name, "count": len(paths)}
                 for name, paths in sorted(staged.items())]
    return {
        "total_images": sum(len(p) for p in staged.values()),
        "n_classes": len(staged),
        "per_class": per_class,
    }


def clear_uploads(upload_dir: str | Path = UPLOAD_DIR,
                  archive_to: str | Path | None = TRAIN_DIR) -> int:
    """Consume staged uploads once they have been learned from.

    Moved into the training set rather than deleted, so the images remain
    available for the next full offline retrain and the upload directory
    accurately reflects "not yet learned from".
    """
    upload_dir = Path(upload_dir)
    moved = 0
    for class_dir in sorted(p for p in upload_dir.iterdir() if p.is_dir()):
        for image in list(class_dir.iterdir()):
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if archive_to is not None:
                destination_dir = Path(archive_to) / class_dir.name
                destination_dir.mkdir(parents=True, exist_ok=True)
                destination = destination_dir / f"upload_{image.name}"
                if destination.exists():
                    destination = destination_dir / f"upload_{uuid.uuid4().hex[:8]}_{image.name}"
                shutil.move(str(image), str(destination))
            else:
                image.unlink()
            moved += 1
        if not any(class_dir.iterdir()):
            class_dir.rmdir()
    return moved


# ==========================================================================
# Embedding stores
# ==========================================================================

def load_replay_buffer(path: str | Path = REPLAY_BUFFER_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Load the stratified training embeddings.

    Stored float16 to halve the file; restored to float32 because scikit-learn
    fits in float64 internally and float16 input just adds a conversion.
    """
    path = Path(path)
    if not path.exists():
        raise RetrainError(
            f"replay buffer missing at {path}. Retraining without it would fit "
            "the head on uploaded images alone and destroy every other class."
        )
    data = np.load(path)
    return data["X"].astype(np.float32), data["y"].astype(np.int64)


def build_training_matrix(X_new: np.ndarray, y_new: np.ndarray,
                          path: str | Path = REPLAY_BUFFER_PATH
                          ) -> tuple[np.ndarray, np.ndarray, int]:
    """Combine the replay buffer with new embeddings in one allocation.

    The obvious version -- expand the buffer to float32, then
    ``np.concatenate([X_buffer, X_new])`` -- holds the buffer and the combined
    matrix simultaneously, roughly 200 MB for a 600-per-class buffer. Inside a
    512 MB container that is the difference between retraining and being OOM
    killed, so the destination is allocated once and the float16 buffer is
    converted directly into its first rows.
    """
    path = Path(path)
    if not path.exists():
        raise RetrainError(
            f"replay buffer missing at {path}. Retraining without it would fit "
            "the head on uploaded images alone and destroy every other class."
        )

    with np.load(path) as data:
        X_buffer = data["X"]                     # left as float16
        y_buffer = data["y"].astype(np.int64)

        n_buffer, n_new = len(X_buffer), len(X_new)
        X = np.empty((n_buffer + n_new, X_buffer.shape[1]), dtype=np.float32)
        X[:n_buffer] = X_buffer                  # float16 -> float32 in place
        X[n_buffer:] = X_new

    y = np.concatenate([y_buffer, y_new.astype(np.int64)])
    return X, y, n_buffer


def load_eval_embeddings(path: str | Path = EVAL_EMBEDDINGS_PATH):
    """Load held-out embeddings used to score a candidate head."""
    path = Path(path)
    if not path.exists():
        return None, None
    data = np.load(path)
    return data["X"].astype(np.float32), data["y"].astype(np.int64)


def save_embeddings(X: np.ndarray, y: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X.astype(np.float16), y=y.astype(np.int64))
    return path


def update_replay_buffer(X_new: np.ndarray, y_new: np.ndarray,
                         per_class: int = REPLAY_SAMPLES_PER_CLASS,
                         path: str | Path = REPLAY_BUFFER_PATH) -> tuple[int, int]:
    """Fold newly learned embeddings into the buffer, keeping it bounded.

    Without the re-subsample the buffer would grow without limit across
    retrains and eventually exceed the container's memory.

    **Everything here stays float16.** The first version took float32 arrays,
    concatenated them and re-subsampled, holding three float32 copies of the
    buffer at once plus zlib's compression buffers. That is what actually
    killed the worker in testing -- not the fit, which adds only ~18 MB. The
    peak landed *after* the model had been registered, so the retrain appeared
    to succeed and then the process vanished at 90%.

    float16 is the buffer's storage dtype anyway, so converting to float32 to
    shuffle rows was pure waste.
    """
    with np.load(path) as data:
        X_old = data["X"]                      # float16, ~21 MB at 300/class
        y_old = data["y"].astype(np.int64)

        X = np.concatenate([X_old, X_new.astype(np.float16)])
        y = np.concatenate([y_old, y_new.astype(np.int64)])

    X, y = build_replay_buffer(X, y, per_class=per_class)
    save_embeddings(X, y, path)
    return len(X), int(len(np.unique(y)))


# ==========================================================================
# The retraining run
# ==========================================================================

def run_retraining(head_kind: str | None = None,
                   upload_dir: str | Path = UPLOAD_DIR,
                   consume_uploads: bool = True,
                   max_regression: float = RETRAIN_MAX_F1_REGRESSION,
                   progress: Callable[[str, float], None] | None = None,
                   triggered_by: str = "manual") -> dict[str, Any]:
    """Run one retraining cycle. Returns a report suitable for the API."""
    def step(message: str, fraction: float) -> None:
        print(f"  [{fraction:>5.0%}] {message}")
        if progress is not None:
            progress(message, fraction)

    started = datetime.now(timezone.utc)
    registry = ModelRegistry()

    # Inherit the active version's architecture unless one is named explicitly.
    # Defaulting to a fixed kind silently swapped the deployed head type on
    # every retrain (mlp -> logreg here), which cost ~0.007 macro-F1 on its own
    # and made the promotion gate reject a change it should never have seen.
    if head_kind is None:
        active = registry.active_version()
        head_kind = active.head_kind if active else "logreg"

    class_names = (json.loads(Path(CLASS_NAMES_PATH).read_text())
                   if Path(CLASS_NAMES_PATH).exists() else None)

    # -- 1. staged uploads ------------------------------------------------
    step("collecting staged uploads", 0.05)
    staged = collect_uploads(upload_dir, class_names)
    n_new = sum(len(paths) for paths in staged.values())
    if n_new == 0:
        raise RetrainError(
            "no staged images to retrain on. Upload labelled images first."
        )
    step(f"found {n_new} new images across {len(staged)} classes", 0.10)

    # -- 2. embed ---------------------------------------------------------
    from src.prediction import get_predictor

    predictor = get_predictor()
    label_of = {name: i for i, name in enumerate(class_names or [])}

    step("embedding uploaded images through the frozen backbone", 0.15)
    new_features, new_labels = [], []
    for i, (class_name, paths) in enumerate(sorted(staged.items()), 1):
        if class_name not in label_of:
            continue
        embeddings = predictor.embed_images([str(p) for p in paths])
        new_features.append(embeddings)
        new_labels.append(np.full(len(embeddings), label_of[class_name], dtype=np.int64))
        step(f"embedded {class_name} ({len(paths)} images)",
             0.15 + 0.35 * (i / max(len(staged), 1)))

    if not new_features:
        raise RetrainError("no uploaded images matched a known class")

    X_new = np.concatenate(new_features).astype(np.float32)
    y_new = np.concatenate(new_labels)

    # -- 3. combine with the replay buffer --------------------------------
    step("loading replay buffer", 0.55)
    X_train, y_train, n_buffer = build_training_matrix(X_new, y_new)
    step(f"training on {len(X_train):,} embeddings "
         f"({n_buffer:,} replay + {len(X_new):,} new)", 0.60)

    # -- 4. fit -----------------------------------------------------------
    step(f"fitting {head_kind} head", 0.65)
    head = train_head(X_train, y_train, kind=head_kind, seed=RANDOM_SEED)

    # Release the training matrix before scoring: it is the largest single
    # allocation in the container and is not needed past this point.
    del X_train
    gc.collect()

    # -- 5. score ---------------------------------------------------------
    step("evaluating candidate on held-out embeddings", 0.80)
    X_eval, y_eval = load_eval_embeddings()
    if X_eval is None:
        metrics: dict[str, Any] = {}
        promote, reason = True, "no held-out set available; promoting unverified"
    else:
        proba = head.predict_proba(X_eval)
        # Re-map columns to canonical class indices: the head only has columns
        # for classes it saw, which need not be all 38.
        n_classes = len(class_names) if class_names else proba.shape[1]
        aligned = np.zeros((len(proba), n_classes), dtype=np.float32)
        aligned[:, head.classes_.astype(int)] = proba

        metrics = evaluate_predictions(y_eval, aligned, class_names)
        promote, reason = registry.should_promote(metrics, max_regression)

    # -- 6. register ------------------------------------------------------
    step("registering new version", 0.90)
    version = registry.next_version()
    head_file = f"head_{version}.pkl"
    save_head(head, MODEL_DIR / head_file)

    previous = registry.active_version()
    before_metrics = dict(previous.metrics) if previous else {}

    registry.register(
        version=version,
        head_file=head_file,
        metrics=metrics,
        activate=promote,
        n_train_samples=int(n_buffer + len(X_new)),
        n_new_samples=int(len(X_new)),
        head_kind=head_kind,
        source="retrain",
        notes=f"Triggered by {triggered_by}. {reason}",
    )

    # -- 7. consume + refresh ---------------------------------------------
    moved = 0
    if promote:
        # Release the evaluation set first: this is the container's second
        # memory peak and it does not need to overlap with the first.
        del X_eval
        gc.collect()
        update_replay_buffer(X_new, y_new)
        gc.collect()
        predictor.reload()
        if consume_uploads:
            step("archiving consumed uploads", 0.95)
            moved = clear_uploads(upload_dir)

    step("done", 1.0)
    finished = datetime.now(timezone.utc)

    def delta(key: str) -> float | None:
        before, after = before_metrics.get(key), metrics.get(key)
        return round(after - before, 4) if (before is not None and after is not None) else None

    return {
        "version": version,
        "promoted": promote,
        "reason": reason,
        "triggered_by": triggered_by,
        "head_kind": head_kind,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "duration_seconds": round((finished - started).total_seconds(), 2),
        "n_new_images": int(len(X_new)),
        "n_replay_images": int(n_buffer),
        "n_train_images": int(n_buffer + len(X_new)),
        "n_uploads_archived": moved,
        "metrics_before": {k: before_metrics.get(k) for k in
                           ("accuracy", "f1_macro", "top5_accuracy", "mean_confidence")},
        "metrics_after": {k: metrics.get(k) for k in
                          ("accuracy", "f1_macro", "top5_accuracy", "mean_confidence")},
        "metrics_delta": {k: delta(k) for k in
                          ("accuracy", "f1_macro", "top5_accuracy", "mean_confidence")},
    }


# ==========================================================================
# Background jobs
#
# Retraining takes seconds, not milliseconds, so the API dispatches it to a
# thread and hands back a job id rather than holding the request open.
# ==========================================================================

class RetrainJobManager:
    """Tracks retraining jobs, allowing at most one at a time.

    Job state is mirrored to disk. Retraining is the most memory-hungry thing
    the container does, and a worker that gets OOM-killed mid-job takes its
    in-memory job table with it -- uvicorn silently respawns the worker, health
    checks keep passing, and the user's job simply returns "unknown job" with
    no error recorded anywhere. Persisting means a killed job is reported as
    failed instead of disappearing.
    """

    # A running job must refresh its heartbeat within this window or it is
    # treated as abandoned. Comfortably longer than the slowest observed
    # retrain step (embedding a large upload batch), so a healthy job is never
    # declared dead.
    STALL_TIMEOUT_SECONDS = 180

    def __init__(self, max_history: int = 20,
                 state_path: str | Path | None = None):
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []
        self._max_history = max_history
        self._running: str | None = None
        # Jobs this process is executing; their in-memory state is
        # authoritative and must never be overwritten from disk.
        self._own_jobs: set[str] = set()
        self._state_path = Path(state_path) if state_path else RETRAIN_JOBS_PATH
        self._restore()

    # -- persistence -------------------------------------------------------

    def _restore(self) -> None:
        """Load jobs from disk, marking genuinely abandoned ones as failed."""
        for job in self._read_disk_jobs():
            self._jobs[job["job_id"]] = job
            self._order.append(job["job_id"])

    def _read_disk_jobs(self) -> list[dict[str, Any]]:
        """Read persisted jobs, resolving the status of stale "running" ones.

        A job flagged "running" is *not* necessarily dead. With more than one
        worker -- or more than one replica behind the balancer -- the process
        reading this file is frequently not the one executing the job, and
        declaring it failed on sight reports a perfectly healthy retrain as
        having crashed. That bug was observed: a completed, promoted retrain
        was reported to the caller as "worker exited".

        A heartbeat distinguishes the two. Progress updates refresh
        ``updated_at``; only a running job that has gone quiet for longer than
        the stall timeout is treated as abandoned.
        """
        if not self._state_path.exists():
            return []
        try:
            data = json.loads(self._state_path.read_text())
        except (OSError, json.JSONDecodeError):
            return []

        now = datetime.now(timezone.utc)
        jobs = []
        for job in data.get("jobs", []):
            if job.get("status") == "running":
                stalled_for = self._seconds_since(job.get("updated_at"), now)
                if stalled_for is None or stalled_for > self.STALL_TIMEOUT_SECONDS:
                    job["status"] = "failed"
                    job["message"] = "interrupted -- worker exited"
                    job["error"] = (
                        f"no progress for {stalled_for:.0f}s"
                        if stalled_for is not None else "no heartbeat recorded"
                    ) + (
                        "; the worker running this job exited before it "
                        "finished (most likely killed for exceeding the "
                        "container memory limit during retraining)"
                    )
            jobs.append(job)
        return jobs

    @staticmethod
    def _seconds_since(timestamp: str | None, now: datetime) -> float | None:
        if not timestamp:
            return None
        try:
            return (now - datetime.fromisoformat(timestamp)).total_seconds()
        except ValueError:
            return None

    def _persist_locked(self) -> None:
        """Write job state, stamping heartbeats. Caller must hold the lock."""
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for job_id in self._order:
            if self._jobs[job_id].get("status") == "running":
                self._jobs[job_id]["updated_at"] = stamp
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(
                {"jobs": [self._jobs[j] for j in self._order]}, indent=2))
            tmp.replace(self._state_path)
        except OSError:
            # Never let bookkeeping take down the retraining path.
            pass

    @property
    def is_running(self) -> bool:
        return self._running is not None

    def submit(self, **kwargs) -> dict[str, Any]:
        """Start a retraining job unless one is already in flight.

        Concurrent retrains would race on the registry and the replay buffer,
        so a second request is refused rather than queued -- the caller gets
        the in-flight job id back and can poll that instead.
        """
        with self._lock:
            if self._running is not None:
                return {
                    "job_id": self._running,
                    "status": "already_running",
                    "message": "a retraining job is already in progress",
                }

            job_id = uuid.uuid4().hex[:12]
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "progress": 0.0,
                "message": "queued",
                "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "result": None,
                "error": None,
            }
            self._order.append(job_id)
            self._own_jobs.add(job_id)
            self._running = job_id
            self._trim()
            self._persist_locked()

        thread = threading.Thread(target=self._run, args=(job_id, kwargs), daemon=True)
        thread.start()
        return self._jobs[job_id]

    def _run(self, job_id: str, kwargs: dict[str, Any]) -> None:
        def progress(message: str, fraction: float) -> None:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job["message"] = message
                    job["progress"] = round(fraction, 3)
                    self._persist_locked()

        try:
            result = run_retraining(progress=progress, **kwargs)
            with self._lock:
                self._jobs[job_id].update(
                    status="completed", progress=1.0,
                    message=("promoted" if result["promoted"] else "rejected"),
                    result=result)
                self._persist_locked()
        except Exception as exc:
            with self._lock:
                self._jobs[job_id].update(
                    status="failed", message=str(exc), error=str(exc))
                self._persist_locked()
        finally:
            with self._lock:
                self._running = None

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            owned = job_id in self._own_jobs
        # A job this process is running is authoritative in memory. One it only
        # read from disk may have been advanced since by whoever owns it.
        if job is not None and owned:
            return job
        # Not ours -- another worker may have run it, or a previous incarnation
        # of this one did before being restarted.
        self._reload_from_disk()
        with self._lock:
            return self._jobs.get(job_id)

    def _reload_from_disk(self) -> None:
        """Pull in jobs owned by another worker or replica."""
        for job in self._read_disk_jobs():
            with self._lock:
                if job["job_id"] not in self._jobs:
                    self._jobs[job["job_id"]] = job
                    self._order.append(job["job_id"])

    def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            return [self._jobs[j] for j in reversed(self._order[-limit:])]

    def _trim(self) -> None:
        while len(self._order) > self._max_history:
            self._jobs.pop(self._order.pop(0), None)


_job_manager: RetrainJobManager | None = None
_job_lock = threading.Lock()


def get_job_manager() -> RetrainJobManager:
    global _job_manager
    if _job_manager is None:
        with _job_lock:
            if _job_manager is None:
                _job_manager = RetrainJobManager()
    return _job_manager


if __name__ == "__main__":
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description="Retrain the classifier head.")
    parser.add_argument("--head", default=None, choices=["logreg", "sgd", "mlp"],
                        help="head architecture; defaults to the active version's, "
                             "so a retrain never silently swaps it")
    parser.add_argument("--keep-uploads", action="store_true")
    args = parser.parse_args()

    report = run_retraining(head_kind=args.head,
                            consume_uploads=not args.keep_uploads,
                            triggered_by="cli")
    print(_json.dumps(report, indent=2))
