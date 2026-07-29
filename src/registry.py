"""Model registry: version history, promotion and rollback.

Every retrain produces a *new* head rather than overwriting the current one.
Because a head is only a few hundred kilobytes, keeping the full history costs
almost nothing and buys two things that matter in production: an audit trail of
how the model changed as field data arrived, and instant rollback when a
retrain turns out worse than the version it replaced.

The registry is a single JSON file so it stays inspectable by hand -- useful
when the service is running somewhere you cannot attach a debugger.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import MODEL_DIR, REGISTRY_PATH


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ModelVersion:
    """One trained head and everything needed to judge or restore it."""

    version: str
    head_file: str
    created_at: str = field(default_factory=_utc_now)
    n_train_samples: int = 0
    n_new_samples: int = 0
    head_kind: str = "logreg"
    source: str = "notebook"          # "notebook" | "retrain"
    notes: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Compact view for API responses and the UI version table.

        Deliberately drops `per_class` and `confusion_matrix` -- a 38x38 matrix
        per version would dominate every response payload.
        """
        return {
            "version": self.version,
            "created_at": self.created_at,
            "source": self.source,
            "head_kind": self.head_kind,
            "n_train_samples": self.n_train_samples,
            "n_new_samples": self.n_new_samples,
            "notes": self.notes,
            "accuracy": self.metrics.get("accuracy"),
            "f1_macro": self.metrics.get("f1_macro"),
            "top5_accuracy": self.metrics.get("top5_accuracy"),
            "mean_confidence": self.metrics.get("mean_confidence"),
        }


class ModelRegistry:
    """Read/write wrapper around registry.json."""

    def __init__(self, path: str | Path = REGISTRY_PATH,
                 model_dir: str | Path = MODEL_DIR):
        self.path = Path(path)
        self.model_dir = Path(model_dir)
        self.active: str | None = None
        self.versions: dict[str, ModelVersion] = {}
        self.load()

    # -- persistence -------------------------------------------------------

    def load(self) -> None:
        if not self.path.exists():
            self.active, self.versions = None, {}
            return

        data = json.loads(self.path.read_text())
        self.active = data.get("active")
        self.versions = {
            entry["version"]: ModelVersion(**entry)
            for entry in data.get("versions", [])
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "active": self.active,
            "updated_at": _utc_now(),
            "versions": [asdict(v) for v in self.versions.values()],
        }
        # Write-then-rename so a crash mid-write cannot leave a truncated
        # registry behind, which would make the service unbootable.
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.path)

    # -- queries -----------------------------------------------------------

    def next_version(self) -> str:
        numbers = [
            int(v[1:]) for v in self.versions
            if v.startswith("v") and v[1:].isdigit()
        ]
        return f"v{max(numbers, default=0) + 1}"

    def active_version(self) -> ModelVersion | None:
        return self.versions.get(self.active) if self.active else None

    def head_path(self, version: str | None = None) -> Path | None:
        """Resolve a version to its head file on disk."""
        entry = self.versions.get(version or self.active or "")
        if entry is None:
            return None
        path = self.model_dir / entry.head_file
        return path if path.exists() else None

    def history(self) -> list[dict[str, Any]]:
        """Version summaries, newest first."""
        return [
            {**v.summary(), "active": v.version == self.active}
            for v in sorted(self.versions.values(),
                            key=lambda v: v.created_at, reverse=True)
        ]

    # -- mutations ---------------------------------------------------------

    def register(self, version: str, head_file: str,
                 metrics: dict[str, Any] | None = None,
                 activate: bool = True, **kwargs) -> ModelVersion:
        """Record a newly trained head, optionally promoting it to active."""
        entry = ModelVersion(
            version=version,
            head_file=head_file,
            metrics=metrics or {},
            **kwargs,
        )
        self.versions[version] = entry
        if activate or self.active is None:
            self.active = version
        self.save()
        return entry

    def activate(self, version: str) -> ModelVersion:
        """Promote an existing version -- the rollback path."""
        if version not in self.versions:
            raise KeyError(f"unknown model version: {version}")
        if self.head_path(version) is None:
            raise FileNotFoundError(
                f"head file for {version} is missing from {self.model_dir}; "
                "cannot activate a version whose artifact is gone"
            )
        self.active = version
        self.save()
        return self.versions[version]

    def should_promote(self, candidate_metrics: dict[str, Any],
                       max_regression: float) -> tuple[bool, str]:
        """Decide whether a retrained head is safe to promote.

        Retraining on user-uploaded data is the one place where an automated
        pipeline can quietly make the model worse, so promotion is gated on
        macro-F1 rather than being automatic. Macro-F1 is the gate because it
        is the metric that notices when rare disease classes have been
        sacrificed to fit whatever the user just uploaded.

        **What the candidate is compared against matters.** v1 is fitted on the
        full 38k training split; a retrained head is fitted on the replay
        buffer plus new uploads. The buffer is a stratified subset, so it costs
        a measured ~0.019 macro-F1 on its own -- a constant offset from having
        less data, not a regression caused by the new images. Comparing against
        v1 would charge every retrain for that offset and reject honest ones,
        which is indistinguishable from the gate working.

        So when the active version records `buffer_baseline_f1` -- the score of
        a head fitted on the buffer alone, written at export time -- that is
        the reference. The tolerance then measures the thing it claims to: did
        the new data make this worse?
        """
        current = self.active_version()
        if current is None or not current.metrics:
            return True, "no active version to compare against"

        after = candidate_metrics.get("f1_macro")
        if after is None:
            return True, "macro-F1 unavailable; promoting by default"

        baseline = current.metrics.get("buffer_baseline_f1")
        if baseline is not None:
            reference = "buffer baseline"
        else:
            baseline = current.metrics.get("f1_macro")
            reference = "active version"
        if baseline is None:
            return True, "macro-F1 unavailable; promoting by default"

        delta = after - baseline
        if delta < -max_regression:
            return False, (
                f"macro-F1 regressed {abs(delta):.4f} vs the {reference} "
                f"({baseline:.4f} -> {after:.4f}), exceeding the "
                f"{max_regression:.4f} tolerance"
            )
        direction = "improved on" if delta >= 0 else "within tolerance of"
        return True, (f"macro-F1 {direction} the {reference} "
                      f"({baseline:.4f} -> {after:.4f})")

    def prune(self, keep: int = 10) -> list[str]:
        """Delete the oldest heads, never touching the active one."""
        ordered = sorted(self.versions.values(), key=lambda v: v.created_at, reverse=True)
        removed = []
        for entry in ordered[keep:]:
            if entry.version == self.active:
                continue
            path = self.model_dir / entry.head_file
            if path.exists():
                path.unlink()
            del self.versions[entry.version]
            removed.append(entry.version)
        if removed:
            self.save()
        return removed


def bootstrap_registry(head_source: str | Path, metrics: dict[str, Any],
                       n_train_samples: int = 0,
                       head_kind: str = "logreg") -> ModelVersion:
    """Seed the registry with v1 from the notebook's trained head.

    Called once at the end of the notebook's export step; the API expects a
    registry to already exist when it starts.
    """
    registry = ModelRegistry()
    version = registry.next_version()
    head_file = f"head_{version}.pkl"

    destination = registry.model_dir / head_file
    destination.parent.mkdir(parents=True, exist_ok=True)
    if Path(head_source).resolve() != destination.resolve():
        shutil.copy2(head_source, destination)

    entry = registry.register(
        version=version,
        head_file=head_file,
        metrics=metrics,
        n_train_samples=n_train_samples,
        head_kind=head_kind,
        source="notebook",
        notes="Initial head trained on frozen MobileNetV2 embeddings.",
    )
    print(f"Registered {version} as active ({head_file})")
    return entry
