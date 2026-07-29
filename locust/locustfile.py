"""Locust load test for the prediction API.

    locust -f locust/locustfile.py --host http://localhost:8080

Task weights model realistic traffic rather than hammering one endpoint: a
dashboard polls health and metrics continuously while users classify images,
so prediction dominates but is not the only load.

Two details that make the numbers trustworthy:

* **Real leaf images are posted, not synthetic noise.** Payload size and JPEG
  decode cost are part of the latency being measured, and random bytes would
  understate both.
* **Every response's serving replica is recorded.** With several containers
  behind nginx, that is the evidence load was actually distributed rather than
  absorbed by one replica -- without it, a "4 container" run could silently be
  a 1-container run. The distribution is printed when the test stops.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from locust import HttpUser, between, events, task

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIRS = [PROJECT_ROOT / "data" / "test", PROJECT_ROOT / "data" / "train"]

_IMAGES: list[bytes] = []
_replica_hits: Counter[str] = Counter()


def _load_sample_images(limit: int = 40) -> list[bytes]:
    """Read a spread of real leaf images into memory once.

    Held in memory so disk I/O never shows up inside a measured request.
    """
    payloads: list[bytes] = []
    for root in SAMPLE_DIRS:
        if not root.is_dir():
            continue
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            # Prefer the committed samples so the test also runs on a fresh
            # clone that has not downloaded the full dataset.
            candidates = sorted(class_dir.glob("SAMPLE_*")) or sorted(
                p for p in class_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if candidates:
                payloads.append(candidates[0].read_bytes())
            if len(payloads) >= limit:
                return payloads
    return payloads


@events.test_start.add_listener
def on_start(environment, **_):
    global _IMAGES
    _IMAGES = _load_sample_images()
    _replica_hits.clear()
    if not _IMAGES:
        raise RuntimeError(
            "no sample images found under data/. Run: python src/acquire_data.py")
    print(f"[locust] loaded {len(_IMAGES)} sample images "
          f"(mean {sum(len(i) for i in _IMAGES) // len(_IMAGES) / 1024:.0f} KB)")


@events.test_stop.add_listener
def on_stop(environment, **_):
    """Report which replicas served the traffic."""
    total = sum(_replica_hits.values())
    print("\n" + "=" * 62)
    print(f"  replicas that served traffic: {len(_replica_hits)}")
    for name, count in _replica_hits.most_common():
        print(f"    {name:<24} {count:>7,} requests  ({count / max(total, 1):.1%})")
    if len(_replica_hits) == 1:
        print("\n  NOTE: only one replica responded. If this run was meant to")
        print("  exercise several containers, the balancer is not distributing.")
    print("=" * 62)


class PredictionUser(HttpUser):
    """A user that classifies leaf images and watches the dashboard."""

    # Think time. Zero wait would measure how fast Locust can spin, not how
    # the service behaves under realistic concurrency.
    wait_time = between(0.1, 0.6)

    @task(20)
    def predict(self):
        payload = random.choice(_IMAGES)
        with self.client.post(
            "/predict",
            files={"file": ("leaf.png", payload, "image/png")},
            name="POST /predict",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                body = response.json()
                if "prediction" not in body:
                    response.failure("response missing a prediction")
                else:
                    response.success()
            elif response.status_code == 503:
                # No model loaded is a deployment problem, not a load problem;
                # marking it distinguishes the two in the report.
                response.failure("503 model not ready")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(3)
    def health(self):
        with self.client.get("/health", name="GET /health",
                             catch_response=True) as response:
            if response.status_code == 200:
                host = response.json().get("hostname")
                if host:
                    _replica_hits[host] += 1
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)
    def metrics(self):
        self.client.get("/metrics", name="GET /metrics")

    @task(1)
    def batch_predict(self):
        """A small batch, to show amortised per-image cost under load."""
        chosen = random.sample(_IMAGES, min(4, len(_IMAGES)))
        self.client.post(
            "/predict/batch",
            files=[("files", (f"leaf{i}.png", payload, "image/png"))
                   for i, payload in enumerate(chosen)],
            name="POST /predict/batch",
        )
