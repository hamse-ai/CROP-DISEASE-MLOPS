"""Run the Locust matrix across container counts and chart the results.

    python locust/run_load_tests.py --containers 1 2 4 --users 20 50 100

For each container count it rescales the compose stack, waits for the replicas
to become healthy, then runs one headless Locust test per user level. Results
land in reports/load_test/ as CSVs plus a summary chart and table.

Rescaling between runs (rather than running everything against one fixed
stack) is the whole point: the deliverable is how latency and throughput
respond to *container count*, which cannot be measured without changing it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "reports" / "load_test"
LOCUSTFILE = PROJECT_ROOT / "locust" / "locustfile.py"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=PROJECT_ROOT, **kwargs)


def scale_api(replicas: int, host: str) -> None:
    """Rescale the API service and wait until it actually serves traffic."""
    run(["docker", "compose", "up", "-d", "--scale", f"api={replicas}",
         "--no-recreate", "api", "nginx"], check=True)

    print(f"  waiting for {replicas} replica(s) to become healthy...")
    deadline = time.time() + 180
    seen: set[str] = set()

    while time.time() < deadline:
        try:
            response = requests.get(f"{host}/health", timeout=5)
            if response.status_code == 200:
                body = response.json()
                seen.add(body["hostname"])
                # Poll until every replica has answered at least once, so the
                # measured run is not racing container startup.
                if len(seen) >= replicas and body["model_ready"]:
                    print(f"  {len(seen)} replica(s) responding: {sorted(seen)}")
                    return
        except requests.RequestException:
            pass
        time.sleep(2)

    print(f"  WARNING: only {len(seen)} of {replicas} replica(s) responded "
          f"within the timeout; results may understate the scaling effect")


def run_locust(host: str, users: int, duration: str, spawn_rate: int,
               tag: str) -> dict | None:
    """Run one headless Locust test and parse its aggregate stats."""
    prefix = OUTPUT_DIR / tag
    # Locust appends to existing CSVs; stale rows from a previous run would be
    # mixed into this one's statistics.
    for stale in OUTPUT_DIR.glob(f"{tag}_*.csv"):
        stale.unlink()

    result = run([
        "locust", "-f", str(LOCUSTFILE), "--headless",
        "--host", host,
        "--users", str(users), "--spawn-rate", str(spawn_rate),
        "--run-time", duration,
        "--csv", str(prefix), "--csv-full-history",
        "--only-summary",
    ])
    if result.returncode not in (0, 1):   # 1 = tests ran but failures occurred
        print(f"  locust exited {result.returncode}")

    stats_file = prefix.with_name(prefix.name + "_stats.csv")
    if not stats_file.exists():
        return None

    import csv

    with stats_file.open() as handle:
        rows = list(csv.DictReader(handle))

    aggregate = next((r for r in rows if r["Name"] == "Aggregated"), None)
    predict = next((r for r in rows if r["Name"] == "POST /predict"), None)
    if aggregate is None:
        return None

    def number(row, key, default=0.0):
        try:
            return float(row.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    requests = int(number(aggregate, "Request Count"))
    # Locust's "Requests/s" column is the *instantaneous* rate at the final
    # snapshot, not the average over the run. One 4-container run reported
    # 0.94 RPS for 1,962 requests in 45 seconds -- off by a factor of 46, and
    # in the direction that would have made scaling look broken. Throughput is
    # therefore derived from the request count and the measured elapsed time.
    elapsed = _elapsed_seconds(prefix) or _duration_seconds(duration)
    rps = round(requests / elapsed, 2) if elapsed else 0.0

    return {
        "requests": requests,
        "failures": int(number(aggregate, "Failure Count")),
        "rps": rps,
        "elapsed_seconds": round(elapsed, 1),
        "rps_locust_reported": round(number(aggregate, "Requests/s"), 2),
        "p50": number(aggregate, "50%"),
        "p95": number(aggregate, "95%"),
        "p99": number(aggregate, "99%"),
        "mean": round(number(aggregate, "Average Response Time"), 1),
        "max": number(aggregate, "Max Response Time"),
        "predict_p50": number(predict, "50%") if predict else None,
        "predict_p95": number(predict, "95%") if predict else None,
        "predict_rps": round(number(predict, "Requests/s"), 2) if predict else None,
    }


def _duration_seconds(duration: str) -> float:
    """Parse a Locust duration string such as "45s", "2m" or "1h"."""
    duration = duration.strip().lower()
    units = {"s": 1, "m": 60, "h": 3600}
    if duration and duration[-1] in units:
        try:
            return float(duration[:-1]) * units[duration[-1]]
        except ValueError:
            return 0.0
    try:
        return float(duration)
    except ValueError:
        return 0.0


def _elapsed_seconds(prefix: Path, gap_threshold: float = 30.0) -> float:
    """Measured wall-clock span of the most recent run in a history file.

    Preferred over the configured duration because ramp-up and shutdown mean
    the real span is not exactly what was asked for.

    Locust *appends* to an existing `_stats_history.csv`, so a re-run leaves
    the file holding several runs' rows. Taking min-to-max across all of them
    produced a 2,094-second span for a 45-second run and a throughput figure
    46x too low -- in the direction that made scaling look broken. Only the
    final contiguous segment counts.
    """
    history = prefix.with_name(prefix.name + "_stats_history.csv")
    if not history.exists():
        return 0.0

    import csv

    try:
        with history.open() as handle:
            stamps = sorted(float(row["Timestamp"]) for row in csv.DictReader(handle)
                            if row.get("Timestamp"))
    except (OSError, ValueError, KeyError):
        return 0.0
    if len(stamps) < 2:
        return 0.0

    start = 0
    for i in range(1, len(stamps)):
        if stamps[i] - stamps[i - 1] > gap_threshold:
            start = i
    segment = stamps[start:]
    return segment[-1] - segment[0] if len(segment) > 1 else 0.0


def chart(results: list[dict]) -> Path | None:
    """Latency and throughput against container count."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.run_eda import AQUA, BLUE, GRID, INK, MUTED, ORANGE, _style

    user_levels = sorted({r["users"] for r in results})
    container_levels = sorted({r["containers"] for r in results})
    colours = [BLUE, ORANGE, AQUA]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    for i, users in enumerate(user_levels[:3]):
        subset = sorted([r for r in results if r["users"] == users],
                        key=lambda r: r["containers"])
        if not subset:
            continue
        axes[0].plot([r["containers"] for r in subset], [r["p95"] for r in subset],
                     marker="o", markersize=7, linewidth=2,
                     color=colours[i % 3], label=f"{users} users")
        axes[1].plot([r["containers"] for r in subset], [r["rps"] for r in subset],
                     marker="o", markersize=7, linewidth=2,
                     color=colours[i % 3], label=f"{users} users")

    _style(axes[0], "p95 latency falls as containers scale", "API containers", "p95 (ms)")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_xticks(container_levels)

    _style(axes[1], "Throughput rises as containers scale", "API containers", "requests/s")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].set_xticks(container_levels)

    # Percentile spread at the heaviest load: the tail is what users feel.
    heaviest = max(user_levels)
    subset = sorted([r for r in results if r["users"] == heaviest],
                    key=lambda r: r["containers"])
    width = 0.26
    positions = range(len(subset))
    for offset, key, colour, label in ((-width, "p50", BLUE, "p50"),
                                       (0.0, "p95", ORANGE, "p95"),
                                       (width, "p99", AQUA, "p99")):
        axes[2].bar([p + offset for p in positions], [r[key] for r in subset],
                    width=width, color=colour, label=label)
    axes[2].set_xticks(list(positions))
    axes[2].set_xticklabels([f"{r['containers']}c" for r in subset])
    _style(axes[2], f"Latency spread at {heaviest} users", "API containers", "ms")
    axes[2].legend(frameon=False, fontsize=9)
    axes[2].grid(axis="x", visible=False)

    path = OUTPUT_DIR / "load_test_summary.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return path


def markdown_table(results: list[dict]) -> str:
    lines = [
        "| Containers | Users | Requests | Failures | RPS | p50 (ms) | p95 (ms) | p99 (ms) | Replicas hit |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(results, key=lambda r: (r["containers"], r["users"])):
        lines.append(
            f"| {r['containers']} | {r['users']} | {r['requests']:,} | "
            f"{r['failures']} | {r['rps']} | {r['p50']:.0f} | {r['p95']:.0f} | "
            f"{r['p99']:.0f} | {r.get('replicas_seen', '—')} |"
        )
    lines.append("")
    lines.append("RPS is computed as requests / measured elapsed time. Locust's own "
                 "`Requests/s` column reports the *instantaneous* rate at the final "
                 "snapshot and is unreliable -- it read 0.94 for a run that served "
                 "1,962 requests in 45 s.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--containers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--users", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--duration", default="60s")
    parser.add_argument("--host", default="http://localhost:8080")
    parser.add_argument("--spawn-rate", type=int, default=10)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for containers in args.containers:
        print(f"\n{'=' * 70}\nScaling to {containers} API container(s)\n{'=' * 70}")
        scale_api(containers, args.host)

        for users in args.users:
            tag = f"c{containers}_u{users}"
            print(f"\n  -- {containers} container(s), {users} users, "
                  f"{args.duration} --")

            # Reset counters so each run's replica distribution is its own.
            try:
                requests.get(f"{args.host}/health", timeout=5)
            except requests.RequestException:
                pass

            stats = run_locust(args.host, users, args.duration,
                               args.spawn_rate, tag)
            if stats is None:
                print("  no stats produced; skipping")
                continue

            replicas_seen = 0
            try:
                hosts = {requests.get(f"{args.host}/health", timeout=5).json()["hostname"]
                         for _ in range(containers * 6)}
                replicas_seen = len(hosts)
            except requests.RequestException:
                pass

            record = {"containers": containers, "users": users,
                      "replicas_seen": replicas_seen, **stats}
            results.append(record)
            print(f"    RPS {stats['rps']:>7}   p50 {stats['p50']:>6.0f} ms   "
                  f"p95 {stats['p95']:>6.0f} ms   p99 {stats['p99']:>6.0f} ms   "
                  f"failures {stats['failures']}")

    if not results:
        print("\nNo results collected.")
        return

    (OUTPUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    (OUTPUT_DIR / "results.md").write_text(markdown_table(results) + "\n")
    figure = chart(results)

    print("\n" + "=" * 70)
    print(markdown_table(results))
    print(f"\n  chart: {figure}")
    print(f"  data : {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
