"""Train the deployed classifier and export every serving artifact.

    python src/train_export.py --embed --train --export

Runs in three stages so an interrupted run can resume rather than restart:

1. **embed**  -- push train/val/test images through the frozen backbone once
   and cache the embeddings. This is the only expensive stage.
2. **train**  -- fit several classifier heads on those embeddings, choose by
   validation macro-F1, then score the winner on the held-out test split.
3. **export** -- write the backbone, the winning head, the replay buffer, the
   held-out evaluation embeddings and the registry entry.

Embeddings are extracted with the **exported ONNX backbone**, not the Keras
one. Both are numerically identical (verified below), but generating training
embeddings with the exact artifact that will serve them removes any
possibility of train/serve skew.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    BACKBONE_PATH,
    CLASS_NAMES_PATH,
    DATA_DIR,
    EVAL_EMBEDDINGS_PATH,
    EVAL_SAMPLES_PER_CLASS,
    IMAGE_SIZE,
    MODEL_DIR,
    RANDOM_SEED,
    REPLAY_BUFFER_PATH,
    REPLAY_SAMPLES_PER_CLASS,
    REPORTS_DIR,
    TEST_DIR,
    TRAIN_DIR,
)
from src.model import (  # noqa: E402
    build_replay_buffer,
    confusable_pairs,
    evaluate_predictions,
    save_head,
    summarise_metrics,
    train_head,
)
from src.preprocessing import list_images, load_image, preprocess_for_backbone  # noqa: E402

VAL_DIR = DATA_DIR / "val"
EMBED_DIR = MODEL_DIR / "embeddings"
HEAD_CANDIDATES = ["logreg", "sgd", "mlp"]


# ==========================================================================
# Stage 1 -- embed
# ==========================================================================

def export_backbone() -> Path:
    """Export MobileNetV2 to ONNX and assert it matches the Keras original."""
    import warnings

    warnings.filterwarnings("ignore")
    import tensorflow as tf
    import tf2onnx

    from src.model import build_backbone, verify_onnx_parity

    BACKBONE_PATH.parent.mkdir(parents=True, exist_ok=True)
    backbone = build_backbone()

    print(f"Exporting backbone to {BACKBONE_PATH}...")
    spec = (tf.TensorSpec((None, *IMAGE_SIZE, 3), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(backbone, input_signature=spec, opset=13,
                               output_path=str(BACKBONE_PATH))
    print(f"  {BACKBONE_PATH.stat().st_size / 1e6:.1f} MB")

    # Parity on real leaves, not noise: random input can mask a layout bug
    # that only shows on structured images.
    samples = []
    for class_dir in sorted(p for p in TEST_DIR.iterdir() if p.is_dir())[:12]:
        samples += [load_image(p) for p in list_images(class_dir)[:2]]
    batch = preprocess_for_backbone(np.stack(samples))

    print("Verifying ONNX/Keras parity...")
    verify_onnx_parity(backbone, batch, BACKBONE_PATH)
    return BACKBONE_PATH


def embed_split(split_dir: Path, class_names: list[str], name: str,
                batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Embed every image in a split, streaming batch by batch.

    Images are loaded and released per batch rather than all at once: the full
    training split would need tens of gigabytes as decoded float32 arrays.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 0          # let ORT use every core
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(BACKBONE_PATH), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    paths, labels = [], []
    for index, class_name in enumerate(class_names):
        class_paths = list_images(split_dir / class_name)
        paths += class_paths
        labels += [index] * len(class_paths)

    total = len(paths)
    print(f"  {name}: {total:,} images across {len(class_names)} classes")

    features = np.empty((total, 1280), dtype=np.float32)
    started = time.perf_counter()

    for start in range(0, total, batch_size):
        window = paths[start:start + batch_size]
        batch = preprocess_for_backbone(
            np.stack([load_image(p, IMAGE_SIZE) for p in window]))
        features[start:start + len(window)] = session.run(
            None, {input_name: batch})[0]

        done = start + len(window)
        if done % (batch_size * 20) == 0 or done == total:
            elapsed = time.perf_counter() - started
            rate = done / max(elapsed, 1e-9)
            remaining = (total - done) / max(rate, 1e-9)
            print(f"    {done:>6,}/{total:,}  {rate:5.1f} img/s  "
                  f"eta {remaining / 60:4.1f} min", flush=True)

    return features, np.asarray(labels, dtype=np.int64)


def stage_embed(class_names: list[str]) -> None:
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    for name, split_dir in (("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR)):
        target = EMBED_DIR / f"{name}.npz"
        if target.exists():
            print(f"  {name}: cached, skipping")
            continue
        X, y = embed_split(split_dir, class_names, name)
        np.savez(target, X=X, y=y)
        print(f"  saved {target} ({target.stat().st_size / 1e6:.0f} MB)\n")


def load_embeddings(name: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(EMBED_DIR / f"{name}.npz")
    return data["X"].astype(np.float32), data["y"].astype(np.int64)


# ==========================================================================
# Stage 2 -- train and select
# ==========================================================================

def stage_train(class_names: list[str]) -> tuple[object, dict, dict]:
    """Fit head candidates, select on validation, then score once on test."""
    X_train, y_train = load_embeddings("train")
    X_val, y_val = load_embeddings("val")
    X_test, y_test = load_embeddings("test")

    print(f"\n  train {X_train.shape}  val {X_val.shape}  test {X_test.shape}")

    results, heads = {}, {}
    for kind in HEAD_CANDIDATES:
        print(f"\n  fitting {kind}...")
        started = time.perf_counter()
        try:
            head = train_head(X_train, y_train, kind=kind, seed=RANDOM_SEED)
        except Exception as exc:
            print(f"    failed: {exc}")
            continue
        duration = time.perf_counter() - started

        metrics = evaluate_predictions(y_val, head.predict_proba(X_val), class_names)
        results[kind] = {"f1_macro": metrics["f1_macro"],
                         "accuracy": metrics["accuracy"],
                         "fit_seconds": round(duration, 1)}
        heads[kind] = head
        print(f"    {duration:6.1f}s  val macro-F1 {metrics['f1_macro']:.4f}  "
              f"accuracy {metrics['accuracy']:.4f}")

    if not heads:
        raise RuntimeError("every head candidate failed to fit")

    # Selection is on macro-F1, not accuracy: with a 36x imbalance, accuracy
    # would happily pick a head that has given up on the rare classes.
    best_kind = max(results, key=lambda k: results[k]["f1_macro"])
    best_head = heads[best_kind]
    print(f"\n  selected: {best_kind} (val macro-F1 "
          f"{results[best_kind]['f1_macro']:.4f})")

    # The test split is touched exactly once, after selection.
    print("\n  scoring the selected head on the held-out test split...")
    test_metrics = evaluate_predictions(
        y_test, best_head.predict_proba(X_test), class_names)
    print(summarise_metrics(test_metrics, "test set"))

    test_metrics["head_kind"] = best_kind
    test_metrics["selection"] = results
    return best_head, test_metrics, results


# ==========================================================================
# Stage 3 -- export
# ==========================================================================

def stage_export(head, metrics: dict, class_names: list[str]) -> None:
    from src.registry import bootstrap_registry

    X_train, y_train = load_embeddings("train")
    X_test, y_test = load_embeddings("test")

    print("\n  building replay buffer "
          f"({REPLAY_SAMPLES_PER_CLASS} per class)...")
    X_replay, y_replay = build_replay_buffer(X_train, y_train,
                                             per_class=REPLAY_SAMPLES_PER_CLASS)
    np.savez_compressed(REPLAY_BUFFER_PATH, X=X_replay, y=y_replay)
    print(f"    {X_replay.shape} -> {REPLAY_BUFFER_PATH.name} "
          f"({REPLAY_BUFFER_PATH.stat().st_size / 1e6:.1f} MB)")

    print(f"\n  building held-out evaluation set "
          f"({EVAL_SAMPLES_PER_CLASS} per class)...")
    X_eval, y_eval = build_replay_buffer(X_test, y_test,
                                         per_class=EVAL_SAMPLES_PER_CLASS)
    np.savez_compressed(EVAL_EMBEDDINGS_PATH, X=X_eval, y=y_eval)
    print(f"    {X_eval.shape} -> {EVAL_EMBEDDINGS_PATH.name} "
          f"({EVAL_EMBEDDINGS_PATH.stat().st_size / 1e6:.1f} MB)")

    staging = MODEL_DIR / "head_candidate.pkl"
    save_head(head, staging)

    bootstrap_registry(staging, metrics, n_train_samples=int(len(X_train)),
                       head_kind=metrics.get("head_kind", "logreg"))
    staging.unlink(missing_ok=True)

    # Full metrics, including the 38x38 confusion matrix, go to reports/ --
    # far too large to sit in every registry response.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    pairs = confusable_pairs(metrics, class_names, top_n=15)
    pairs.to_csv(REPORTS_DIR / "confusable_pairs.csv", index=False)

    print("\n  most-confused class pairs:")
    for row in pairs.head(8).itertuples():
        marker = "same crop" if row.same_crop else "ACROSS CROPS"
        print(f"    {row.count:>4}  {row.true[:34]:<34} -> "
              f"{row.predicted[:34]:<34} [{marker}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", action="store_true", help="export the ONNX backbone")
    parser.add_argument("--embed", action="store_true", help="cache embeddings")
    parser.add_argument("--train", action="store_true", help="fit and select a head")
    parser.add_argument("--export", action="store_true", help="write serving artifacts")
    parser.add_argument("--all", action="store_true", help="run every stage")
    args = parser.parse_args()

    if args.all:
        args.backbone = args.embed = args.train = args.export = True
    if not any((args.backbone, args.embed, args.train, args.export)):
        parser.error("choose at least one stage, or --all")

    class_names = json.loads(Path(CLASS_NAMES_PATH).read_text())
    print(f"{len(class_names)} classes\n")

    if args.backbone or (args.embed and not BACKBONE_PATH.exists()):
        print("=" * 70 + "\nSTAGE 1a: export backbone\n" + "=" * 70)
        export_backbone()

    if args.embed:
        print("\n" + "=" * 70 + "\nSTAGE 1b: embed splits\n" + "=" * 70)
        stage_embed(class_names)

    head = metrics = None
    if args.train:
        print("\n" + "=" * 70 + "\nSTAGE 2: train and select head\n" + "=" * 70)
        head, metrics, _ = stage_train(class_names)

    if args.export:
        if head is None:
            raise SystemExit("--export requires --train in the same run")
        print("\n" + "=" * 70 + "\nSTAGE 3: export artifacts\n" + "=" * 70)
        stage_export(head, metrics, class_names)

    print("\nDone.")


if __name__ == "__main__":
    main()
