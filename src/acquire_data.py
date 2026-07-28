"""Data acquisition: fetch PlantVillage and build a stratified train/val/test split.

The upstream repository ships three copies of every image -- `color`, `grayscale`
and `segmented`. We only want `raw/color`, so a *sparse* checkout is used: a
blobless partial clone downloads the tree first and materialises only the paths
we ask for, which avoids pulling roughly three times more data than needed.

Run directly:

    python src/acquire_data.py                # download + split
    python src/acquire_data.py --samples-only # rebuild the committed git samples
    python src/acquire_data.py --skip-download

The split is stratified *per class* and seeded, so re-running is idempotent and
the same image always lands in the same split.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    CLASS_NAMES_PATH,
    CLASS_SEPARATOR,
    DATASET_REPO,
    DATASET_SUBDIR,
    DATA_DIR,
    N_CLASSES,
    RANDOM_SEED,
    RAW_DIR,
    SAMPLE_PREFIX,
    SAMPLES_PER_CLASS,
    TEST_DIR,
    TEST_FRACTION,
    TRAIN_DIR,
    TRAIN_FRACTION,
    VAL_FRACTION,
    ensure_dirs,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
VAL_DIR = DATA_DIR / "val"
CLONE_DIR = RAW_DIR / "PlantVillage-Dataset"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a command, streaming output, and fail loudly."""
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def download() -> Path:
    """Sparse-clone the colour variant of PlantVillage. Returns the image root."""
    source_root = CLONE_DIR / DATASET_SUBDIR
    if source_root.exists() and any(source_root.iterdir()):
        print(f"Dataset already present at {source_root} -- skipping download.")
        return source_root

    CLONE_DIR.parent.mkdir(parents=True, exist_ok=True)
    if CLONE_DIR.exists():
        # A previous run died partway through; a half-clone is worse than none.
        shutil.rmtree(CLONE_DIR)

    print(f"Sparse-cloning {DATASET_SUBDIR} from {DATASET_REPO} (~1.5 GB)...")
    _run([
        "git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1",
        DATASET_REPO, str(CLONE_DIR),
    ])
    _run(["git", "sparse-checkout", "init", "--cone"], cwd=CLONE_DIR)
    _run(["git", "sparse-checkout", "set", DATASET_SUBDIR], cwd=CLONE_DIR)
    _run(["git", "checkout"], cwd=CLONE_DIR)

    if not source_root.exists():
        raise RuntimeError(f"Expected {source_root} after checkout, but it is missing.")
    return source_root


def index_classes(source_root: Path) -> dict[str, list[Path]]:
    """Map each class directory name to its sorted list of image paths.

    Sorting matters: it makes the seeded shuffle below fully reproducible
    regardless of filesystem iteration order.
    """
    classes: dict[str, list[Path]] = {}
    for class_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        images = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix in IMAGE_SUFFIXES
        )
        if images:
            classes[class_dir.name] = images

    if len(classes) != N_CLASSES:
        print(f"  WARNING: found {len(classes)} classes, expected {N_CLASSES}")
    return classes


def split_dataset(classes: dict[str, list[Path]], move: bool = False) -> dict:
    """Copy images into data/{train,val,test}/<class>/ stratified per class.

    Stratifying per class (rather than shuffling globally) guarantees every
    class appears in every split, which matters here because class sizes range
    from ~150 to ~5,500 images.
    """
    rng = random.Random(RANDOM_SEED)
    transfer = shutil.move if move else shutil.copy2
    counts: dict[str, dict[str, int]] = defaultdict(dict)

    for split_dir in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        split_dir.mkdir(parents=True, exist_ok=True)

    for i, (class_name, images) in enumerate(sorted(classes.items()), 1):
        shuffled = list(images)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * TRAIN_FRACTION)
        n_val = int(n * VAL_FRACTION)
        splits = {
            "train": (TRAIN_DIR, shuffled[:n_train]),
            "val": (VAL_DIR, shuffled[n_train:n_train + n_val]),
            # Remainder goes to test so no image is dropped by rounding.
            "test": (TEST_DIR, shuffled[n_train + n_val:]),
        }

        for split_name, (split_root, paths) in splits.items():
            dest_dir = split_root / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in paths:
                dest = dest_dir / src.name
                if not dest.exists():
                    transfer(str(src), str(dest))
            counts[class_name][split_name] = len(paths)

        print(f"  [{i:2d}/{len(classes)}] {class_name:<45} "
              f"n={n:>5}  train={n_train:>5} val={n_val:>4} test={n - n_train - n_val:>4}")

    return dict(counts)


def write_class_names(classes: dict[str, list[Path]]) -> list[str]:
    """Persist the canonical, sorted class order.

    Every downstream artifact -- the classifier head, the registry, the API
    response -- indexes into this list, so it must never be reordered.
    """
    class_names = sorted(classes.keys())
    CLASS_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLASS_NAMES_PATH.write_text(json.dumps(class_names, indent=2))
    print(f"Wrote {len(class_names)} class names to {CLASS_NAMES_PATH}")
    return class_names


def extract_git_samples() -> int:
    """Copy a few images per class back under a SAMPLE_ prefix.

    These are the only images committed to git (see .gitignore), so a fresh
    clone can run the API and UI end to end without the 1.5 GB download.
    """
    total = 0
    for split_root in (TRAIN_DIR, TEST_DIR):
        for class_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
            existing = [p for p in class_dir.iterdir()
                        if p.name.startswith(SAMPLE_PREFIX)]
            if len(existing) >= SAMPLES_PER_CLASS:
                continue
            candidates = sorted(
                p for p in class_dir.iterdir()
                if p.is_file() and not p.name.startswith(SAMPLE_PREFIX)
                and p.suffix in IMAGE_SUFFIXES
            )
            for src in candidates[:SAMPLES_PER_CLASS - len(existing)]:
                # Normalise the extension so .gitignore's negation rules match.
                dest = class_dir / f"{SAMPLE_PREFIX}{src.stem}.png"
                if not dest.exists():
                    shutil.copy2(src, dest)
                    total += 1
    print(f"Extracted {total} sample images for git")
    return total


def summarise(counts: dict) -> None:
    totals = defaultdict(int)
    for per_split in counts.values():
        for split_name, n in per_split.items():
            totals[split_name] += n
    sizes = [sum(v.values()) for v in counts.values()]
    print("\n" + "=" * 68)
    print(f"  classes           : {len(counts)}")
    print(f"  images            : {sum(totals.values()):,}")
    print(f"  train/val/test    : {totals['train']:,} / {totals['val']:,} / {totals['test']:,}")
    print(f"  smallest class    : {min(sizes):,} images")
    print(f"  largest class     : {max(sizes):,} images")
    print(f"  imbalance ratio   : {max(sizes) / min(sizes):.1f}x")
    print("=" * 68)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true",
                        help="use an already-downloaded clone")
    parser.add_argument("--samples-only", action="store_true",
                        help="only rebuild the git-tracked sample images")
    parser.add_argument("--move", action="store_true",
                        help="move instead of copy, halving peak disk usage")
    args = parser.parse_args()

    ensure_dirs()

    if args.samples_only:
        extract_git_samples()
        return

    source_root = (CLONE_DIR / DATASET_SUBDIR) if args.skip_download else download()
    print(f"\nIndexing classes under {source_root}...")
    classes = index_classes(source_root)

    write_class_names(classes)
    print(f"\nSplitting {sum(len(v) for v in classes.values()):,} images "
          f"({int(TRAIN_FRACTION * 100)}/{int(VAL_FRACTION * 100)}/"
          f"{int(TEST_FRACTION * 100)}, seed={RANDOM_SEED})...")
    counts = split_dataset(classes, move=args.move)

    (DATA_DIR / "split_counts.json").write_text(json.dumps(counts, indent=2))
    extract_git_samples()
    summarise(counts)


if __name__ == "__main__":
    main()
