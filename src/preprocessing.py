"""Image preprocessing, EDA feature extraction and tf.data pipelines.

This module is imported by two very different runtimes, so it is split in two:

* **Core** (top of file) -- pure NumPy/Pillow/OpenCV. This is what the deployed
  API imports. It must never require TensorFlow, because the serving container
  deliberately does not install it (see the architecture note in README.md).
* **Training** (bottom of file) -- tf.data pipelines and augmentation. These
  import TensorFlow *lazily*, inside the functions that need it, so that a
  container without TF can still import this module.

The EDA feature extractors in the middle are the quantitative basis for the
notebook's feature-interpretation section.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from src.config import (
    CLASS_SEPARATOR,
    IMAGE_SIZE,
    RANDOM_SEED,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# ==========================================================================
# Core image path -- NumPy/Pillow only, safe to import without TensorFlow
# ==========================================================================

def load_image(source: str | Path | bytes, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Load an image from a path or raw bytes as a resized uint8 RGB array.

    Accepting bytes lets the API feed an uploaded file straight in without
    touching the filesystem.
    """
    if isinstance(source, bytes):
        img = Image.open(io.BytesIO(source))
    else:
        img = Image.open(source)

    # Uploads may be palettised, greyscale or RGBA; the backbone wants 3 channels.
    img = img.convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def preprocess_for_backbone(images: np.ndarray) -> np.ndarray:
    """Scale uint8 RGB to the [-1, 1] range MobileNetV2 was trained on.

    This reimplements `keras.applications.mobilenet_v2.preprocess_input` in
    NumPy so the serving container does not need TensorFlow. The notebook
    asserts the two agree exactly before exporting -- see the parity check.
    """
    x = np.asarray(images, dtype=np.float32)
    if x.ndim == 3:
        x = x[np.newaxis, ...]
    return (x / 127.5) - 1.0


def prepare_batch(sources: Sequence[str | Path | bytes],
                  size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Load and preprocess several images into one NHWC float32 batch."""
    if not sources:
        return np.empty((0, *size, 3), dtype=np.float32)
    return preprocess_for_backbone(
        np.stack([load_image(s, size) for s in sources])
    )


def list_images(directory: str | Path, include_samples: bool = True) -> list[Path]:
    """List image files directly inside a directory."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix in IMAGE_SUFFIXES
        and (include_samples or not p.name.startswith("SAMPLE_"))
    )


def split_class_name(class_name: str) -> tuple[str, str]:
    """Split "Tomato___Late_blight" into ("Tomato", "Late blight").

    Crop and condition are separate analytical axes -- the notebook shows that
    crop identity dominates the embedding geometry while condition is the
    harder, finer-grained signal -- so nearly every plot needs this split.
    """
    if CLASS_SEPARATOR in class_name:
        crop, condition = class_name.split(CLASS_SEPARATOR, 1)
    else:
        crop, condition = class_name, "unknown"
    crop = crop.replace("_", " ").replace(",", "").strip()
    condition = condition.replace("_", " ").strip()
    return crop, condition


def is_healthy(class_name: str) -> bool:
    return "healthy" in class_name.lower()


# ==========================================================================
# EDA feature extraction
#
# These hand-crafted features answer a question that should be settled before
# any CNN is trained: is there measurable signal separating healthy from
# diseased leaves? Each maps to one interpretation in the notebook.
# ==========================================================================

def colour_statistics(img: np.ndarray) -> dict[str, float]:
    """Per-channel colour statistics and vegetation indices.

    Chlorosis and necrosis -- yellowing and browning -- are the visible
    signature of most foliar disease, so a healthy leaf should sit measurably
    higher on the green axis than a diseased one.
    """
    x = img.astype(np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    total = r + g + b + 1e-6

    # Excess Green: the standard vegetation index from precision agriculture.
    exg = 2.0 * g - r - b

    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    return {
        "mean_r": float(r.mean()),
        "mean_g": float(g.mean()),
        "mean_b": float(b.mean()),
        "std_r": float(r.std()),
        "std_g": float(g.std()),
        "std_b": float(b.std()),
        "green_ratio": float((g / total).mean()),
        "excess_green": float(exg.mean()),
        # Brown/yellow lesions push red above green; healthy tissue does not.
        "redness_index": float(((r - g) / total).mean()),
        "mean_hue": float(hsv[..., 0].mean()),
        "mean_saturation": float(hsv[..., 1].mean()),
        "mean_value": float(hsv[..., 2].mean()),
        "brightness": float(x.mean()),
    }


def texture_features(img: np.ndarray) -> dict[str, float]:
    """Edge and texture statistics used as a lesion-density proxy.

    Necrotic spots, rust pustules and blight margins introduce high-frequency
    intensity transitions that a uniformly green healthy leaf lacks.
    """
    import cv2
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edges = cv2.Canny(gray, 100, 200)

    return {
        # Variance of the Laplacian: the classic focus/detail measure.
        "laplacian_var": float(laplacian.var()),
        # Fraction of pixels sitting on a detected edge.
        "edge_density": float((edges > 0).mean()),
        "gray_std": float(gray.std()),
        "gray_entropy": _entropy(gray),
    }


def background_uniformity(img: np.ndarray, patch: int = 24) -> dict[str, float]:
    """Measure how uniform the image corners are.

    This is the project's most important EDA finding, and it is a warning
    rather than a feature. PlantVillage leaves are photographed on a uniform
    laboratory background. A model that reaches 99% here is partly reading
    acquisition conditions, and will degrade on a loan officer's field photo.
    That expected degradation is precisely what the retraining pipeline and the
    confidence-drift trigger exist to absorb.
    """
    h, w = img.shape[:2]
    corners = np.concatenate([
        img[:patch, :patch].reshape(-1, 3),
        img[:patch, -patch:].reshape(-1, 3),
        img[-patch:, :patch].reshape(-1, 3),
        img[-patch:, -patch:].reshape(-1, 3),
    ])
    corner_mean = corners.mean(axis=0)

    # Low spread across all four corners => a controlled, uniform backdrop.
    corner_std = float(corners.std(axis=0).mean())

    # How much of the frame resembles that background colour.
    distance = np.linalg.norm(img.astype(np.float32) - corner_mean, axis=-1)
    background_fraction = float((distance < 40).mean())

    return {
        "corner_std": corner_std,
        "background_fraction": background_fraction,
        "corner_brightness": float(corner_mean.mean()),
    }


def _entropy(gray: np.ndarray) -> float:
    """Shannon entropy of the 8-bit intensity histogram."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def extract_features(source: str | Path | bytes) -> dict[str, float]:
    """All EDA features for a single image."""
    img = load_image(source)
    return {**colour_statistics(img), **texture_features(img),
            **background_uniformity(img)}


def build_feature_table(split_dir: str | Path,
                        samples_per_class: int = 80,
                        seed: int = RANDOM_SEED,
                        progress: bool = True):
    """Sample images per class and tabulate their EDA features.

    Returns a pandas DataFrame with one row per image. Sampling keeps this
    tractable -- the full 54k images would take far longer for no extra insight.
    """
    import pandas as pd

    split_dir = Path(split_dir)
    rng = np.random.default_rng(seed)
    class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())

    rows = []
    for i, class_dir in enumerate(class_dirs, 1):
        images = list_images(class_dir)
        if not images:
            continue
        if len(images) > samples_per_class:
            idx = rng.choice(len(images), samples_per_class, replace=False)
            images = [images[j] for j in idx]

        crop, condition = split_class_name(class_dir.name)
        for path in images:
            try:
                features = extract_features(path)
            except Exception as exc:  # a corrupt file must not abort the whole scan
                print(f"    skipped {path.name}: {exc}")
                continue
            rows.append({
                "class": class_dir.name,
                "crop": crop,
                "condition": condition,
                "healthy": is_healthy(class_dir.name),
                "path": str(path),
                **features,
            })

        if progress:
            print(f"  [{i:2d}/{len(class_dirs)}] {class_dir.name:<45} {len(images):>4} sampled")

    return pd.DataFrame(rows)


def class_distribution(split_dir: str | Path):
    """Count images per class -- the basis of the imbalance interpretation."""
    import pandas as pd

    split_dir = Path(split_dir)
    rows = []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        crop, condition = split_class_name(class_dir.name)
        rows.append({
            "class": class_dir.name,
            "crop": crop,
            "condition": condition,
            "healthy": is_healthy(class_dir.name),
            "count": len(list_images(class_dir)),
        })
    return pd.DataFrame(rows).sort_values("count", ascending=False)


# ==========================================================================
# Training pipelines -- TensorFlow imported lazily
# ==========================================================================

def build_dataset(split_dir: str | Path,
                  class_names: Sequence[str],
                  batch_size: int | None = None,
                  shuffle: bool = False,
                  augment: bool = False,
                  cache: bool = False):
    """Build a tf.data pipeline yielding ([-1, 1] float32 images, int labels).

    Augmentation is applied *after* batching and only when requested, so the
    validation and test pipelines stay deterministic.
    """
    import tensorflow as tf
    from src.config import BATCH_SIZE

    batch_size = batch_size or BATCH_SIZE

    ds = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="int",
        class_names=list(class_names),   # pinned to the canonical order
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=RANDOM_SEED,
    )

    # image_dataset_from_directory yields uint8-valued floats in [0, 255].
    ds = ds.map(lambda x, y: ((x / 127.5) - 1.0, y),
                num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        ds = ds.cache()
    if augment:
        layers = augmentation_layers()
        ds = ds.map(lambda x, y: (layers(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def augmentation_layers():
    """Augmentations chosen for what actually varies in field photography.

    A leaf has no canonical orientation, and phone photos vary in exposure and
    distance -- so flips, rotation, zoom and brightness/contrast jitter are all
    label-preserving here. Colour *hue* is deliberately left alone: hue shift is
    the disease signal itself, so perturbing it would destroy the label.
    """
    import tensorflow as tf

    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=RANDOM_SEED),
        tf.keras.layers.RandomRotation(0.2, fill_mode="reflect", seed=RANDOM_SEED),
        tf.keras.layers.RandomZoom(0.15, fill_mode="reflect", seed=RANDOM_SEED),
        tf.keras.layers.RandomContrast(0.15, seed=RANDOM_SEED),
        tf.keras.layers.RandomBrightness(0.15, value_range=(-1.0, 1.0), seed=RANDOM_SEED),
    ], name="augmentation")


def compute_class_weights(split_dir: str | Path,
                          class_names: Sequence[str]) -> dict[int, float]:
    """Inverse-frequency class weights.

    Class sizes span roughly 150 to 5,500 images. Without weighting, the loss is
    dominated by the large tomato classes and rare classes are effectively
    ignored -- which is exactly the failure macro-F1 exposes.
    """
    split_dir = Path(split_dir)
    counts = np.array([len(list_images(split_dir / name)) for name in class_names],
                      dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    return {i: float(w) for i, w in enumerate(weights)}


def iter_image_paths(split_dir: str | Path,
                     class_names: Sequence[str]) -> Iterable[tuple[Path, int]]:
    """Yield (path, label index) pairs in canonical class order."""
    split_dir = Path(split_dir)
    for label, name in enumerate(class_names):
        for path in list_images(split_dir / name):
            yield path, label
