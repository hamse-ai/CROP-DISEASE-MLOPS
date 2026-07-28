"""Model architectures, head training, evaluation metrics and artifact export.

Like `preprocessing`, this module is imported by runtimes with very different
dependencies, so TensorFlow is imported *lazily* inside the functions that need
it. The parts the deployed API relies on -- head training, the metric suite and
the replay-buffer builder -- are pure NumPy/scikit-learn and stay importable in
a container with no TensorFlow installed.

Four configurations are compared in the notebook:

1. a CNN trained from scratch (baseline: how far does the data alone get us?)
2. MobileNetV2 frozen + dense head (transfer learning)
3. MobileNetV2 with its final blocks unfrozen (fine-tuning)
4. classical heads fitted on frozen embeddings (the deployed configuration)

Configuration 4 is what ships, because it is the only one whose retraining step
is cheap enough to run inside a 512 MB container.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from src.config import (
    BACKBONE_PATH,
    EMBEDDING_DIM,
    IMAGE_SIZE,
    N_CLASSES,
    RANDOM_SEED,
    REPLAY_SAMPLES_PER_CLASS,
)


# ==========================================================================
# Deep architectures (TensorFlow -- training only)
# ==========================================================================

def build_backbone(trainable: bool = False):
    """MobileNetV2 feature extractor: 224x224x3 -> 1280-d embedding.

    `include_top=False` with `pooling="avg"` gives a global-average-pooled
    embedding rather than a spatial map. MobileNetV2 is chosen over a heavier
    backbone deliberately: it exports to a ~14 MB ONNX file, which is what lets
    the whole feature extractor live inside the free-tier memory budget.
    """
    import tensorflow as tf

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    backbone.trainable = trainable
    return backbone


def build_scratch_cnn(n_classes: int = N_CLASSES):
    """Baseline CNN trained from random initialisation.

    Included as an honest control: without it there is no way to say how much
    of the final accuracy comes from ImageNet transfer rather than the dataset.
    """
    import tensorflow as tf
    L = tf.keras.layers

    model = tf.keras.Sequential(name="scratch_cnn")
    model.add(L.Input(shape=(*IMAGE_SIZE, 3)))
    for filters in (32, 64, 128, 256):
        model.add(L.Conv2D(filters, 3, padding="same", use_bias=False))
        model.add(L.BatchNormalization())
        model.add(L.ReLU())
        model.add(L.Conv2D(filters, 3, padding="same", use_bias=False))
        model.add(L.BatchNormalization())
        model.add(L.ReLU())
        model.add(L.MaxPooling2D())
    model.add(L.GlobalAveragePooling2D())
    model.add(L.Dropout(0.3))
    model.add(L.Dense(256, activation="relu"))
    model.add(L.Dropout(0.3))
    model.add(L.Dense(n_classes, activation="softmax"))
    return model


def build_transfer_model(n_classes: int = N_CLASSES,
                         fine_tune_from: int | None = None,
                         dropout: float = 0.3):
    """MobileNetV2 plus a dense classifier head.

    `fine_tune_from=None` freezes the whole backbone (configuration 2).
    Passing a layer index unfreezes everything from there on (configuration 3);
    ~100 corresponds to MobileNetV2's last few inverted-residual blocks, which
    is where ImageNet features stop being generic enough for leaf lesions.
    """
    import tensorflow as tf
    L = tf.keras.layers

    backbone = build_backbone(trainable=fine_tune_from is not None)
    if fine_tune_from is not None:
        for layer in backbone.layers[:fine_tune_from]:
            layer.trainable = False
        # BatchNorm must stay frozen when fine-tuning on a small dataset:
        # updating its running statistics with small batches destabilises
        # the pretrained weights.
        for layer in backbone.layers[fine_tune_from:]:
            if isinstance(layer, L.BatchNormalization):
                layer.trainable = False

    inputs = L.Input(shape=(*IMAGE_SIZE, 3))
    x = backbone(inputs, training=False)
    x = L.Dropout(dropout)(x)
    outputs = L.Dense(n_classes, activation="softmax", name="predictions")(x)

    name = "mobilenetv2_finetuned" if fine_tune_from is not None else "mobilenetv2_frozen"
    return tf.keras.Model(inputs, outputs, name=name)


def compile_model(model, learning_rate: float = 1e-3):
    """Compile with the metrics that actually drive decisions here.

    Accuracy alone is misleading on a 38-class set with a ~36x imbalance ratio,
    so top-5 accuracy is tracked alongside it and macro-F1 is computed properly
    at evaluation time.
    """
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )
    return model


def default_callbacks(checkpoint_path: str | Path, patience: int = 5):
    """Early stopping, LR decay and checkpointing.

    Checkpointing every epoch mirrors the Colab-disconnect resilience used in
    the previous summative: if the runtime drops, rerunning the training cell
    resumes rather than restarting.
    """
    import tensorflow as tf

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=patience,
            restore_best_weights=True, mode="max", verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=max(2, patience // 2),
            min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor="val_accuracy",
            save_best_only=True, mode="max", verbose=0),
    ]


# ==========================================================================
# Embeddings
# ==========================================================================

def extract_embeddings(backbone, dataset, verbose: bool = True):
    """Run a tf.data pipeline through the backbone, returning (X, y).

    Embeddings are computed once and reused for every classical head, which is
    what makes the head comparison cheap enough to run on CPU.
    """
    features, labels = [], []
    for i, (batch_x, batch_y) in enumerate(dataset):
        features.append(backbone(batch_x, training=False).numpy())
        labels.append(batch_y.numpy())
        if verbose and i % 50 == 0:
            print(f"    batch {i}: {sum(len(f) for f in features):,} images embedded")

    X = np.concatenate(features).astype(np.float32)
    y = np.concatenate(labels).astype(np.int64)
    if verbose:
        print(f"  embedded {len(X):,} images -> {X.shape}")
    return X, y


def build_replay_buffer(X: np.ndarray, y: np.ndarray,
                        per_class: int = REPLAY_SAMPLES_PER_CLASS,
                        seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Take a stratified subsample of training embeddings.

    This is the anti-forgetting mechanism. Retraining on uploaded images alone
    would collapse the model onto whatever the user just uploaded; every
    retrain therefore fits on this buffer plus the new data.

    Stored as float16, which halves the file to roughly 29 MB with no
    measurable effect on head accuracy -- the difference matters when the whole
    container has 512 MB.
    """
    rng = np.random.default_rng(seed)
    keep = []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        if len(idx) > per_class:
            idx = rng.choice(idx, per_class, replace=False)
        keep.append(idx)

    keep = np.sort(np.concatenate(keep))
    return X[keep].astype(np.float16), y[keep]


# ==========================================================================
# Classifier head -- pure scikit-learn, importable without TensorFlow
# ==========================================================================

def train_head(X: np.ndarray, y: np.ndarray,
               kind: str = "logreg",
               class_weight: str | dict | None = "balanced",
               seed: int = RANDOM_SEED,
               **kwargs):
    """Fit a classifier on frozen embeddings.

    `logreg` is the default and the deployed choice: on 1280-d embeddings it
    fits in seconds, calibrates well enough for a confidence-based drift
    signal, and pickles to a few hundred KB.

    `sgd` is kept because it supports `partial_fit`, which is the escape hatch
    if the replay buffer ever outgrows memory.
    """
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.neural_network import MLPClassifier

    X = np.asarray(X, dtype=np.float32)

    if kind == "logreg":
        clf = LogisticRegression(
            max_iter=kwargs.pop("max_iter", 1500),
            C=kwargs.pop("C", 1.0),
            class_weight=class_weight,
            n_jobs=-1,
            random_state=seed,
            **kwargs,
        )
    elif kind == "sgd":
        clf = SGDClassifier(
            loss="log_loss",
            alpha=kwargs.pop("alpha", 1e-4),
            max_iter=kwargs.pop("max_iter", 50),
            class_weight=class_weight,
            random_state=seed,
            n_jobs=-1,
            **kwargs,
        )
    elif kind == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=kwargs.pop("hidden_layer_sizes", (256,)),
            max_iter=kwargs.pop("max_iter", 200),
            random_state=seed,
            **kwargs,
        )
    else:
        raise ValueError(f"unknown head type: {kind!r}")

    clf.fit(X, y)
    return clf


def save_head(clf, path: str | Path) -> Path:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, path, compress=3)
    return path


def load_head(path: str | Path):
    import joblib

    return joblib.load(path)


# ==========================================================================
# Evaluation -- pure scikit-learn
# ==========================================================================

def evaluate_predictions(y_true: np.ndarray,
                         y_proba: np.ndarray,
                         class_names: Sequence[str] | None = None) -> dict:
    """Compute the full metric suite from predicted probabilities.

    Accuracy is reported but is not the headline number. With classes ranging
    from ~150 to ~5,500 images, macro-F1 is the honest summary: it weights the
    rare disease classes -- the ones a loan officer most needs correct -- equally
    with the abundant tomato classes.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        top_k_accuracy_score,
    )

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = y_proba.argmax(axis=1)
    n_classes = y_proba.shape[1]
    labels = list(range(n_classes))

    metrics: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n_samples": int(len(y_true)),
        "n_classes": int(n_classes),
    }

    if n_classes > 5:
        metrics["top5_accuracy"] = float(
            top_k_accuracy_score(y_true, y_proba, k=5, labels=labels))

    # One-hot the targets for the threshold-free ranking metrics. These can fail
    # if a class is absent from the evaluation split, so they are guarded.
    y_onehot = np.zeros_like(y_proba)
    y_onehot[np.arange(len(y_true)), y_true] = 1.0
    try:
        metrics["roc_auc_macro"] = float(
            roc_auc_score(y_onehot, y_proba, average="macro", multi_class="ovr"))
        metrics["pr_auc_macro"] = float(
            average_precision_score(y_onehot, y_proba, average="macro"))
    except ValueError as exc:
        metrics["roc_auc_macro"] = None
        metrics["pr_auc_macro"] = None
        metrics["auc_warning"] = str(exc)

    # Mean confidence feeds the production drift trigger, so the notebook and
    # the live service compute it identically.
    metrics["mean_confidence"] = float(y_proba.max(axis=1).mean())

    target_names = list(class_names) if class_names is not None else None
    metrics["per_class"] = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names,
        output_dict=True, zero_division=0)
    metrics["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=labels).tolist()

    return metrics


def summarise_metrics(metrics: dict, title: str = "") -> str:
    """One-line-per-metric summary of the headline numbers."""
    keys = ["accuracy", "top5_accuracy", "f1_macro", "f1_weighted",
            "precision_macro", "recall_macro", "roc_auc_macro",
            "pr_auc_macro", "mean_confidence"]
    lines = [f"  {title}"] if title else []
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            lines.append(f"    {key:<20} {value:.4f}")
    return "\n".join(lines)


def confusable_pairs(metrics: dict, class_names: Sequence[str], top_n: int = 10):
    """Rank the most-confused class pairs from the confusion matrix.

    Drives the notebook's error analysis: the interesting failures are between
    visually similar conditions on the *same* crop (early vs late blight), not
    random noise, and that distinction is what tells you whether the model has
    learned anything botanical.
    """
    import pandas as pd

    cm = np.asarray(metrics["confusion_matrix"], dtype=float)
    support = cm.sum(axis=1, keepdims=True)
    rates = np.divide(cm, np.maximum(support, 1))

    rows = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                rows.append({
                    "true": class_names[i],
                    "predicted": class_names[j],
                    "count": int(cm[i, j]),
                    "rate": float(rates[i, j]),
                    "same_crop": class_names[i].split("___")[0] == class_names[j].split("___")[0],
                })

    return (pd.DataFrame(rows)
            .sort_values("count", ascending=False)
            .head(top_n)
            .reset_index(drop=True))


# ==========================================================================
# Export
# ==========================================================================

def export_backbone_onnx(backbone, path: str | Path = BACKBONE_PATH,
                         opset: int = 13) -> Path:
    """Convert the frozen backbone to ONNX for TensorFlow-free serving.

    This is the step that decouples the serving container from TensorFlow. The
    caller must verify numerical parity against the Keras model afterwards --
    `verify_onnx_parity` below -- because a silent export mismatch would
    degrade every downstream prediction without raising anything.
    """
    import tensorflow as tf
    import tf2onnx

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    spec = (tf.TensorSpec((None, *IMAGE_SIZE, 3), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(
        backbone, input_signature=spec, opset=opset, output_path=str(path))

    print(f"Exported backbone to {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def verify_onnx_parity(backbone, sample_images: np.ndarray,
                       path: str | Path = BACKBONE_PATH,
                       tolerance: float = 1e-4) -> dict:
    """Check the ONNX backbone reproduces the Keras backbone's embeddings.

    Raises if the two disagree. Failing loudly here is the entire point: an
    undetected export mismatch is invisible in every later metric.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    keras_out = backbone.predict(sample_images, verbose=0)
    onnx_out = session.run(None, {input_name: sample_images.astype(np.float32)})[0]

    max_diff = float(np.abs(keras_out - onnx_out).max())
    correlation = float(np.corrcoef(keras_out.ravel(), onnx_out.ravel())[0, 1])

    result = {
        "max_abs_diff": max_diff,
        "correlation": correlation,
        "passed": max_diff < tolerance,
        "n_samples": int(len(sample_images)),
    }
    if not result["passed"]:
        raise AssertionError(
            f"ONNX/Keras embedding mismatch: max abs diff {max_diff:.2e} "
            f"exceeds tolerance {tolerance:.0e}. Do not deploy this artifact."
        )

    print(f"  ONNX parity OK: max diff {max_diff:.2e}, correlation {correlation:.6f}")
    return result


def save_class_names(class_names: Sequence[str], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(class_names), indent=2))
    return path
