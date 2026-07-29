"""Generate notebook/colab_deep_models.ipynb — the GPU half of the comparison.

Configurations 1–3 (scratch CNN, frozen transfer, fine-tuned) need a GPU: a
single epoch over 38k images at 224x224 takes hours on CPU. This notebook runs
them on a Colab T4 and writes `reports/deep_model_results.json`, which the main
notebook merges to complete the four-way comparison.

It deliberately clones this repository and reuses `src/acquire_data.py` rather
than re-implementing the split. The seed and the sorted file ordering are what
make the Colab splits byte-identical to the local ones — without that, the deep
models and the deployed head would be scored on different test sets and the
comparison would be meaningless.

    python notebook/build_colab_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent / "colab_deep_models.ipynb"

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


# ==========================================================================
md(r"""
# Deep model comparison — Colab GPU

Companion to `crop_disease_mlops.ipynb`. That notebook trains and deploys **configuration 4**
(frozen MobileNetV2 + a scikit-learn head), which needs no GPU. This one trains the three
configurations that do:

| # | Configuration | What it answers |
|---|---|---|
| 1 | CNN from scratch | How far does the data alone get us, with no ImageNet? |
| 2 | MobileNetV2 frozen + dense head | How much does transfer learning buy? |
| 3 | MobileNetV2 fine-tuned | Does unfreezing the last blocks help further? |

It writes `reports/deep_model_results.json`, which the main notebook merges to produce the
full four-way table.

---

## Before you start

1. **Runtime → Change runtime type → T4 GPU.** The first cell will refuse to continue on CPU.
2. Set `REPO_URL` below to this project's GitHub URL.
3. **Runtime → Run all.**

### Why this clones the repo

The splits must be *identical* to the local run, or the deep models and the deployed head
are scored on different test sets and nothing is comparable. Rather than re-implement
splitting here and hope it matches, this notebook clones the repo and calls the same
`src/acquire_data.py`, whose fixed seed and sorted file ordering make the result
reproducible.
""")

code(r"""
#@title Configuration { display-mode: "form" }

# Set this to your repository URL.
REPO_URL = "https://github.com/hamse-ai/crop-disease-mlops.git"  #@param {type:"string"}

# FAST_MODE trains on a stratified 40% of the training split for fewer epochs.
# Use it for a first pass; turn it off for the numbers you actually report.
#   FAST_MODE = True   ~35 min total on a T4
#   FAST_MODE = False  ~2 hours total on a T4
FAST_MODE = True  #@param {type:"boolean"}

# Checkpoints survive a Colab disconnect. Re-running a training cell resumes
# from the last completed epoch rather than starting over.
USE_DRIVE = True  #@param {type:"boolean"}

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64 if FAST_MODE else 32

EPOCHS = {"scratch": 12, "frozen": 8, "finetune": 6} if FAST_MODE \
    else {"scratch": 30, "frozen": 15, "finetune": 12}
TRAIN_FRACTION = 0.40 if FAST_MODE else 1.0

print(f"FAST_MODE={FAST_MODE}  epochs={EPOCHS}  train_fraction={TRAIN_FRACTION}")
""")

code(r"""
# Refuse to run on CPU rather than silently taking 10 hours.
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    raise SystemExit(
        "No GPU detected. Runtime -> Change runtime type -> T4 GPU, then Run all.\n"
        "On CPU a single scratch-CNN epoch over 38k images takes roughly an hour."
    )

print(f"TensorFlow {tf.__version__}")
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Mixed precision roughly halves epoch time on a T4. The final softmax is
# forced back to float32 so the loss stays numerically stable.
if not FAST_MODE:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("mixed precision: enabled")
""")

code(r"""
import os
from pathlib import Path

if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    CHECKPOINT_ROOT = Path("/content/drive/MyDrive/crop_disease_colab")
else:
    CHECKPOINT_ROOT = Path("/content/checkpoints")
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
print("checkpoints ->", CHECKPOINT_ROOT)
""")

md(r"""
## 1. Clone the repo and fetch the data

`src/acquire_data.py` performs a blobless sparse checkout of PlantVillage's colour variant
and applies the same seeded, per-class stratified split as the local run.
""")

code(r"""
%cd /content
if not Path("/content/crop-disease-mlops").exists():
    !git clone --quiet $REPO_URL crop-disease-mlops
%cd /content/crop-disease-mlops

import sys
sys.path.insert(0, "/content/crop-disease-mlops")
print("repo:", Path.cwd())
""")

code(r"""
# ~1.5 GB. A few minutes on Colab's network.
if not (Path("data/train").exists() and len(list(Path("data/train").glob("*/"))) >= 38):
    !python src/acquire_data.py --move
else:
    print("data already present")
""")

code(r"""
import json
import numpy as np

from src.config import CLASS_NAMES_PATH, DATA_DIR, RANDOM_SEED, TEST_DIR, TRAIN_DIR
from src.preprocessing import build_dataset, compute_class_weights, list_images

VAL_DIR = DATA_DIR / "val"
class_names = json.loads(Path(CLASS_NAMES_PATH).read_text())

counts = {s: sum(len(list_images(Path(d) / c)) for c in class_names)
          for s, d in (("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR))}
print(f"{len(class_names)} classes")
for split, n in counts.items():
    print(f"  {split:<6} {n:>7,}")

# Sanity: these must match the local run, or nothing below is comparable.
assert len(class_names) == 38, "expected 38 classes"
""")

md(r"""
## 2. Data pipelines

Built with `src/preprocessing.build_dataset`, the same function the local notebook uses —
same 224×224 resize, same [−1, 1] rescaling, same augmentation set (and the same deliberate
absence of hue jitter, since hue shift *is* the disease signal).
""")

code(r"""
train_ds = build_dataset(TRAIN_DIR, class_names, batch_size=BATCH_SIZE,
                         shuffle=True, augment=True)
val_ds = build_dataset(VAL_DIR, class_names, batch_size=BATCH_SIZE)
test_ds = build_dataset(TEST_DIR, class_names, batch_size=BATCH_SIZE)

if TRAIN_FRACTION < 1.0:
    # Take whole batches from an already-shuffled stream: cheaper than
    # re-stratifying and close enough to stratified at this batch count.
    n_batches = int((counts["train"] / BATCH_SIZE) * TRAIN_FRACTION)
    train_ds = train_ds.take(n_batches)
    print(f"FAST_MODE: training on ~{n_batches * BATCH_SIZE:,} images "
          f"({TRAIN_FRACTION:.0%} of the split)")

class_weights = compute_class_weights(TRAIN_DIR, class_names)
print(f"class weights span {min(class_weights.values()):.2f} to "
      f"{max(class_weights.values()):.2f} — the 36x imbalance")
""")

md(r"""
## 3. Training helper

Every configuration is trained the same way so the comparison is fair: same optimiser,
same callbacks, same class weighting, same early-stopping criterion on validation macro
accuracy.

Checkpointing is per-epoch. If the Colab runtime drops, re-running the cell resumes from
the last completed epoch instead of starting over.
""")

code(r"""
import time

from src.model import compile_model, default_callbacks


def train(model, name, epochs, learning_rate=1e-3):
    checkpoint = CHECKPOINT_ROOT / f"{name}.keras"
    csv_log = CHECKPOINT_ROOT / f"{name}_history.csv"

    compile_model(model, learning_rate=learning_rate)
    callbacks = default_callbacks(checkpoint, patience=4)
    callbacks.append(tf.keras.callbacks.CSVLogger(str(csv_log), append=True))

    if checkpoint.exists():
        try:
            model.load_weights(checkpoint)
            print(f"[{name}] resumed from checkpoint")
        except Exception as exc:
            print(f"[{name}] could not resume ({exc}); training from scratch")

    started = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        class_weight=class_weights, callbacks=callbacks, verbose=1)
    elapsed = time.time() - started
    print(f"[{name}] {elapsed/60:.1f} min, {len(history.history['loss'])} epochs")
    return model, history, elapsed
""")

code(r"""
from src.model import evaluate_predictions


def evaluate(model, name, history, elapsed):
    # Score on the held-out test split and package the result for merging.
    probabilities, labels = [], []
    for batch_x, batch_y in test_ds:
        probabilities.append(model.predict(batch_x, verbose=0))
        labels.append(batch_y.numpy())

    y_proba = np.concatenate(probabilities).astype(np.float64)
    y_true = np.concatenate(labels)

    metrics = evaluate_predictions(y_true, y_proba, class_names)
    trainable = sum(int(np.prod(v.shape)) for v in model.trainable_weights)

    print(f"\n[{name}]  macro-F1 {metrics['f1_macro']:.4f}   "
          f"accuracy {metrics['accuracy']:.4f}   top-5 {metrics.get('top5_accuracy', 0):.4f}")

    return {
        "name": name,
        "total_params": int(model.count_params()),
        "trainable_params": int(trainable),
        "epochs_run": len(history.history["loss"]),
        "train_seconds": round(elapsed, 1),
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "metrics": metrics,
    }


results = {}
""")

md(r"""
## 4. Configuration 1 — CNN from scratch

The control. Without it there is no way to say how much of the final accuracy comes from
ImageNet transfer rather than from the dataset itself.
""")

code(r"""
from src.model import build_scratch_cnn

tf.keras.utils.set_random_seed(SEED)
model, history, elapsed = train(build_scratch_cnn(len(class_names)),
                                "scratch_cnn", EPOCHS["scratch"])
results["scratch_cnn"] = evaluate(model, "scratch_cnn", history, elapsed)
del model
tf.keras.backend.clear_session()
""")

md(r"""
## 5. Configuration 2 — MobileNetV2 frozen + dense head

Transfer learning with the backbone entirely frozen. This is the closest deep analogue to
the deployed configuration 4 — the difference is that the head here is a Keras dense layer
trained through the augmentation pipeline, rather than a scikit-learn head fitted on
embeddings cached once.
""")

code(r"""
from src.model import build_transfer_model

tf.keras.utils.set_random_seed(SEED)
model, history, elapsed = train(build_transfer_model(len(class_names), fine_tune_from=None),
                                "mobilenetv2_frozen", EPOCHS["frozen"])
results["mobilenetv2_frozen"] = evaluate(model, "mobilenetv2_frozen", history, elapsed)
del model
tf.keras.backend.clear_session()
""")

md(r"""
## 6. Configuration 3 — MobileNetV2 fine-tuned

The last inverted-residual blocks are unfrozen, at a **10× lower learning rate** — full-rate
gradients would wash out the pretrained weights in the first few steps.

BatchNorm layers stay frozen throughout: updating their running statistics on small batches
destabilises a pretrained backbone, and is a common way fine-tuning silently performs worse
than the frozen model it started from.
""")

code(r"""
tf.keras.utils.set_random_seed(SEED)
model, history, elapsed = train(build_transfer_model(len(class_names), fine_tune_from=100),
                                "mobilenetv2_finetuned", EPOCHS["finetune"],
                                learning_rate=1e-4)
results["mobilenetv2_finetuned"] = evaluate(model, "mobilenetv2_finetuned", history, elapsed)
del model
tf.keras.backend.clear_session()
""")

md(r"""
## 7. Results

Written to `reports/deep_model_results.json`. Download it and drop it into the repo's
`reports/` directory — the main notebook picks it up automatically and renders the
four-way comparison.
""")

code(r"""
import pandas as pd

meta = {
    "fast_mode": FAST_MODE,
    "train_fraction": TRAIN_FRACTION,
    "batch_size": BATCH_SIZE,
    "image_size": list(IMAGE_SIZE),
    "epochs_configured": EPOCHS,
    "seed": SEED,
    "n_train": counts["train"],
    "n_val": counts["val"],
    "n_test": counts["test"],
    "n_classes": len(class_names),
    "gpu": tf.config.list_physical_devices("GPU")[0].name,
}

payload = {"meta": meta, "models": results}
out = Path("reports/deep_model_results.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2))

table = pd.DataFrame([
    {"configuration": r["name"],
     "macro-F1": r["metrics"]["f1_macro"],
     "accuracy": r["metrics"]["accuracy"],
     "top-5": r["metrics"].get("top5_accuracy"),
     "trainable params": f"{r['trainable_params']:,}",
     "epochs": r["epochs_run"],
     "minutes": round(r["train_seconds"] / 60, 1)}
    for r in results.values()
]).sort_values("macro-F1", ascending=False)

display(table.style.format({"macro-F1": "{:.4f}", "accuracy": "{:.4f}",
                            "top-5": "{:.4f}"}).hide(axis="index"))

if FAST_MODE:
    print("\nFAST_MODE was on: trained on a subset for fewer epochs.")
    print("These numbers understate every configuration. Re-run with")
    print("FAST_MODE = False for the figures you report.")
""")

code(r"""
from google.colab import files
files.download("reports/deep_model_results.json")
""")

md(r"""
### Merging back

Place the downloaded `deep_model_results.json` in the repo's `reports/` directory and
re-run `crop_disease_mlops.ipynb`. Section 4 detects it and renders the full four-way
comparison; without it, that section states plainly that the deep baselines were not run
rather than leaving a gap.
""")

nb["cells"] = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
}

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells: "
      f"{sum(c['cell_type']=='markdown' for c in cells)} markdown, "
      f"{sum(c['cell_type']=='code' for c in cells)} code)")
