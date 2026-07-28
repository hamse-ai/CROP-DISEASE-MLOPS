# Crop Disease Classification — End-to-End ML Pipeline

Image classification of crop leaf disease (38 classes, 14 crop species), deployed as a
containerised prediction API with a monitoring UI, bulk data upload and one-click retraining.

> **Status:** in development. Deployed URLs, evaluation results, load-test tables and the video
> demo link are added as each stage lands — see [Build progress](#build-progress).

## The problem

Salaam MFB lends to smallholder farmers, and crop disease is a leading driver of agricultural
loan default. A previous project predicted *corporate* financial distress from financial ratios;
this one predicts the **upstream physical cause** of agricultural distress — crop disease — from a
photograph of a leaf, so that a loan officer can assess collateral risk in the field from a phone
photo rather than waiting for a harvest to fail.

## Architecture

The model is deliberately split in two:

```
image → [ MobileNetV2 · ImageNet · frozen · ONNX ] → 1280-d embedding → [ head ] → 38 classes
              never retrained, ~14 MB                     small, cheap, retrainable
```

This split is what makes the rest of the system practical:

| Consequence | Why it matters |
| --- | --- |
| Serving container needs only `onnxruntime`, `numpy`, `scikit-learn`, `pillow` | No TensorFlow, so the image fits the 512 MB free-tier memory limit |
| Retraining = embed new images, refit the head | Runs in seconds on CPU, so the retrain button works **on the deployed URL**, not just locally |
| A head is a few hundred KB | Keeping every version and supporting instant rollback costs almost nothing |
| Inference is a single small matmul after the backbone | Keeps latency low under load |

Retraining always fits on a **replay buffer** (a stratified sample of the original training
embeddings) *plus* the newly uploaded images — never on the uploads alone, which would cause
catastrophic forgetting.

## Repository layout

```
crop-disease-mlops/
├── notebook/     # the analysis: acquisition → EDA → training → evaluation → export
├── src/          # preprocessing, model, prediction, retraining, registry, monitoring
├── api/          # FastAPI prediction + retraining service
├── ui/           # Streamlit dashboard
├── data/         # train/ test/ uploads/ (full dataset is gitignored)
├── models/       # frozen backbone, classifier heads, class names, registry
├── docker/       # API + UI images, nginx load balancer
├── locust/       # load-test definition
└── reports/      # EDA figures, load-test results
```

## Setup

```bash
conda create -y -n cropml python=3.11
conda activate cropml
pip install -r requirements.txt
```

Fetch the dataset (~1.5 GB — the repo ships only 5 sample images per class):

```bash
python src/acquire_data.py
```

## Build progress

- [x] Repo scaffold and tooling
- [ ] Data acquisition and stratified splitting
- [ ] Preprocessing pipeline
- [ ] EDA and feature interpretation
- [ ] Model training and evaluation
- [ ] Artifact export and prediction-parity check
- [ ] Inference, registry, retraining and drift trigger
- [ ] FastAPI service
- [ ] Streamlit UI
- [ ] Docker and deployment
- [ ] Load testing across container counts

## Dataset

PlantVillage — 54,303 leaf images, 38 healthy/diseased classes across 14 crop species,
256×256 colour, from [spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset).

> Hughes, D.P. and Salathé, M. *An open access repository of images on plant health to enable the
> development of mobile disease diagnostics.* arXiv:1511.08060, 2015.
