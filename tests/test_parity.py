"""Regression tests for the ONNX/Keras parity check.

The check itself is the deployment gate, so its failure semantics need testing:
it must reject a genuinely broken export and must not reject a correct one over
float32 accumulation noise. An earlier absolute-tolerance version did the
latter -- failing at 1.03e-4 on a batch where every prediction agreed exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BACKBONE_PATH
from src.model import verify_onnx_parity


class _FakeBackbone:
    """Returns fixed embeddings, so the comparison is fully controlled."""

    def __init__(self, output: np.ndarray):
        self._output = output

    def predict(self, _inputs, verbose=0):
        return self._output


class _FakeHead:
    """Softmax over the first few embedding dimensions."""

    def predict_proba(self, X):
        logits = np.asarray(X, dtype=np.float64)[:, :4]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)


@pytest.mark.skipif(not Path(BACKBONE_PATH).exists(),
                    reason="backbone not exported yet")
def test_real_export_passes_with_downstream_check():
    """The shipped artifact must clear the gate on real images."""
    from src.model import load_head
    from src.preprocessing import list_images, load_image, preprocess_for_backbone
    from src.registry import ModelRegistry

    import onnxruntime as ort  # noqa: F401  (ensures the runtime is installed)

    test_dir = Path(__file__).resolve().parent.parent / "data" / "test"
    images = []
    for class_dir in sorted(d for d in test_dir.iterdir() if d.is_dir())[:10]:
        images += [load_image(p) for p in list_images(class_dir)[:2]]
    if not images:
        pytest.skip("no test images available")

    batch = preprocess_for_backbone(np.stack(images))

    import warnings
    warnings.filterwarnings("ignore")
    from src.model import build_backbone

    head_path = ModelRegistry().head_path()
    head = load_head(head_path) if head_path else None

    result = verify_onnx_parity(build_backbone(), batch, BACKBONE_PATH, head=head)

    assert result["passed"]
    assert result["max_relative_diff"] < 1e-3
    if head is not None:
        assert result["top1_agreement"] == 1.0


def test_float32_noise_does_not_fail_the_gate(monkeypatch, tmp_path):
    """Tiny perturbations that change no prediction must pass.

    This is the exact regression: absolute error crossing 1e-4 while relative
    error stays at ~1e-5 and top-1 agreement is perfect.
    """
    rng = np.random.default_rng(0)
    embeddings = rng.uniform(0, 5, (75, 1280)).astype(np.float32)
    noisy = embeddings + rng.normal(0, 2e-5, embeddings.shape).astype(np.float32)

    class _Session:
        def get_inputs(self):
            return [type("I", (), {"name": "input"})()]

        def run(self, _outputs, _feed):
            return [noisy]

    import onnxruntime as ort
    monkeypatch.setattr(ort, "InferenceSession", lambda *a, **k: _Session())

    result = verify_onnx_parity(_FakeBackbone(embeddings),
                                np.zeros((75, 224, 224, 3), np.float32),
                                tmp_path / "fake.onnx", head=_FakeHead())

    assert result["passed"], "float32 noise must not fail the gate"
    assert result["top1_agreement"] == 1.0


def test_genuinely_broken_export_is_rejected(monkeypatch, tmp_path):
    """A real mismatch must still raise -- the gate has to bite."""
    rng = np.random.default_rng(1)
    embeddings = rng.uniform(0, 5, (40, 1280)).astype(np.float32)
    # Shuffling the first dimensions changes predicted classes: exactly the
    # kind of silent layout bug the check exists to catch.
    broken = embeddings.copy()
    broken[:, :4] = broken[:, :4][:, ::-1]

    class _Session:
        def get_inputs(self):
            return [type("I", (), {"name": "input"})()]

        def run(self, _outputs, _feed):
            return [broken]

    import onnxruntime as ort
    monkeypatch.setattr(ort, "InferenceSession", lambda *a, **k: _Session())

    with pytest.raises(AssertionError, match="parity failed"):
        verify_onnx_parity(_FakeBackbone(embeddings),
                           np.zeros((40, 224, 224, 3), np.float32),
                           tmp_path / "fake.onnx", head=_FakeHead())
