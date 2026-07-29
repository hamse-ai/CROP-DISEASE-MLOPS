"""Guards on what the serving container is allowed to depend on.

The API image deliberately omits TensorFlow and OpenCV. Both are easy to
reintroduce by accident -- a convenience import at the top of a module is all
it takes -- and the failure would only surface as a broken deploy or a silently
fatter image, not as a test failure. So the constraint is asserted here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Modules the API actually imports, directly or transitively.
REQUEST_PATH_MODULES = ["src.config", "src.preprocessing", "src.prediction",
                        "src.registry", "src.monitoring", "src.retrain",
                        "api.schemas", "api.main"]

FORBIDDEN = {
    "cv2": "opencv-python-headless (~137 MB) is not in api/requirements.txt",
    "tensorflow": "TensorFlow is not installed in the serving image",
    "matplotlib": "matplotlib is not installed in the serving image",
}


def _import_probe(modules: list[str], forbidden: list[str]) -> subprocess.CompletedProcess:
    """Import modules in a subprocess where `forbidden` cannot be imported.

    Blocking at the meta-path level reproduces the container faithfully: the
    package is genuinely absent there, so a module-level `import cv2` raises
    rather than quietly succeeding as it would in this dev environment.
    """
    script = f"""
import sys

BLOCKED = {forbidden!r}

class Blocker:
    def find_module(self, name, path=None):
        return self if name.split(".")[0] in BLOCKED else None
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{{name}} is blocked: not installed in the serving image")
        return None

sys.meta_path.insert(0, Blocker())
sys.path.insert(0, {str(PROJECT_ROOT)!r})

for module in {modules!r}:
    __import__(module)
print("OK")
"""
    return subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=180)


@pytest.mark.parametrize("package,reason", sorted(FORBIDDEN.items()))
def test_request_path_imports_without(package, reason):
    """Importing the whole API must not require these packages."""
    result = _import_probe(REQUEST_PATH_MODULES, [package])
    assert result.returncode == 0, (
        f"the API failed to import without {package}.\n{reason}\n"
        f"stderr:\n{result.stderr[-1500:]}"
    )


def test_api_requirements_exclude_heavy_packages():
    """The manifest itself must stay clean -- the image is built from it."""
    manifest = (PROJECT_ROOT / "api" / "requirements.txt").read_text().lower()
    declared = [line.split("==")[0].split("[")[0].strip()
                for line in manifest.splitlines()
                if line.strip() and not line.strip().startswith("#")]

    for package in ("opencv-python-headless", "opencv-python", "tensorflow",
                    "torch", "matplotlib"):
        assert package not in declared, (
            f"{package} is declared in api/requirements.txt. The serving image "
            "is kept lean deliberately; add it to the dev requirements instead."
        )


def test_eda_extractors_still_work_with_cv2_present():
    """The dev environment keeps cv2, and the EDA path must still function.

    Dropping cv2 from the *serving* image must not mean dropping the feature
    extractors the notebook depends on.
    """
    import numpy as np

    from src.preprocessing import colour_statistics, texture_features

    image = np.full((64, 64, 3), 120, np.uint8)
    assert "excess_green" in colour_statistics(image)
    assert "edge_density" in texture_features(image)
