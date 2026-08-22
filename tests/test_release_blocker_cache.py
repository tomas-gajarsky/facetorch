import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from facetorch.artifacts import detect_model_format, get_model_manifest
from facetorch.exceptions import ModelCompatibilityError
from facetorch.model_cache import inspect_legacy_cache


REPO_ROOT = Path(__file__).resolve().parents[1]


def _legacy_manifest(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "manifest_version": 1,
        "manifest_revision": "restart-test-v1",
        "status": "provisional",
        "models": {
            "toy": {
                "task": "test",
                "source": "huggingface",
                "repo_id": "owner/toy",
                "revision": "a" * 40,
                "artifacts": [
                    {
                        "id": "toy-legacy",
                        "filename": "model.pt",
                        "format": "torchscript",
                        "sha256": digest,
                        "size_bytes": path.stat().st_size,
                        "torch_min": "2.11",
                        "torch_max_exclusive": "2.12",
                        "devices": ["cpu"],
                        "schema_major": None,
                        "schema_minor": None,
                        "validation_metadata": None,
                    }
                ],
            }
        },
    }


@pytest.mark.release_blocker
def test_verified_legacy_cache_loads_in_two_independent_processes(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    path_cache = cache_dir / "model.pt"
    scripted = torch.jit.trace(torch.nn.Identity(), torch.zeros((1, 4)))
    torch.jit.save(scripted, str(path_cache))
    manifest_json = json.dumps(_legacy_manifest(path_cache))
    script = """
import json
import torch
from facetorch.artifacts import ArtifactManifest
from facetorch.base import BaseModel
from facetorch.downloader import DownloaderHuggingFace

class Model(BaseModel):
    def run(self, tensor):
        return self.inference(tensor)

manifest = ArtifactManifest.from_json(__import__('os').environ['TEST_MANIFEST'])
downloader = DownloaderHuggingFace(
    file_id='owner/toy', repo_id='owner/toy', filename='model.pt2',
    path_local=__import__('os').environ['TEST_CONFIGURED_PATH'],
    manifest_id='toy', manifest=manifest, torch_version='2.11.0',
    device='cpu', offline=True, allow_legacy_models=True,
)
model = Model(downloader=downloader, device=torch.device('cpu'))
assert torch.equal(model.run(torch.ones((1, 4))), torch.ones((1, 4)))
assert downloader.path_local.endswith('model.pt')
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT)
    environment["TEST_MANIFEST"] = manifest_json
    environment["TEST_CONFIGURED_PATH"] = str(cache_dir / "model.pt2")

    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr


@pytest.mark.release_blocker
def test_legacy_models_require_explicit_opt_in_and_cpu_eligibility():
    manifest = get_model_manifest()
    default_candidates = manifest.candidates(
        "au-opengraph",
        torch_version="2.11.0",
        device="cpu",
        allow_legacy_models=False,
    )
    assert all(candidate.format == "pt2" for candidate in default_candidates)

    with pytest.raises(ModelCompatibilityError):
        manifest.candidates(
            "au-opengraph",
            torch_version="2.10.0",
            device="cuda",
            allow_legacy_models=True,
        )


@pytest.mark.release_blocker
def test_mislabeled_legacy_cache_is_detected_without_execution(tmp_path):
    path_cache = tmp_path / "model.pt2"
    scripted = torch.jit.trace(torch.nn.Identity(), torch.zeros((1, 4)))
    torch.jit.save(scripted, str(path_cache))

    assert detect_model_format(path_cache) == "torchscript"
    inspected = inspect_legacy_cache(tmp_path)
    assert len(inspected) == 1
    assert inspected[0].mislabeled is True
    assert inspected[0].detected_format == "torchscript"
