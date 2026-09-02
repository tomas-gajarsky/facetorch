from pathlib import Path
import re
import subprocess

import pytest
import torch
import yaml

from facetorch import Dimensions, Face, InferenceError, Location, Prediction
from facetorch.downloader import DownloaderHuggingFace

REPO_ROOT = Path(__file__).resolve().parents[1]


def _first_existing(candidates):
    return next((path for path in candidates if path.is_file()), None)


def _marked_code(content, marker, language):
    marked = content.split(f"<!-- {marker}:start -->", 1)[1]
    marked = marked.split(f"<!-- {marker}:end -->", 1)[0].strip()
    prefix = f"```{language}\n"
    assert marked.startswith(prefix)
    assert marked.endswith("```")
    return marked[len(prefix) : -len("```")].rstrip()


def _runtime_example_source():
    guide = (REPO_ROOT / "docs" / "custom-components.md").read_text(encoding="utf-8")
    return _marked_code(
        guide,
        "facetorch-extension-runtime-example",
        "python",
    )


@pytest.mark.release_blocker
def test_unpublished_v1_changelog_is_not_marked_as_released():
    tag = subprocess.run(
        ["git", "tag", "--list", "v1.0.0"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if tag:
        return

    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    current_section = changelog.split("## 1.0.0rc3", 1)[1].split("\n## ", 1)[0]
    assert "Released on" not in current_section
    assert (
        "Unreleased" in changelog.splitlines()[2]
        or "release candidate" in current_section.lower()
    )


@pytest.mark.release_blocker
def test_current_rc_identity_and_model_governance_prose_are_consistent():
    project = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    compatibility = (REPO_ROOT / "docs/model-compatibility.md").read_text(
        encoding="utf-8"
    )

    assert 'version = "1.0.0rc3"' in project
    assert "v1.0.0-rc.3" in changelog
    assert "release_eligible: false" not in compatibility
    assert "governance is still incomplete" not in compatibility
    assert "all ten records are release-eligible" in readme.lower()
    assert "release_eligible: true" in compatibility


@pytest.mark.release_blocker
def test_v1_migration_guide_documents_compatibility_and_cache_rollback():
    migration = _first_existing(
        [
            REPO_ROOT / "MIGRATION.md",
            REPO_ROOT / "docs" / "migration-v1.md",
            REPO_ROOT / "docs" / "migration.md",
        ]
    )
    assert migration is not None

    content = migration.read_text(encoding="utf-8").lower()
    for required in ("0.6", "deprecat", "cache", "rollback", "batch_size"):
        assert required in content


@pytest.mark.release_blocker
def test_rc_onboarding_selects_exact_candidate_channels():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    migration = (REPO_ROOT / "docs" / "migration-v1.md").read_text(encoding="utf-8")
    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    combined = f"{readme}\n{migration}"
    assert '"facetorch==1.0.0rc3"' in combined
    assert '"torch==2.13.0+cpu"' in combined
    assert '"torch==2.13.0+cu130"' in combined
    assert "FACETORCH_DOCKER_TAG=1.0.0-rc.3" in combined
    assert "${FACETORCH_DOCKER_TAG:-1.0.0-rc.3}" in compose
    assert "facetorch:latest" not in compose
    assert "facetorch-gpu:latest" not in compose
    assert "pip install facetorch\n" not in readme
    assert "conda install -c conda-forge facetorch" not in readme
    assert "facetorch-user-guide-a0e9fd2a5552" not in readme
    assert "facetorch-app" not in readme


@pytest.mark.release_blocker
def test_security_policy_defines_private_contact_and_response_targets():
    security = REPO_ROOT / "SECURITY.md"
    assert security.is_file()

    content = security.read_text(encoding="utf-8").lower()
    assert "private" in content
    assert "five business days" in content or "5 business days" in content
    assert "fourteen" in content or "14" in content
    assert "supported version" in content


@pytest.mark.release_blocker
def test_release_contains_a_per_model_rights_and_integrity_manifest():
    manifest = _first_existing(
        [
            REPO_ROOT / "MODEL_MANIFEST.json",
            REPO_ROOT / "MODEL_MANIFEST.yaml",
            REPO_ROOT / "facetorch" / "models" / "governance.json",
            REPO_ROOT / "facetorch" / "resources" / "model_manifest.json",
            REPO_ROOT / "facetorch" / "resources" / "model_manifest.yaml",
        ]
    )
    assert manifest is not None

    content = manifest.read_text(encoding="utf-8").lower()
    for required in ("sha256", "revision", "license", "provenance", "limitations"):
        assert required in content


@pytest.mark.release_blocker
def test_model_notice_separates_code_and_weight_rights():
    notice = REPO_ROOT / "MODEL_NOTICE.md"
    assert notice.is_file()

    content = notice.read_text(encoding="utf-8").lower()
    for required in (
        "apache-2.0",
        "does not",
        "model weights",
        "governance.json",
        "redistribution",
        "consequential decisions",
    ):
        assert required in content


@pytest.mark.release_blocker
def test_release_runbook_covers_yank_revocation_and_rollback():
    runbook = _first_existing(
        [
            REPO_ROOT / "RELEASING.md",
            REPO_ROOT / "docs" / "release-runbook.md",
            REPO_ROOT / "docs" / "release.md",
        ]
    )
    assert runbook is not None

    content = runbook.read_text(encoding="utf-8").lower()
    for required in ("yank", "revocation", "rollback", "latest", "release candidate"):
        assert required in content


@pytest.mark.release_blocker
def test_generated_api_docs_cover_the_public_top_level_modules():
    package_root = REPO_ROOT / "facetorch"
    documented = REPO_ROOT / "docs" / "facetorch"
    missing = [
        module.stem
        for module in sorted(package_root.glob("*.py"))
        if module.name != "__init__.py"
        and not (documented / f"{module.stem}.html").is_file()
    ]
    assert missing == []


@pytest.mark.release_blocker
def test_generated_analyzer_docs_describe_the_v1_result_contract():
    content = (REPO_ROOT / "docs" / "facetorch" / "analyzer" / "core.html").read_text(
        encoding="utf-8"
    )
    assert "AnalysisResult" in content
    assert "face_batch_size" in content
    assert "If return_img_data is False" not in content

    base_docs = (REPO_ROOT / "docs" / "facetorch" / "base.html").read_text(
        encoding="utf-8"
    )
    assert "raw <code>.pth</code> state dictionary" in base_docs
    assert "weights are extracted from the TorchScript file" not in base_docs


@pytest.mark.release_blocker
def test_extension_guide_separates_private_and_shipped_model_paths(tmp_path):
    guide = (REPO_ROOT / "docs" / "custom-components.md").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    migration = (REPO_ROOT / "docs" / "migration-v1.md").read_text(encoding="utf-8")

    for required in (
        "packaged-manifest mode",
        "direct external mode",
        "analyzer.predictors",
        "analyzer.detector",
        "Custom predictor contract",
        "Custom detector contract",
        "load_config_from_path()",
        'overrides=["+analyzer.device=cpu"]',
        "generic exporter for an arbitrary user model",
        "Contributing an officially shipped model",
        "deselect the defaults",
        "from facetorch.analyzer.detector import DetectorPostprocessorProtocol",
        "raw state dictionary saved with a `.pt` suffix",
        "Direct Hugging Face mode supports authenticated `.pt2`",
    ):
        assert required in guide

    direct_yaml = _marked_code(
        guide,
        "facetorch-direct-artifact-yaml",
        "yaml",
    )
    for required in (
        "repo_id:",
        "filename:",
        "revision:",
        "sha256:",
        "size_bytes:",
        "expected_format: pt2",
        "device:",
    ):
        assert required in direct_yaml
    assert "manifest_id:" not in direct_yaml

    direct_config = yaml.safe_load(direct_yaml)
    downloader = direct_config["downloader"]
    assert downloader["_target_"] == "facetorch.downloader.DownloaderHuggingFace"
    assert len(downloader["revision"]) == 40
    assert len(downloader["sha256"]) == 64
    assert isinstance(downloader["size_bytes"], int)
    assert "manifest_id" not in downloader

    direct_values = dict(downloader)
    direct_values.pop("_target_")
    direct_values["path_local"] = str(tmp_path / direct_values["filename"])
    direct_values["offline"] = True
    direct_values["device"] = "cpu"
    descriptor = DownloaderHuggingFace(**direct_values)._resolve_candidates()[0]
    assert descriptor.repo_id == direct_values["repo_id"]
    assert descriptor.revision == direct_values["revision"]
    assert descriptor.filename == direct_values["filename"]
    assert descriptor.sha256 == direct_values["sha256"]
    assert descriptor.size_bytes == direct_values["size_bytes"]
    assert descriptor.format == "pt2"

    repository_revision = r"[0-9a-f]{40}"
    for path in (
        "docs/custom-components.md",
        "docs/migration-v1.md",
        "docs/model-compatibility.md",
        "docs/model-publication.md",
        "facetorch/models/governance.json",
    ):
        assert re.search(
            rf"https://github\.com/tomas-gajarsky/facetorch/blob/"
            rf"{repository_revision}/{re.escape(path)}",
            readme,
        )
    assert "github.com/tomas-gajarsky/facetorch/blob/main/" not in readme
    assert "raw.githubusercontent.com/tomas-gajarsky/facetorch/main/" not in readme
    assert "custom-components.md" in migration

    for required in (
        "model_defs/",
        " validate \\",
        "--model-ids",
        "tests/conftest.py",
        "conf/tests.config.N.yaml",
        "flake8 --config=.flake8",
    ):
        assert required in guide


@pytest.mark.release_blocker
def test_extension_guide_runtime_example_executes_without_model_access(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("FACETORCH_CACHE_DIR", str(tmp_path / "cache"))
    source = _runtime_example_source()

    exec(compile(source, "docs/custom-components.md", "exec"), {})


@pytest.mark.release_blocker
def test_extension_predictor_contract_batches_and_preserves_order(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("FACETORCH_CACHE_DIR", str(tmp_path / "cache"))
    namespace = {}
    source = _runtime_example_source()
    exec(compile(source, "docs/custom-components.md", "exec"), namespace)
    analyzer = namespace["analyzer"]
    analyzer.utilizers = {}

    class ThreeFaceDetector:
        def run(self, data):
            data.faces = [
                Face(
                    indx=index,
                    loc=Location(x1=0, y1=0, x2=4, y2=4),
                    dims=Dimensions(height=4, width=4),
                    tensor=data.tensor[0],
                    ratio=1.0,
                )
                for index in range(3)
            ]
            return data

    class RecordingPredictor:
        max_batch_size = 2

        def __init__(self):
            self.batch_sizes = []
            self.next_index = 0

        def run(self, faces):
            self.batch_sizes.append(len(faces))
            predictions = [
                Prediction(label=str(self.next_index + offset))
                for offset in range(len(faces))
            ]
            self.next_index += len(faces)
            return predictions

    predictor = RecordingPredictor()
    analyzer.detector = ThreeFaceDetector()
    analyzer.predictors["recording"] = predictor
    result = analyzer.run(
        torch.zeros((3, 4, 4), dtype=torch.uint8),
        include_predictors=["recording"],
    )

    assert predictor.batch_sizes == [2, 1]
    assert [face.preds["recording"].label for face in result.faces] == ["0", "1", "2"]


@pytest.mark.release_blocker
def test_extension_predictor_contract_rejects_wrong_prediction_count(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("FACETORCH_CACHE_DIR", str(tmp_path / "cache"))
    namespace = {}
    source = _runtime_example_source()
    exec(compile(source, "docs/custom-components.md", "exec"), namespace)
    analyzer = namespace["analyzer"]
    analyzer.utilizers = {}

    class WrongCountPredictor:
        max_batch_size = None

        def run(self, _faces):
            return []

    analyzer.predictors["wrong_count"] = WrongCountPredictor()
    with pytest.raises(InferenceError, match="0 prediction.*1 input face"):
        analyzer.run(
            torch.zeros((3, 4, 4), dtype=torch.uint8),
            include_predictors=["wrong_count"],
        )
