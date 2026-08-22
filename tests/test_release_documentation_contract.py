from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _first_existing(candidates):
    return next((path for path in candidates if path.is_file()), None)


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
    v1_section = changelog.split("## 0.6.2", 1)[0]
    assert "Released on" not in v1_section
    assert "Unreleased" in v1_section or "release candidate" in v1_section.lower()


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
    content = (
        REPO_ROOT / "docs" / "facetorch" / "analyzer" / "core.html"
    ).read_text(encoding="utf-8")
    assert "AnalysisResult" in content
    assert "face_batch_size" in content
    assert "If return_img_data is False" not in content
