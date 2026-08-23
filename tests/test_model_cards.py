import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pytest
import yaml

from scripts import render_model_cards as model_card_renderer
from scripts.render_model_cards import (
    ModelCardError,
    render_model_cards,
    render_model_documents,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_UPSTREAM_LICENSE_SHA256 = {
    "https://github.com/biubug6/Pytorch_Retinaface/blob/b984b4b775b2c4dced95c1eadd195a5c7d32a60b/LICENSE.MIT": "41034e3430e3b7fd63031bcc6f9dd9740fa910328f3168208ca032807f026147",
    "https://github.com/choyingw/SynergyNet/blob/9de11e2831b85254d776ca748fa2ffd68aedf4ba/LICENSE": "fb0caa3ee56ab0a267ddc98fac1cb010e54cc6994103ee217d7631337b0bbe78",
    "https://github.com/lingjivoo/OpenGraphAU/blob/a0ad10d516ed121f476cbc6d66a84ebfc033a53f/LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "https://github.com/CVI-SZU/ME-GraphAU/blob/dda53cba00c427a0494468c549e464a96ba7b2fa/LICENSE": "a4420e16a065cd7b27e4171ce7b962cea16c313ed3cb070d3ddd8c223784154e",
    "https://github.com/selimsef/dfdc_deepfake_challenge/blob/89c6290490bac96b29193a4061b3db9dd3933e36/LICENSE": "0671780885c3623f2f1320c9668eea54252deea384a86acb8013abc3d3adf64c",
    "https://github.com/1adrianb/unsupervised-face-representation/blob/8bdb3cc7fddbee147d7188c0bda103272238d1bb/LICENSE": "5a9669baf02e6285f0f05f4b25b3f702e5f25c27d24967d94eb7708781c49d5e",
    "https://github.com/cydonia999/VGGFace2-pytorch/blob/c6e10f277b31b972c78fac68a40464a36a46a10d/LICENSE": "6a44da9f4025320e490417d0c19dc15e82fbffa56896f45565d3d93432215545",
    "https://github.com/sb-ai-lab/EmotiEffLib/blob/520a051c64cd191521e5934655314e769a319684/LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "https://github.com/kdhht2334/ELIM_FER/blob/fb1aa0d495ffcf38494c487135390ba1a5521fc4/LICENSE": "71f4adc55401be1d3fec0ebbef9c9844880d9e74fb06e1191866150fc9361c98",
    "https://github.com/mk-minchul/AdaFace/blob/c60eaa786a42c03444f3df7096dbaf9d57ae010d/LICENSE": "95b6e493eb9dba27f2150304e790ae254bab18d1611f4d6e2ade28fa3a271583",
    "https://github.com/junuke/UNPG/blob/a6f9c1731a68fc035eb8fe8198f5a5c643825a5b/LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "https://github.com/IrvingMeng/MagFace/blob/99bae614ac2643b9694bf18e0c5645272ff6acfa/LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.release_blocker
def test_model_cards_render_from_the_release_contract(tmp_path):
    manifest = _json(REPO_ROOT / "facetorch/models/manifest.json")
    governance = _json(REPO_ROOT / "facetorch/models/governance.json")
    catalog = _json(REPO_ROOT / "model_cards/catalog.json")
    documents = render_model_documents()
    rendered = render_model_cards(tmp_path)

    assert set(rendered) == set(manifest["models"])
    assert set(catalog["models"]) == set(manifest["models"])
    for model_id, model in manifest["models"].items():
        root = tmp_path / model_id
        assert set(rendered[model_id]) == {
            "LICENSE",
            "README.md",
            "THIRD_PARTY_NOTICES.md",
        }
        assert {
            filename: (root / filename).read_bytes()
            for filename in rendered[model_id]
        } == documents[model_id]
        card = (root / "README.md").read_text(encoding="utf-8")
        frontmatter = yaml.safe_load(card.split("---", 2)[1])
        rights = governance["models"][model_id]["rights"]
        expected_tag = {
            "MIT": "mit",
            "Apache-2.0": "apache-2.0",
        }[rights["weights_license"]]
        assert frontmatter["license"] == expected_tag
        assert model_id in card
        assert "Mapping method:" in card
        assert governance["models"][model_id]["source_checkpoint"][
            "verification_result"
        ] in card
        assert f"on {governance['approved_on']}." in card
        assert all(artifact["filename"] in card for artifact in model["artifacts"])
        assert "license_copyright" not in catalog["models"][model_id]
        assert "additional_mit_notices" not in catalog["models"][model_id]

        license_bytes = (root / "LICENSE").read_bytes()
        notices = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        sources = governance["models"][model_id]["upstream_sources"]
        assert sources
        for source in sources:
            assert source["license_role"] in {"weights", "notice"}
            assert source["license_url"].startswith(
                f"{source['url']}/blob/{source['revision']}/"
            )
            assert source["license_file"].startswith(
                "model_cards/upstream_licenses/"
            )
            assert source["license_sha256"] == EXPECTED_UPSTREAM_LICENSE_SHA256[
                source["license_url"]
            ]
            source_bytes = (REPO_ROOT / source["license_file"]).read_bytes()
            assert "license_strip_final_newline" not in source
            assert hashlib.sha256(source_bytes).hexdigest() == source["license_sha256"]
            if source["license_role"] == "weights":
                assert license_bytes == source_bytes
                assert source["code_license"] == rights["weights_license"]
            else:
                assert source_bytes.decode("utf-8").rstrip("\n") in notices
            if source["code_license"] == "Apache-2.0":
                assert {"notice_url", "notice_file", "notice_sha256"} <= set(
                    source
                )
                notice_values = [
                    source["notice_url"],
                    source["notice_file"],
                    source["notice_sha256"],
                ]
                assert all(value is None for value in notice_values) or all(
                    value is not None for value in notice_values
                )
                notice_bytes = model_card_renderer._source_notice_bytes(source)
                if notice_bytes is not None:
                    assert notice_bytes.decode("utf-8").rstrip("\n") in notices

    assert set(EXPECTED_UPSTREAM_LICENSE_SHA256) == {
        source["license_url"]
        for record in governance["models"].values()
        for source in record["upstream_sources"]
    }


@pytest.mark.release_blocker
def test_renderer_rejects_an_upstream_license_digest_mismatch():
    governance = _json(REPO_ROOT / "facetorch/models/governance.json")
    source = dict(
        governance["models"]["detector-retinaface"]["upstream_sources"][0]
    )
    source["license_sha256"] = "0" * 64
    with pytest.raises(ModelCardError, match="license digest mismatch"):
        model_card_renderer._source_license_bytes(source)


@pytest.mark.release_blocker
def test_renderer_wraps_a_missing_upstream_license_as_model_card_error():
    governance = _json(REPO_ROOT / "facetorch/models/governance.json")
    source = dict(
        governance["models"]["detector-retinaface"]["upstream_sources"][0]
    )
    source["license_file"] = "model_cards/upstream_licenses/missing-LICENSE"
    with pytest.raises(ModelCardError, match="Cannot read upstream license file"):
        model_card_renderer._source_license_bytes(source)


@pytest.mark.release_blocker
def test_renderer_rejects_an_upstream_license_url_from_another_source():
    governance = _json(REPO_ROOT / "facetorch/models/governance.json")
    source = dict(
        governance["models"]["detector-retinaface"]["upstream_sources"][0]
    )
    source["license_url"] = source["license_url"].replace(
        "biubug6/Pytorch_Retinaface", "someone/else"
    )
    with pytest.raises(ModelCardError, match="source and revision"):
        model_card_renderer._source_license_bytes(source)


@pytest.mark.release_blocker
def test_renderer_requires_explicit_apache_notice_state():
    governance = _json(REPO_ROOT / "facetorch/models/governance.json")
    source = dict(governance["models"]["au-opengraph"]["upstream_sources"][0])
    source.pop("notice_url")
    with pytest.raises(ModelCardError, match="NOTICE state"):
        model_card_renderer._source_notice_bytes(source)


@pytest.mark.release_blocker
def test_renderer_wraps_a_missing_upstream_notice_as_model_card_error():
    governance = _json(REPO_ROOT / "facetorch/models/governance.json")
    source = dict(governance["models"]["au-opengraph"]["upstream_sources"][0])
    source.update(
        notice_url=(
            f"{source['url']}/blob/{source['revision']}/NOTICE"
        ),
        notice_file="model_cards/upstream_notices/missing-NOTICE",
        notice_sha256="0" * 64,
    )
    with pytest.raises(ModelCardError, match="Cannot read upstream NOTICE file"):
        model_card_renderer._source_notice_bytes(source)


@pytest.mark.release_blocker
def test_renderer_rejects_unapproved_governance(tmp_path, monkeypatch):
    governance = _json(REPO_ROOT / "facetorch/models/governance.json")
    governance["status"] = "pending"
    path = tmp_path / "governance.json"
    _write_json(path, governance)
    monkeypatch.setattr(model_card_renderer, "GOVERNANCE_PATH", path)
    with pytest.raises(ModelCardError, match="approved governance"):
        render_model_documents()


@pytest.mark.release_blocker
@pytest.mark.parametrize(
    ("field_path", "invalid_value", "message"),
    [
        (("license_policy", "status"), "pending", "approved license policy"),
        (
            ("models", "detector-retinaface", "status"),
            "pending",
            "governance status must be approved",
        ),
        (
            ("models", "detector-retinaface", "release_eligible"),
            False,
            "release_eligible must be true",
        ),
        (
            (
                "models",
                "detector-retinaface",
                "rights",
                "redistribution",
            ),
            "pending",
            "rights.redistribution must be approved",
        ),
        (
            ("models", "detector-retinaface", "rights", "attribution"),
            "pending",
            "rights.attribution must be approved",
        ),
        (
            (
                "models",
                "detector-retinaface",
                "rights",
                "owner_approval",
            ),
            "pending",
            "rights.owner_approval must be approved",
        ),
        (
            (
                "models",
                "detector-retinaface",
                "source_checkpoint",
                "upstream_checkpoint_mapping",
            ),
            "unverified",
            "checkpoint mapping must be verified",
        ),
        (
            (
                "models",
                "detector-retinaface",
                "source_checkpoint",
                "hosted_sha256_verified",
            ),
            False,
            "hosted SHA-256 verification must be true",
        ),
    ],
)
def test_renderer_rejects_contradictory_governance_approval_values(
    field_path,
    invalid_value,
    message,
    tmp_path,
    monkeypatch,
):
    governance = _json(REPO_ROOT / "facetorch/models/governance.json")
    target = governance
    for name in field_path[:-1]:
        target = target[name]
    target[field_path[-1]] = invalid_value
    path = tmp_path / "governance.json"
    _write_json(path, governance)
    monkeypatch.setattr(model_card_renderer, "GOVERNANCE_PATH", path)
    with pytest.raises(ModelCardError, match=message):
        render_model_documents()


@pytest.mark.release_blocker
def test_renderer_wraps_invalid_utf8_contract_as_model_card_error(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "governance.json"
    path.write_bytes(b"\xff")
    monkeypatch.setattr(model_card_renderer, "GOVERNANCE_PATH", path)
    with pytest.raises(ModelCardError, match="Cannot read JSON contract document"):
        render_model_documents()


@pytest.mark.release_blocker
def test_renderer_rejects_a_manifest_model_missing_from_the_catalog(
    tmp_path,
    monkeypatch,
):
    catalog = _json(REPO_ROOT / "model_cards/catalog.json")
    catalog["models"].pop("detector-retinaface")
    path = tmp_path / "catalog.json"
    _write_json(path, catalog)
    monkeypatch.setattr(model_card_renderer, "CATALOG_PATH", path)
    with pytest.raises(ModelCardError, match="do not cover"):
        render_model_documents()


@pytest.mark.release_blocker
def test_renderer_wraps_missing_checkpoint_artifacts_as_model_card_error(
    tmp_path,
    monkeypatch,
):
    governance = _json(REPO_ROOT / "facetorch/models/governance.json")
    governance["models"]["detector-retinaface"]["source_checkpoint"].pop(
        "upstream_artifacts"
    )
    path = tmp_path / "governance.json"
    _write_json(path, governance)
    monkeypatch.setattr(model_card_renderer, "GOVERNANCE_PATH", path)
    with pytest.raises(
        ModelCardError,
        match="source_checkpoint.*upstream_artifacts",
    ):
        render_model_documents()


@pytest.mark.release_blocker
@pytest.mark.parametrize(
    "field_path",
    [
        ("rights",),
        ("rights", "weights_license"),
        ("upstream_sources",),
        ("source_checkpoint",),
        ("intended_use",),
        ("limitations",),
    ],
)
def test_renderer_rejects_missing_required_model_governance_fields(
    field_path,
    tmp_path,
    monkeypatch,
):
    governance = _json(REPO_ROOT / "facetorch/models/governance.json")
    target = governance["models"]["detector-retinaface"]
    for name in field_path[:-1]:
        target = target[name]
    target.pop(field_path[-1])
    path = tmp_path / "governance.json"
    _write_json(path, governance)
    monkeypatch.setattr(model_card_renderer, "GOVERNANCE_PATH", path)
    with pytest.raises(ModelCardError, match="detector-retinaface"):
        render_model_documents()


@pytest.mark.release_blocker
def test_corrected_cards_preserve_the_non_generic_model_contracts(tmp_path):
    render_model_cards(tmp_path)
    embed = (tmp_path / "embed-resnet50/README.md").read_text(encoding="utf-8")
    assert "normalized 128-dimensional representation" in embed
    assert "3,000 projection logits" in embed
    assert "it is not a 2,048-dimensional SENet embedding" in embed

    b0 = (tmp_path / "fer-efficientnet-b0/README.md").read_text(encoding="utf-8")
    b2 = (tmp_path / "fer-efficientnet-b2/README.md").read_text(encoding="utf-8")
    assert "license: apache-2.0" in b0
    assert "license: apache-2.0" in b2
    assert "EmotiEffLib" in b0 and "EmotiEffLib" in b2

    va = (tmp_path / "va-elim/README.md").read_text(encoding="utf-8")
    assert "ELIM_FER" in va
    assert "enc2.t7" in va and "reg2.t7" in va and "header2.t7" in va

    magface = (tmp_path / "verify-magface/README.md").read_text(encoding="utf-8")
    assert "license: apache-2.0" in magface
    assert "UNPG" in magface
    assert "founder_attested_chain_of_custody" in magface


@pytest.mark.upstream_network
@pytest.mark.skipif(
    os.environ.get("FACETORCH_RUN_UPSTREAM_NETWORK") != "1",
    reason="set FACETORCH_RUN_UPSTREAM_NETWORK=1 for pinned upstream checks",
)
def test_pinned_upstream_license_bytes_and_apache_notice_state():
    governance = _json(REPO_ROOT / "facetorch/models/governance.json")
    headers = {"User-Agent": "facetorch-release-audit"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"

    seen = set()
    for record in governance["models"].values():
        for source in record["upstream_sources"]:
            identity = (source["url"], source["revision"], source["license_url"])
            if identity in seen:
                continue
            seen.add(identity)
            raw_url = source["license_url"].replace(
                "https://github.com/", "https://raw.githubusercontent.com/", 1
            ).replace("/blob/", "/", 1)
            with urlopen(Request(raw_url, headers=headers), timeout=30) as response:
                license_bytes = response.read()
            assert hashlib.sha256(license_bytes).hexdigest() == source[
                "license_sha256"
            ]

            if source["code_license"] != "Apache-2.0":
                continue
            if source["notice_url"] is not None:
                notice_url = source["notice_url"].replace(
                    "https://github.com/", "https://raw.githubusercontent.com/", 1
                ).replace("/blob/", "/", 1)
                with urlopen(
                    Request(notice_url, headers=headers), timeout=30
                ) as response:
                    notice_bytes = response.read()
                assert hashlib.sha256(notice_bytes).hexdigest() == source[
                    "notice_sha256"
                ]
                continue

            parsed = urlparse(source["url"])
            owner, repo = parsed.path.strip("/").split("/", 1)
            tree_url = (
                f"https://api.github.com/repos/{owner}/{repo}/git/trees/"
                f"{source['revision']}?recursive=1"
            )
            with urlopen(Request(tree_url, headers=headers), timeout=30) as response:
                tree = json.load(response)
            assert tree.get("truncated") is False
            notice_paths = [
                item["path"]
                for item in tree.get("tree", [])
                if Path(item["path"]).name.lower()
                in {"notice", "notice.txt", "notice.md"}
            ]
            assert notice_paths == [], source["url"]
