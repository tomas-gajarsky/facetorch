import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.render_model_cards import render_model_cards, render_model_documents


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
        assert all(artifact["filename"] in card for artifact in model["artifacts"])
        assert "license_copyright" not in catalog["models"][model_id]
        assert "additional_mit_notices" not in catalog["models"][model_id]

        license_bytes = (root / "LICENSE").read_bytes()
        notices = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        sources = governance["models"][model_id]["upstream_sources"]
        assert sources
        for source in sources:
            assert source["license_role"] in {"weights", "notice"}
            assert source["revision"] in source["license_url"]
            assert source["license_sha256"] == EXPECTED_UPSTREAM_LICENSE_SHA256[
                source["license_url"]
            ]
            source_bytes = (REPO_ROOT / source["license_file"]).read_bytes()
            if source.get("license_strip_final_newline"):
                assert source_bytes.endswith(b"\n")
                source_bytes = source_bytes[:-1]
            assert hashlib.sha256(source_bytes).hexdigest() == source["license_sha256"]
            if source["license_role"] == "weights":
                assert license_bytes == source_bytes
                assert source["code_license"] == rights["weights_license"]
            else:
                assert source_bytes.decode("utf-8").rstrip("\n") in notices

    assert set(EXPECTED_UPSTREAM_LICENSE_SHA256) == {
        source["license_url"]
        for record in governance["models"].values()
        for source in record["upstream_sources"]
    }


@pytest.mark.release_blocker
def test_corrected_cards_preserve_the_non_generic_model_contracts(tmp_path):
    render_model_cards(tmp_path)
    embed = (tmp_path / "embed-resnet50/README.md").read_text(encoding="utf-8")
    assert "normalized 128-dimensional representation" in embed
    assert "3,000 projection logits" in embed
    assert "2,048-dimensional SENet" in embed

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
