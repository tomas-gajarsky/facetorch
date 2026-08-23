import json
from pathlib import Path

import pytest
import yaml

from scripts.render_model_cards import render_model_cards


REPO_ROOT = Path(__file__).resolve().parents[1]


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.release_blocker
def test_model_cards_render_from_the_release_contract(tmp_path):
    manifest = _json(REPO_ROOT / "facetorch/models/manifest.json")
    governance = _json(REPO_ROOT / "facetorch/models/governance.json")
    catalog = _json(REPO_ROOT / "model_cards/catalog.json")
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
        assert "Copyright" in (root / "LICENSE").read_text(encoding="utf-8") or (
            rights["weights_license"] == "Apache-2.0"
        )


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
