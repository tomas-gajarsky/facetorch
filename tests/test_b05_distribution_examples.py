import ast
import json
from pathlib import Path
import re

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.release_blocker
def test_all_readme_python_blocks_are_valid_source():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\s*\n(.*?)```", readme, flags=re.DOTALL)

    assert blocks
    for block in blocks:
        ast.parse(block)


@pytest.mark.release_blocker
def test_notebook_is_clean_and_uses_candidate_public_contract():
    notebook_path = REPO_ROOT / "notebooks" / "facetorch_notebook_demo.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    combined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert notebook["nbformat"] == 4
    assert code_cells
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(cell.get("outputs") == [] for cell in code_cells)
    for cell in code_cells:
        ast.parse("".join(cell.get("source", [])))

    assert "facetorch==1.0.0rc3" in combined
    assert "v1.0.0-rc.3" in combined
    assert "v1.0.0rc3" not in combined
    assert "/main/" not in combined
    assert "load_config(" in combined
    assert "include_tensors=True" in combined
    assert "face_batch_size=" in combined
    assert "result.image" in combined
    assert "OmegaConf.load" not in combined
    assert "return_img_data" not in combined
    assert "response.img" not in combined
