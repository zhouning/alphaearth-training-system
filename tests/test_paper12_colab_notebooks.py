from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COLAB_DIR = REPO_ROOT / "colab"


def read_notebook_text(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )


def test_paper12_loveda_full_finetune_colab_notebook_contract():
    path = COLAB_DIR / "paper12_loveda_full_finetune_colab.ipynb"
    text = read_notebook_text(path)

    assert "Colab Pro+ A100 40GB" in text
    assert "/content/AlphaEarth-System/data/weights/raw_data/loveda" in text
    assert "/content/drive/MyDrive/paper12_results" in text
    assert "/content/loveda_full_finetune_runs" in text
    assert "scripts/download_public_datasets.py --dataset loveda" in text
    assert "loveda_lulc_full_finetune_u2r.yaml" in text
    assert "loveda_lulc_full_finetune_r2u.yaml" in text
    assert "python -m geoadapter.bench.run_benchmark" in text
    assert "expected_rows = 3" in text


def test_paper12_eurosat_channel_bridge_colab_notebook_contract():
    path = COLAB_DIR / "paper12_eurosat_channel_bridge_colab.ipynb"
    text = read_notebook_text(path)

    assert "Colab Pro L4" in text
    assert "/content/AlphaEarth-System/data/eurosat" in text
    assert "/content/drive/MyDrive/paper12_results" in text
    assert "/content/eurosat_channel_bridge_runs" in text
    assert "scripts/download_public_datasets.py --dataset eurosat" in text
    assert "eurosat_channel_bridge.yaml" in text
    assert "python -m geoadapter.bench.run_benchmark" in text
    assert "expected_rows = 12" in text
    assert "learned_bridge_houlsby" in text
