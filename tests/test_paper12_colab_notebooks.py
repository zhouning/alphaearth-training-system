from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
COLAB_DIR = REPO_ROOT / "colab"
CONFIG_DIR = REPO_ROOT / "geoadapter" / "bench" / "configs"
PAPER12_RESULTS_BRANCH = "paper12-results-colab-20260619"


def read_notebook_text(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )


def test_paper12_loveda_full_finetune_colab_notebook_contract():
    path = COLAB_DIR / "paper12_loveda_full_finetune_colab.ipynb"
    text = read_notebook_text(path)

    assert f"blob/{PAPER12_RESULTS_BRANCH}/colab/paper12_loveda_full_finetune_colab.ipynb" in text
    assert f"--branch {PAPER12_RESULTS_BRANCH}" in text
    assert "git rev-parse --abbrev-ref HEAD" in text
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

    assert f"blob/{PAPER12_RESULTS_BRANCH}/colab/paper12_eurosat_channel_bridge_colab.ipynb" in text
    assert f"--branch {PAPER12_RESULTS_BRANCH}" in text
    assert "git rev-parse --abbrev-ref HEAD" in text
    assert "Colab Pro L4" in text
    assert "/content/AlphaEarth-System/data/eurosat" in text
    assert "/content/drive/MyDrive/paper12_results" in text
    assert "/content/eurosat_channel_bridge_runs" in text
    assert "scripts/download_public_datasets.py --dataset eurosat" in text
    assert "eurosat_channel_bridge.yaml" in text
    assert "eurosat_channel_bridge_archive_pre_rerun" in text
    assert "eurosat_channel_bridge_summary.json" in text
    assert "python -m geoadapter.bench.run_benchmark" in text
    assert "expected_rows = 12" in text
    assert "learned_bridge_houlsby" in text


def test_paper12_peft_capacity_sweep_colab_notebook_contract():
    path = COLAB_DIR / "paper12_peft_capacity_sweep_colab.ipynb"
    text = read_notebook_text(path)

    assert f"blob/{PAPER12_RESULTS_BRANCH}/colab/paper12_peft_capacity_sweep_colab.ipynb" in text
    assert f"--branch {PAPER12_RESULTS_BRANCH}" in text
    assert "git rev-parse --abbrev-ref HEAD" in text
    assert "Colab Pro L4" in text
    assert "/content/AlphaEarth-System/data/eurosat" in text
    assert "/content/drive/MyDrive/paper12_results" in text
    assert "/content/peft_capacity_sweep_runs" in text
    assert "scripts/download_public_datasets.py --dataset eurosat" in text
    assert "eurosat_peft_capacity_sweep.yaml" in text
    assert "peft_capacity_sweep.json" in text
    assert "peft_capacity_sweep_summary.json" in text
    assert "python -m geoadapter.bench.run_benchmark" in text
    assert "expected_rows = 30" in text
    assert "lora_split_qkv_r64" in text
    assert "houlsby_d64" in text


def test_paper12_peft_capacity_sweep_config_contract():
    cfg = yaml.safe_load(
        (CONFIG_DIR / "eurosat_peft_capacity_sweep.yaml").read_text(encoding="utf-8")
    )

    assert cfg["experiment"]["name"] == "eurosat_peft_capacity_sweep"
    assert cfg["experiment"]["dataset"] == "eurosat"
    assert cfg["experiment"]["dataset_root"] == "./data/eurosat"
    assert cfg["experiment"]["epochs"] == 50
    assert cfg["experiment"]["batch_size"] == 64
    assert cfg["experiment"]["seeds"] == [42, 123, 456]
    assert cfg["modalities"] == [{"preset": "s2_full"}]
    assert cfg["prithvi"]["pretrained"] is True
    assert cfg["prithvi"]["checkpoint"] == "data/weights/prithvi/Prithvi_100M.pt"

    methods = cfg["methods"]
    assert len(methods) == 10
    assert [method["name"] for method in methods] == [
        "linear_probe",
        "lora_split_qkv_r4",
        "lora_split_qkv_r8",
        "lora_split_qkv_r16",
        "lora_split_qkv_r32",
        "lora_split_qkv_r64",
        "houlsby_d8",
        "houlsby_d16",
        "houlsby_d32",
        "houlsby_d64",
    ]
    assert [method.get("rank") for method in methods if method["name"].startswith("lora")] == [
        4,
        8,
        16,
        32,
        64,
    ]
    assert [
        method.get("bottleneck_dim")
        for method in methods
        if method["name"].startswith("houlsby")
    ] == [8, 16, 32, 64]
