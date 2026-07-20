from __future__ import annotations

import ast
import json
from pathlib import Path

import yaml

from scripts.make_paper12_colab_notebooks import (
    existing_notebook_matches,
    has_execution_artifacts,
)


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


def test_paper12_second_backbone_config_contract():
    cfg = yaml.safe_load(
        (CONFIG_DIR / "eurosat_second_backbone.yaml").read_text(encoding="utf-8")
    )

    assert cfg["experiment"]["name"] == "eurosat_second_backbone"
    assert cfg["experiment"]["dataset"] == "eurosat"
    assert cfg["experiment"]["dataset_root"] == "./data/eurosat"
    assert cfg["experiment"]["epochs"] == 50
    assert cfg["experiment"]["batch_size"] == 64
    assert cfg["experiment"]["seeds"] == [42, 123, 456]
    assert cfg["experiment"]["allow_synthetic_fallback"] is False

    assert cfg["backbone"] == {
        "name": "satmae_vit_base",
        "family": "satmae",
        "pretrained": True,
        "checkpoint": "data/weights/satmae/satmae_vit_base.pth",
        "input_channels": 10,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "patch_size": 16,
    }
    assert cfg["modalities"] == [{"preset": "s2_full"}, {"preset": "rgb"}]
    assert [method["name"] for method in cfg["methods"]] == [
        "satmae_linear_probe",
        "satmae_lora_split_qkv_r8",
        "satmae_houlsby_d64",
    ]

    matrix_size = (
        len(cfg["modalities"])
        * len(cfg["methods"])
        * len(cfg["experiment"]["seeds"])
    )
    assert matrix_size == 18


def test_paper12_second_backbone_eurosat_colab_notebook_contract():
    path = COLAB_DIR / "paper12_second_backbone_eurosat_colab.ipynb"
    text = read_notebook_text(path)

    assert f"blob/{PAPER12_RESULTS_BRANCH}/colab/paper12_second_backbone_eurosat_colab.ipynb" in text
    assert f"--branch {PAPER12_RESULTS_BRANCH}" in text
    assert "git rev-parse --abbrev-ref HEAD" in text
    assert "Colab Pro L4" in text
    assert "/content/AlphaEarth-System/data/eurosat" in text
    assert "/content/drive/MyDrive/paper12_results" in text
    assert "/content/second_backbone_eurosat_runs" in text
    assert "scripts/download_public_datasets.py --dataset eurosat" in text
    assert "eurosat_second_backbone.yaml" in text
    assert "second_backbone_eurosat.json" in text
    assert "second_backbone_eurosat_summary.json" in text
    assert "satmae_vit_base.pth" in text
    assert "python -m geoadapter.bench.run_benchmark" in text
    assert "python -m geoadapter.bench.second_backbone_summary" in text
    assert "expected_rows = 18" in text


def test_paper12_geovlm_prompt_segmentation_colab_contract():
    path = COLAB_DIR / "paper12_geovlm_prompt_segmentation_colab.ipynb"
    text = read_notebook_text(path)

    assert "blob/master/colab/paper12_geovlm_prompt_segmentation_colab.ipynb" in text
    assert "pip install -q -e '.[geovlm]' torchgeo" in text
    assert "google/siglip-base-patch16-224" in text
    assert "Prithvi_100M.pt" in text
    assert "geovlm_prompt_segmentation.yaml" in text
    assert "--stage seed42" in text
    assert "--stage full" in text
    assert "geovlm_prompt_segmentation.json" in text
    assert "geovlm_prompt_segmentation_summary.json" in text
    assert "mvp_status" in text
    assert "expected method/seed pairs = 6" in text
    assert (
        "/content/drive/MyDrive/paper12_checkpoints/geovlm_prompt_segmentation"
        in text
    )


def test_notebook_generator_preserves_newlines_and_execution_artifacts():
    rendered = json.dumps({"cells": []}, indent=1)
    assert existing_notebook_matches(rendered + "\n", rendered) is True
    executed = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "outputs": [{"output_type": "stream", "text": ["done\n"]}],
            }
        ]
    }
    assert has_execution_artifacts(json.dumps(executed)) is True


def test_paper12_geovlm_prompt_segmentation_code_cells_are_valid_python():
    path = COLAB_DIR / "paper12_geovlm_prompt_segmentation_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    first_markdown = "".join(notebook["cells"][0]["source"])
    assert "\n# Paper 12 GeoVLM Prompt Segmentation MVP\n" in first_markdown
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        sanitized = []
        for line in "".join(cell["source"]).splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("!", "%")):
                indentation = line[: len(line) - len(stripped)]
                sanitized.append(indentation + "pass")
            else:
                sanitized.append(line)
        ast.parse("\n".join(sanitized), filename=f"cell-{index}")
