from __future__ import annotations

import ast
import json
import shutil
import sys
import types
from pathlib import Path

import pytest
import yaml

from scripts.make_paper12_colab_notebooks import (
    existing_notebook_matches,
    has_execution_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
COLAB_DIR = REPO_ROOT / "colab"
CONFIG_DIR = REPO_ROOT / "geoadapter" / "bench" / "configs"
PAPER12_RESULTS_BRANCH = "paper12-results-colab-20260619"
GEOVLM_ARCHIVE_CELL_MARKER = (
    "# Archive the failed seed-42 artifacts once before running the v2 recovery."
)
GEOVLM_RESULTS_SCHEMA_V2 = "paper12.geovlm_prompt_results.v2"
GEOVLM_TRAINING_CONTRACT_V2 = "paper12.geovlm_prompt_training.v2"
GEOVLM_LEGACY_CHECKPOINT = "siglip_film_dense_similarity_houlsby__seed42.pt"
GEOVLM_PROMPT_METHOD = "siglip_film_dense_similarity_houlsby"
GEOVLM_SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
GEOVLM_SIGLIP_REVISION = "siglip-revision-a"
GEOVLM_REQUIRED_CLASSES = ("building", "road", "water")
GEOVLM_SIGLIP_CELL_MARKER = (
    "# 6. Pre-cache the frozen SigLIP text tower and record its resolved revision."
)


def read_notebook_text(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )


def geovlm_archive_cell_source(*, archive_failed_run: bool = False) -> str:
    path = COLAB_DIR / "paper12_geovlm_prompt_segmentation_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
        and GEOVLM_ARCHIVE_CELL_MARKER in "".join(cell.get("source", []))
    ]
    assert len(matches) == 1
    source = matches[0]
    if archive_failed_run:
        assignment = "ARCHIVE_FAILED_RUN = False"
        assert source.count(assignment) == 1
        source = source.replace(assignment, "ARCHIVE_FAILED_RUN = True", 1)
    return source


def geovlm_siglip_cell_source() -> str:
    path = COLAB_DIR / "paper12_geovlm_prompt_segmentation_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
        and GEOVLM_SIGLIP_CELL_MARKER in "".join(cell.get("source", []))
    ]
    assert len(matches) == 1
    return matches[0]


def geovlm_archive_namespace(tmp_path: Path) -> dict[str, object]:
    drive_results_dir = tmp_path / "drive_results"
    checkpoint_dir = tmp_path / "checkpoints"
    preview_dir = tmp_path / "previews"
    local_results_dir = tmp_path / "local_results"
    for path in (
        drive_results_dir,
        checkpoint_dir,
        preview_dir,
        local_results_dir,
    ):
        path.mkdir()

    raw_json = local_results_dir / "geovlm_prompt_segmentation.json"
    summary_json = local_results_dir / "geovlm_prompt_segmentation_summary.json"
    return {
        "DRIVE_RESULTS_DIR": drive_results_dir,
        "CHECKPOINT_DIR": checkpoint_dir,
        "PREVIEW_DIR": preview_dir,
        "RAW_JSON": raw_json,
        "SUMMARY_JSON": summary_json,
        "DRIVE_RAW_JSON": drive_results_dir / raw_json.name,
        "DRIVE_SUMMARY_JSON": drive_results_dir / summary_json.name,
        "SIGLIP_REVISION_PIN": tmp_path / "resolved_revision.txt",
        "shutil": shutil,
    }


def geovlm_complete_v2_rows() -> list[dict[str, object]]:
    return [
        {
            "training_contract": GEOVLM_TRAINING_CONTRACT_V2,
            "method": GEOVLM_PROMPT_METHOD,
            "seed": 42,
            "class_name": class_name,
            "siglip_model_id": GEOVLM_SIGLIP_MODEL_ID,
            "siglip_revision": GEOVLM_SIGLIP_REVISION,
        }
        for class_name in GEOVLM_REQUIRED_CLASSES
    ]


def geovlm_v2_payload_bytes(rows: list[dict[str, object]]) -> bytes:
    return (
        json.dumps(
            {
                "schema": GEOVLM_RESULTS_SCHEMA_V2,
                "training_contract": GEOVLM_TRAINING_CONTRACT_V2,
                "rows": rows,
            },
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


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
    assert 'config["text_encoder"]["cache_dir"] = str(HF_CACHE_DIR)' in text
    assert 'SIGLIP_REVISION_PIN = HF_CACHE_DIR / "resolved_revision.txt"' in text
    assert (
        'DRIVE_CONFIG_COLAB = DRIVE_RESULTS_DIR / "geovlm_prompt_segmentation_colab.yaml"'
        in text
    )
    assert "shutil.copy2(CONFIG_COLAB, DRIVE_CONFIG_COLAB)" in text
    assert "paper12.geovlm_prompt_training.v2" in text
    assert "ARCHIVE_FAILED_RUN = False" in text
    assert "failed_seed42_20260724" in text
    assert "archive it before recovery" in text
    assert "siglip_film_dense_similarity_houlsby__seed42.best.pt" in text
    assert "siglip_film_dense_similarity_houlsby__seed42.pt" in text


def test_geovlm_focused_commands_include_training_contract_tests():
    notebook_text = read_notebook_text(
        COLAB_DIR / "paper12_geovlm_prompt_segmentation_colab.ipynb"
    )
    docs_text = (
        REPO_ROOT / "docs" / "geovlm_prompt_segmentation_mvp.md"
    ).read_text(encoding="utf-8")

    assert "tests/test_geovlm_training.py" in notebook_text
    assert "tests/test_geovlm_training.py" in docs_text


def test_geovlm_siglip_cell_pins_first_revision_and_reuses_it_after_restart(
    tmp_path, monkeypatch
):
    pin = tmp_path / "resolved_revision.txt"
    cache = tmp_path / "cache"
    cache.mkdir()
    model_info_calls = []
    snapshot_calls = []

    def model_info(model_id):
        model_info_calls.append(model_id)
        return types.SimpleNamespace(sha="first-resolved-sha")

    def snapshot_download(**kwargs):
        snapshot_calls.append(kwargs)
        return str(cache / kwargs["revision"])

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            model_info=model_info,
            snapshot_download=snapshot_download,
        ),
    )
    first_namespace = {
        "HF_CACHE_DIR": cache,
        "SIGLIP_REVISION_PIN": pin,
    }
    exec(geovlm_siglip_cell_source(), first_namespace)

    assert pin.read_text(encoding="utf-8") == "first-resolved-sha\n"
    assert model_info_calls == [GEOVLM_SIGLIP_MODEL_ID]
    assert snapshot_calls[-1]["revision"] == "first-resolved-sha"

    def changed_model_info(_model_id):
        raise AssertionError("Hub HEAD must not be queried when a revision pin exists")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            model_info=changed_model_info,
            snapshot_download=snapshot_download,
        ),
    )
    restarted_namespace = {
        "HF_CACHE_DIR": cache,
        "SIGLIP_REVISION_PIN": pin,
    }
    exec(geovlm_siglip_cell_source(), restarted_namespace)

    assert restarted_namespace["SIGLIP_REVISION"] == "first-resolved-sha"
    assert snapshot_calls[-1]["revision"] == "first-resolved-sha"


def test_geovlm_siglip_cell_rejects_empty_revision_pin(tmp_path, monkeypatch):
    pin = tmp_path / "resolved_revision.txt"
    pin.write_text(" \n", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            model_info=lambda _model_id: types.SimpleNamespace(sha="new-sha"),
            snapshot_download=lambda **_kwargs: "unused",
        ),
    )

    with pytest.raises(ValueError, match="revision pin.*non-empty"):
        exec(
            geovlm_siglip_cell_source(),
            {
                "HF_CACHE_DIR": tmp_path / "cache",
                "SIGLIP_REVISION_PIN": pin,
            },
        )


def test_geovlm_archive_cell_resumes_compatible_v2_results(tmp_path):
    namespace = geovlm_archive_namespace(tmp_path)
    namespace["SIGLIP_REVISION_PIN"].write_text(
        GEOVLM_SIGLIP_REVISION + "\n", encoding="utf-8"
    )
    raw_bytes = geovlm_v2_payload_bytes(geovlm_complete_v2_rows())
    summary_bytes = b'{"mvp_status": "incomplete"}\n'
    preview_bytes = b"current preview"
    namespace["DRIVE_RAW_JSON"].write_bytes(raw_bytes)
    namespace["DRIVE_SUMMARY_JSON"].write_bytes(summary_bytes)
    preview = namespace["PREVIEW_DIR"] / "seed42__water.png"
    preview.write_bytes(preview_bytes)
    failed_archive = namespace["DRIVE_RESULTS_DIR"] / "failed_seed42_20260724"
    failed_archive.mkdir()
    archive_sentinel = failed_archive / "archived.txt"
    archive_sentinel.write_bytes(b"completed archive")

    exec(geovlm_archive_cell_source(), namespace)

    assert namespace["RAW_JSON"].read_bytes() == raw_bytes
    assert namespace["SUMMARY_JSON"].read_bytes() == summary_bytes
    assert preview.read_bytes() == preview_bytes
    assert archive_sentinel.read_bytes() == b"completed archive"


def test_geovlm_archive_cell_resumes_empty_v2_results_without_revision_pin(
    tmp_path,
):
    namespace = geovlm_archive_namespace(tmp_path)
    raw_bytes = geovlm_v2_payload_bytes([])
    namespace["DRIVE_RAW_JSON"].write_bytes(raw_bytes)

    exec(geovlm_archive_cell_source(), namespace)

    assert namespace["DRIVE_RAW_JSON"].read_bytes() == raw_bytes
    assert namespace["RAW_JSON"].read_bytes() == raw_bytes
    assert not namespace["SIGLIP_REVISION_PIN"].exists()
    assert not any(
        path.name.startswith("failed_seed42_")
        for path in namespace["DRIVE_RESULTS_DIR"].iterdir()
        if path.is_dir()
    )


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param(geovlm_complete_v2_rows()[:2], id="partial-pair"),
        pytest.param(
            geovlm_complete_v2_rows() + [geovlm_complete_v2_rows()[0]],
            id="duplicate-identity",
        ),
        pytest.param(
            [
                {**row, "method": "unsupported_method"}
                for row in geovlm_complete_v2_rows()
            ],
            id="invalid-method",
        ),
        pytest.param(
            [{**row, "seed": True} for row in geovlm_complete_v2_rows()],
            id="bool-seed",
        ),
        pytest.param(
            [{**row, "seed": 999} for row in geovlm_complete_v2_rows()],
            id="invalid-seed",
        ),
        pytest.param(
            [
                {**row, "class_name": "woodland"}
                for row in geovlm_complete_v2_rows()
            ],
            id="invalid-class",
        ),
        pytest.param(
            [
                {
                    **row,
                    "siglip_revision": (
                        "siglip-revision-b"
                        if row["class_name"] == "water"
                        else row["siglip_revision"]
                    ),
                }
                for row in geovlm_complete_v2_rows()
            ],
            id="mixed-siglip-revision",
        ),
    ],
)
def test_geovlm_archive_cell_archives_v2_rows_with_invalid_identity(
    tmp_path, rows
):
    namespace = geovlm_archive_namespace(tmp_path)
    raw_bytes = geovlm_v2_payload_bytes(rows)
    drive_raw = namespace["DRIVE_RAW_JSON"]
    drive_raw.write_bytes(raw_bytes)

    with pytest.raises(RuntimeError, match="archive it before recovery"):
        exec(geovlm_archive_cell_source(), namespace)

    assert drive_raw.read_bytes() == raw_bytes
    assert not namespace["RAW_JSON"].exists()

    exec(geovlm_archive_cell_source(archive_failed_run=True), namespace)

    failed_archive = namespace["DRIVE_RESULTS_DIR"] / "failed_seed42_20260724"
    staging_archive = (
        namespace["DRIVE_RESULTS_DIR"] / ".failed_seed42_20260724.incomplete"
    )
    assert not drive_raw.exists()
    assert not namespace["RAW_JSON"].exists()
    assert not staging_archive.exists()
    assert (failed_archive / drive_raw.name).read_bytes() == raw_bytes


def test_geovlm_archive_cell_resumes_interrupted_staging_move(
    tmp_path, monkeypatch
):
    namespace = geovlm_archive_namespace(tmp_path)
    preview = namespace["PREVIEW_DIR"] / "seed42__building.png"
    legacy_checkpoint = namespace["CHECKPOINT_DIR"] / GEOVLM_LEGACY_CHECKPOINT
    source_payloads = {
        namespace["DRIVE_RAW_JSON"]: b'{"schema": "legacy.v1"}\n',
        namespace["DRIVE_SUMMARY_JSON"]: b'{"status": "failed"}\n',
        legacy_checkpoint: b"legacy checkpoint",
        preview: b"failed preview",
    }
    for path, payload in source_payloads.items():
        path.write_bytes(payload)

    real_move = shutil.move
    move_calls = []

    def fail_second_move(source, destination):
        move_calls.append((source, destination))
        if len(move_calls) == 2:
            raise OSError("injected archive move failure")
        return real_move(source, destination)

    monkeypatch.setattr(shutil, "move", fail_second_move)
    with pytest.raises(OSError, match="injected archive move failure"):
        exec(geovlm_archive_cell_source(archive_failed_run=True), namespace)

    failed_archive = namespace["DRIVE_RESULTS_DIR"] / "failed_seed42_20260724"
    staging_archive = (
        namespace["DRIVE_RESULTS_DIR"] / ".failed_seed42_20260724.incomplete"
    )
    assert not failed_archive.exists()
    assert staging_archive.is_dir()
    assert namespace["DRIVE_RAW_JSON"].name in {
        path.name for path in staging_archive.iterdir()
    }
    assert any(path.exists() for path in source_payloads)

    monkeypatch.setattr(shutil, "move", real_move)
    exec(geovlm_archive_cell_source(archive_failed_run=True), namespace)

    assert not staging_archive.exists()
    assert failed_archive.is_dir()
    assert {path.name for path in failed_archive.iterdir()} == {
        path.name for path in source_payloads
    }
    for source, payload in source_payloads.items():
        assert not source.exists()
        assert (failed_archive / source.name).read_bytes() == payload


def test_geovlm_archive_cell_rejects_completed_archive_collision(tmp_path):
    namespace = geovlm_archive_namespace(tmp_path)
    failed_raw = namespace["DRIVE_RAW_JSON"]
    failed_raw.write_bytes(b'{"schema": "legacy.v1"}\n')
    failed_archive = namespace["DRIVE_RESULTS_DIR"] / "failed_seed42_20260724"
    failed_archive.mkdir()
    sentinel = failed_archive / "archived.txt"
    sentinel.write_bytes(b"completed archive")

    with pytest.raises(RuntimeError, match="archive already exists"):
        exec(geovlm_archive_cell_source(archive_failed_run=True), namespace)

    assert failed_raw.read_bytes() == b'{"schema": "legacy.v1"}\n'
    assert sentinel.read_bytes() == b"completed archive"
    assert not (
        namespace["DRIVE_RESULTS_DIR"] / ".failed_seed42_20260724.incomplete"
    ).exists()


def test_geovlm_archive_cell_rejects_source_and_staging_collision(tmp_path):
    namespace = geovlm_archive_namespace(tmp_path)
    failed_raw = namespace["DRIVE_RAW_JSON"]
    failed_raw.write_bytes(b'{"schema": "legacy.v1"}\n')
    staging_archive = (
        namespace["DRIVE_RESULTS_DIR"] / ".failed_seed42_20260724.incomplete"
    )
    staging_archive.mkdir()
    staged_raw = staging_archive / failed_raw.name
    staged_raw.write_bytes(b"previous partial move")

    with pytest.raises(RuntimeError, match="source and staged destination both exist"):
        exec(geovlm_archive_cell_source(archive_failed_run=True), namespace)

    assert failed_raw.read_bytes() == b'{"schema": "legacy.v1"}\n'
    assert staged_raw.read_bytes() == b"previous partial move"
    assert not (
        namespace["DRIVE_RESULTS_DIR"] / "failed_seed42_20260724"
    ).exists()


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
