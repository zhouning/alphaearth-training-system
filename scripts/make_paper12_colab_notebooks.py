from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLAB_DIR = ROOT / "colab"
LOVE_OUT = COLAB_DIR / "paper12_loveda_full_finetune_colab.ipynb"
EURO_OUT = COLAB_DIR / "paper12_eurosat_channel_bridge_colab.ipynb"
CAPACITY_OUT = COLAB_DIR / "paper12_peft_capacity_sweep_colab.ipynb"
SECOND_BACKBONE_OUT = COLAB_DIR / "paper12_second_backbone_eurosat_colab.ipynb"
LANDCOVER_DECODER_OUT = COLAB_DIR / "paper12_landcoverai_decoder_ablation_colab.ipynb"
GEOVLM_PROMPT_OUT = COLAB_DIR / "paper12_geovlm_prompt_segmentation_colab.ipynb"
PAPER12_RESULTS_BRANCH = "paper12-results-colab-20260619"


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": to_source_lines(source),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": to_source_lines(source),
    }


def dedented_markdown_cell(source: str) -> dict:
    return markdown_cell(textwrap.dedent(source))


def dedented_code_cell(source: str) -> dict:
    return code_cell(textwrap.dedent(source))


def to_source_lines(source: str) -> list[str]:
    lines = source.strip().splitlines()
    if not lines:
        return []
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def notebook(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
        },
        "cells": cells,
    }


def existing_notebook_matches(existing: str, rendered: str) -> bool:
    return existing.rstrip("\r\n") == rendered.rstrip("\r\n")


def has_execution_artifacts(existing: str) -> bool:
    try:
        payload = json.loads(existing)
    except json.JSONDecodeError:
        return False
    return any(
        cell.get("cell_type") == "code"
        and (cell.get("execution_count") is not None or cell.get("outputs"))
        for cell in payload.get("cells", [])
    )


def loveda_notebook() -> dict:
    return notebook(
        [
            markdown_cell(
                """
                <a href="https://colab.research.google.com/github/zhouning/alphaearth-training-system/blob/paper12-results-colab-20260619/colab/paper12_loveda_full_finetune_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

                # Paper 12 LoveDA Full Fine-Tuning Baseline

                This notebook runs the missing LoveDA cross-domain full fine-tuning baseline for Paper 12 on the public LoveDA dataset.

                **Required runtime:** Colab Pro+ A100 40GB. This experiment updates the full Prithvi backbone on 1024x1024 segmentation tiles. L4 is only reasonable for smoke checks after manually reducing batch size.

                **Storage policy:** download LoveDA to Colab local SSD under `/content/AlphaEarth-System/data/weights/raw_data/loveda`, keep checkpoints under `/content/loveda_full_finetune_runs`, and only persist result JSON files to `/content/drive/MyDrive/paper12_results`.

                **Outputs written to Drive:**
                - `loveda_full_finetune_u2r.json`
                - `loveda_full_finetune_r2u.json`
                - `loveda_full_finetune_summary.json`
                """
            ),
            code_cell(
                """
                # 1. Mount Drive and create the results directory.
                from google.colab import drive
                drive.mount("/content/drive")

                import os

                RESULTS_DIR = "/content/drive/MyDrive/paper12_results"
                os.makedirs(RESULTS_DIR, exist_ok=True)
                print("Drive results directory:", RESULTS_DIR)
                """
            ),
            code_cell(
                """
                # 2. GPU, Python, and disk sanity check.
                !nvidia-smi
                !python --version
                !df -h /content
                """
            ),
            code_cell(
                """
                # 3. Clone the Paper 12 results branch into local SSD.
                %cd /content
                !rm -rf /content/AlphaEarth-System
                !git clone --branch paper12-results-colab-20260619 https://github.com/zhouning/alphaearth-training-system.git /content/AlphaEarth-System
                %cd /content/AlphaEarth-System
                !git rev-parse --abbrev-ref HEAD
                !git rev-parse HEAD
                !git log --oneline -3
                """
            ),
            code_cell(
                """
                # 4. Install the local package and notebook-only helpers.
                %cd /content/AlphaEarth-System
                !pip install -q -e . torchgeo pyyaml huggingface_hub
                """
            ),
            code_cell(
                """
                # 5. Stage Prithvi weights at the path the benchmark expects.
                %cd /content/AlphaEarth-System
                import os
                import shutil
                from huggingface_hub import hf_hub_download

                DRIVE_WEIGHTS = "/content/drive/MyDrive/Prithvi_100M.pt"
                LOCAL_WEIGHTS = "/content/AlphaEarth-System/data/weights/prithvi/Prithvi_100M.pt"
                os.makedirs(os.path.dirname(LOCAL_WEIGHTS), exist_ok=True)

                if os.path.exists(DRIVE_WEIGHTS):
                    shutil.copy(DRIVE_WEIGHTS, LOCAL_WEIGHTS)
                    print("Copied Prithvi weights from Drive.")
                elif not os.path.exists(LOCAL_WEIGHTS):
                    downloaded = hf_hub_download(
                        repo_id="ibm-nasa-geospatial/Prithvi-100M",
                        filename="Prithvi_100M.pt",
                    )
                    shutil.copy(downloaded, LOCAL_WEIGHTS)
                    print("Downloaded Prithvi weights from Hugging Face.")
                else:
                    print("Prithvi weights already present locally.")

                !ls -lh /content/AlphaEarth-System/data/weights/prithvi
                """
            ),
            code_cell(
                """
                # 6. Download the public LoveDA cache into local SSD and smoke one sample per split.
                %cd /content/AlphaEarth-System
                LOVEDA_ROOT = "/content/AlphaEarth-System/data/weights/raw_data/loveda"
                !python scripts/download_public_datasets.py --dataset loveda --loveda-root data/weights/raw_data/loveda --max-samples 1
                !du -sh /content/AlphaEarth-System/data/weights/raw_data/loveda
                """
            ),
            code_cell(
                """
                # 7. Dry-run the experiment matrix before launching the full training jobs.
                %cd /content/AlphaEarth-System
                !python -m geoadapter.bench.run_benchmark --config geoadapter/bench/configs/loveda_lulc_full_finetune_u2r.yaml --dry-run
                !python -m geoadapter.bench.run_benchmark --config geoadapter/bench/configs/loveda_lulc_full_finetune_r2u.yaml --dry-run
                """
            ),
            code_cell(
                """
                # 8. Run the U->R full fine-tuning baseline. Checkpoints stay on local SSD.
                %cd /content/AlphaEarth-System
                !mkdir -p /content/loveda_full_finetune_runs/u2r
                !python -m geoadapter.bench.run_benchmark --config geoadapter/bench/configs/loveda_lulc_full_finetune_u2r.yaml --output /content/drive/MyDrive/paper12_results/loveda_full_finetune_u2r.json --checkpoint-dir /content/loveda_full_finetune_runs/u2r --checkpoint-every 5
                """
            ),
            code_cell(
                """
                # 9. Run the R->U full fine-tuning baseline. Checkpoints stay on local SSD.
                %cd /content/AlphaEarth-System
                !mkdir -p /content/loveda_full_finetune_runs/r2u
                !python -m geoadapter.bench.run_benchmark --config geoadapter/bench/configs/loveda_lulc_full_finetune_r2u.yaml --output /content/drive/MyDrive/paper12_results/loveda_full_finetune_r2u.json --checkpoint-dir /content/loveda_full_finetune_runs/r2u --checkpoint-every 5
                """
            ),
            code_cell(
                """
                # 10. Verify result counts, print per-seed mIoU, and persist a compact summary JSON to Drive.
                import json
                from pathlib import Path
                from statistics import mean, stdev

                results_dir = Path("/content/drive/MyDrive/paper12_results")
                u2r_path = results_dir / "loveda_full_finetune_u2r.json"
                r2u_path = results_dir / "loveda_full_finetune_r2u.json"
                summary_path = results_dir / "loveda_full_finetune_summary.json"

                u2r = json.loads(u2r_path.read_text(encoding="utf-8"))
                r2u = json.loads(r2u_path.read_text(encoding="utf-8"))
                expected_rows = 3
                assert len(u2r) == expected_rows, f"expected {expected_rows} U->R rows, got {len(u2r)}"
                assert len(r2u) == expected_rows, f"expected {expected_rows} R->U rows, got {len(r2u)}"

                def metric_summary(rows, metric):
                    values = [float(row[metric]) for row in rows]
                    return {
                        "mean": mean(values),
                        "std": stdev(values) if len(values) > 1 else 0.0,
                        "values": values,
                    }

                for direction, rows in (("U->R", u2r), ("R->U", r2u)):
                    print(direction)
                    for row in rows:
                        print(
                            {
                                "method": row["method"],
                                "seed": row["seed"],
                                "mIoU": round(float(row["mIoU"]), 4),
                            }
                        )

                summary = {
                    "u2r": metric_summary(u2r, "mIoU"),
                    "r2u": metric_summary(r2u, "mIoU"),
                }
                summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print("Wrote", summary_path)
                print(json.dumps(summary, indent=2))
                """
            ),
        ]
    )


def eurosat_notebook() -> dict:
    return notebook(
        [
            markdown_cell(
                """
                <a href="https://colab.research.google.com/github/zhouning/alphaearth-training-system/blob/paper12-results-colab-20260619/colab/paper12_eurosat_channel_bridge_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

                # Paper 12 EuroSAT Channel-Bridge Ablation

                This notebook runs the public EuroSAT ablation that compares deterministic zero padding against the learned 10->6 channel bridge for Prithvi adaptation.

                **Required runtime:** Colab Pro L4. A100 is optional. T4 is acceptable only for a short smoke run, not for the full 4-method x 3-seed x 50-epoch matrix.

                **Storage policy:** download EuroSAT to Colab local SSD under `/content/AlphaEarth-System/data/eurosat`, keep checkpoints under `/content/eurosat_channel_bridge_runs`, and only persist result JSON files to `/content/drive/MyDrive/paper12_results`.

                **Methods in this run:** `zero_pad_linear_probe`, `learned_bridge_linear_probe`, `zero_pad_houlsby`, `learned_bridge_houlsby`.
                """
            ),
            code_cell(
                """
                # 1. Mount Drive and create the results directory.
                from google.colab import drive
                drive.mount("/content/drive")

                import os

                RESULTS_DIR = "/content/drive/MyDrive/paper12_results"
                os.makedirs(RESULTS_DIR, exist_ok=True)
                print("Drive results directory:", RESULTS_DIR)
                """
            ),
            code_cell(
                """
                # 2. GPU, Python, and disk sanity check.
                !nvidia-smi
                !python --version
                !df -h /content
                """
            ),
            code_cell(
                """
                # 3. Clone the Paper 12 results branch into local SSD.
                %cd /content
                !rm -rf /content/AlphaEarth-System
                !git clone --branch paper12-results-colab-20260619 https://github.com/zhouning/alphaearth-training-system.git /content/AlphaEarth-System
                %cd /content/AlphaEarth-System
                !git rev-parse --abbrev-ref HEAD
                !git rev-parse HEAD
                !git log --oneline -3
                """
            ),
            code_cell(
                """
                # 4. Install the local package and notebook-only helpers.
                %cd /content/AlphaEarth-System
                !pip install -q -e . torchgeo pyyaml huggingface_hub
                """
            ),
            code_cell(
                """
                # 5. Stage Prithvi weights at the path the benchmark expects.
                %cd /content/AlphaEarth-System
                import os
                import shutil
                from huggingface_hub import hf_hub_download

                DRIVE_WEIGHTS = "/content/drive/MyDrive/Prithvi_100M.pt"
                LOCAL_WEIGHTS = "/content/AlphaEarth-System/data/weights/prithvi/Prithvi_100M.pt"
                os.makedirs(os.path.dirname(LOCAL_WEIGHTS), exist_ok=True)

                if os.path.exists(DRIVE_WEIGHTS):
                    shutil.copy(DRIVE_WEIGHTS, LOCAL_WEIGHTS)
                    print("Copied Prithvi weights from Drive.")
                elif not os.path.exists(LOCAL_WEIGHTS):
                    downloaded = hf_hub_download(
                        repo_id="ibm-nasa-geospatial/Prithvi-100M",
                        filename="Prithvi_100M.pt",
                    )
                    shutil.copy(downloaded, LOCAL_WEIGHTS)
                    print("Downloaded Prithvi weights from Hugging Face.")
                else:
                    print("Prithvi weights already present locally.")

                !ls -lh /content/AlphaEarth-System/data/weights/prithvi
                """
            ),
            code_cell(
                """
                # 6. Download the public EuroSAT cache into local SSD and smoke one sample per split.
                %cd /content/AlphaEarth-System
                EUROSAT_ROOT = "/content/AlphaEarth-System/data/eurosat"
                !python scripts/download_public_datasets.py --dataset eurosat --eurosat-root data/eurosat --max-samples 1
                !du -sh /content/AlphaEarth-System/data/eurosat
                """
            ),
            code_cell(
                """
                # 7. Dry-run the full EuroSAT matrix before training.
                %cd /content/AlphaEarth-System
                !python -m geoadapter.bench.run_benchmark --config geoadapter/bench/configs/eurosat_channel_bridge.yaml --dry-run
                """
            ),
            code_cell(
                """
                # 8. Archive any pre-rerun EuroSAT JSON files so the benchmark cannot resume from archive output.
                from datetime import datetime
                import shutil
                from pathlib import Path

                results_dir = Path("/content/drive/MyDrive/paper12_results")
                archive_dir = results_dir / "eurosat_channel_bridge_archive_pre_rerun"
                archive_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                for name in ["eurosat_channel_bridge.json", "eurosat_channel_bridge_summary.json"]:
                    src = results_dir / name
                    if src.exists():
                        dst = archive_dir / f"{stamp}_{name}"
                        shutil.move(str(src), str(dst))
                        print("Archived", src, "->", dst)
                    else:
                        print("No existing", src)
                """
            ),
            code_cell(
                """
                # 9. Run the 4-method x 3-seed EuroSAT benchmark. Checkpoints stay on local SSD.
                %cd /content/AlphaEarth-System
                !mkdir -p /content/eurosat_channel_bridge_runs
                !python -m geoadapter.bench.run_benchmark --config geoadapter/bench/configs/eurosat_channel_bridge.yaml --output /content/drive/MyDrive/paper12_results/eurosat_channel_bridge.json --checkpoint-dir /content/eurosat_channel_bridge_runs --checkpoint-every 5
                """
            ),
            code_cell(
                """
                # 10. Verify result counts, aggregate OA and macro-F1 by method, and persist a compact summary JSON to Drive.
                import json
                from collections import defaultdict
                from pathlib import Path
                from statistics import mean, stdev

                results_dir = Path("/content/drive/MyDrive/paper12_results")
                results_path = results_dir / "eurosat_channel_bridge.json"
                summary_path = results_dir / "eurosat_channel_bridge_summary.json"

                rows = json.loads(results_path.read_text(encoding="utf-8"))
                expected_rows = 12
                assert len(rows) == expected_rows, f"expected {expected_rows} rows, got {len(rows)}"

                grouped = defaultdict(list)
                for row in rows:
                    grouped[row["method"]].append(row)

                summary = {}
                for method, method_rows in sorted(grouped.items()):
                    oa = [float(row["overall_accuracy"]) for row in method_rows]
                    f1 = [float(row["macro_f1"]) for row in method_rows]
                    summary[method] = {
                        "overall_accuracy_mean": mean(oa),
                        "overall_accuracy_std": stdev(oa) if len(oa) > 1 else 0.0,
                        "macro_f1_mean": mean(f1),
                        "macro_f1_std": stdev(f1) if len(f1) > 1 else 0.0,
                        "seeds": [int(row["seed"]) for row in method_rows],
                    }

                for method, payload in summary.items():
                    print(method, payload)

                summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print("Wrote", summary_path)
                """
            ),
        ]
    )



def second_backbone_notebook() -> dict:
    return notebook(
        [
            markdown_cell(
                """
                <a href="https://colab.research.google.com/github/zhouning/alphaearth-training-system/blob/paper12-results-colab-20260619/colab/paper12_second_backbone_eurosat_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

                # Paper 12 Second-Backbone EuroSAT Validation

                This notebook runs the compact second-backbone validation for Paper 12. It uses a SatMAE-compatible ViT backbone and compares linear probing, split-QKV LoRA, and Houlsby adapters on EuroSAT `s2_full` and `rgb`.

                **Required runtime:** Colab Pro L4. A100 is faster but not required. T4 is acceptable only for a one-epoch smoke run.

                **Required weight file:** copy an official SatMAE-compatible checkpoint to `/content/drive/MyDrive/satmae_vit_base.pth` before running the training cell. The notebook fails if this file is missing so the experiment cannot silently run random weights.

                **Outputs written to Drive:**
                - `second_backbone_eurosat.json`
                - `second_backbone_eurosat_summary.json`

                **Expected matrix:** 1 backbone x 3 methods x 2 modalities x 3 seeds = 18 rows.
                """
            ),
            code_cell(
                """
                # 1. Mount Drive and create the results directory.
                from google.colab import drive
                drive.mount("/content/drive")

                import os

                RESULTS_DIR = "/content/drive/MyDrive/paper12_results"
                os.makedirs(RESULTS_DIR, exist_ok=True)
                print("Drive results directory:", RESULTS_DIR)
                """
            ),
            code_cell(
                """
                # 2. GPU, Python, and disk sanity check.
                !nvidia-smi
                !python --version
                !df -h /content
                """
            ),
            code_cell(
                """
                # 3. Clone the Paper 12 results branch into local SSD.
                %cd /content
                !rm -rf /content/AlphaEarth-System
                !git clone --branch paper12-results-colab-20260619 https://github.com/zhouning/alphaearth-training-system.git /content/AlphaEarth-System
                %cd /content/AlphaEarth-System
                !git rev-parse --abbrev-ref HEAD
                !git rev-parse HEAD
                !git log --oneline -3
                """
            ),
            code_cell(
                """
                # 4. Install the local package and notebook helpers.
                %cd /content/AlphaEarth-System
                !pip install -q -e . torchgeo pyyaml
                """
            ),
            code_cell(
                """
                # 5. Stage the SatMAE-compatible checkpoint at the path the config expects.
                %cd /content/AlphaEarth-System
                import os
                import shutil

                DRIVE_WEIGHTS = "/content/drive/MyDrive/satmae_vit_base.pth"
                LOCAL_WEIGHTS = "/content/AlphaEarth-System/data/weights/satmae/satmae_vit_base.pth"
                os.makedirs(os.path.dirname(LOCAL_WEIGHTS), exist_ok=True)

                assert os.path.exists(DRIVE_WEIGHTS), (
                    "Missing /content/drive/MyDrive/satmae_vit_base.pth. "
                    "Copy an official SatMAE-compatible checkpoint to Drive before running this notebook."
                )
                shutil.copy(DRIVE_WEIGHTS, LOCAL_WEIGHTS)
                print("Copied SatMAE checkpoint to", LOCAL_WEIGHTS)
                !ls -lh /content/AlphaEarth-System/data/weights/satmae
                """
            ),
            code_cell(
                """
                # 6. Download the public EuroSAT cache into local SSD and smoke one sample per split.
                %cd /content/AlphaEarth-System
                !python scripts/download_public_datasets.py --dataset eurosat --eurosat-root data/eurosat --max-samples 1
                !du -sh /content/AlphaEarth-System/data/eurosat
                """
            ),
            code_cell(
                """
                # 7. Dry-run the full second-backbone matrix before training.
                %cd /content/AlphaEarth-System
                !python -m geoadapter.bench.run_benchmark --config geoadapter/bench/configs/eurosat_second_backbone.yaml --dry-run
                """
            ),
            code_cell(
                """
                # 8. Run the 18-row second-backbone EuroSAT benchmark.
                # The runner resumes from existing rows in second_backbone_eurosat.json if the Colab session restarts.
                %cd /content/AlphaEarth-System
                !mkdir -p /content/second_backbone_eurosat_runs
                !python -m geoadapter.bench.run_benchmark --config geoadapter/bench/configs/eurosat_second_backbone.yaml --output /content/drive/MyDrive/paper12_results/second_backbone_eurosat.json --checkpoint-dir /content/second_backbone_eurosat_runs --checkpoint-every 5
                """
            ),
            code_cell(
                """
                # 9. Verify row count and write grouped summary JSON.
                import json
                from pathlib import Path

                results_dir = Path("/content/drive/MyDrive/paper12_results")
                raw_path = results_dir / "second_backbone_eurosat.json"
                summary_path = results_dir / "second_backbone_eurosat_summary.json"
                rows = json.loads(raw_path.read_text(encoding="utf-8"))
                expected_rows = 18
                assert len(rows) == expected_rows, f"expected {expected_rows} rows, got {len(rows)}"

                !python -m geoadapter.bench.second_backbone_summary --input /content/drive/MyDrive/paper12_results/second_backbone_eurosat.json --output /content/drive/MyDrive/paper12_results/second_backbone_eurosat_summary.json
                print(summary_path.read_text(encoding="utf-8")[:4000])
                """
            ),
        ]
    )

def capacity_sweep_notebook() -> dict:
    return notebook(
        [
            markdown_cell(
                """
                <a href="https://colab.research.google.com/github/zhouning/alphaearth-training-system/blob/paper12-results-colab-20260619/colab/paper12_peft_capacity_sweep_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

                # Paper 12 PEFT Capacity Sweep

                This notebook runs the reviewer-strengthening EuroSAT capacity sweep for Paper 12. It compares split-QKV LoRA ranks against Houlsby bottleneck widths under the same Prithvi-100M checkpoint, dataset, modality, seeds, and metrics.

                **Required runtime:** Colab Pro L4. A100 is faster but not required. T4 is acceptable only for a smoke run after reducing the matrix manually.

                **Storage policy:** download EuroSAT to Colab local SSD under `/content/AlphaEarth-System/data/eurosat`, keep checkpoints under `/content/peft_capacity_sweep_runs`, and only persist result JSON files to `/content/drive/MyDrive/paper12_results`.

                **Outputs written to Drive:**
                - `peft_capacity_sweep.json`
                - `peft_capacity_sweep_summary.json`

                **Methods in this run:** `linear_probe`, `lora_split_qkv_r4`, `lora_split_qkv_r8`, `lora_split_qkv_r16`, `lora_split_qkv_r32`, `lora_split_qkv_r64`, `houlsby_d8`, `houlsby_d16`, `houlsby_d32`, `houlsby_d64`.

                **Expected matrix:** 10 methods x 1 modality x 3 seeds = 30 rows.
                """
            ),
            code_cell(
                """
                # 1. Mount Drive and create the results directory.
                from google.colab import drive
                drive.mount("/content/drive")

                import os

                RESULTS_DIR = "/content/drive/MyDrive/paper12_results"
                os.makedirs(RESULTS_DIR, exist_ok=True)
                print("Drive results directory:", RESULTS_DIR)
                """
            ),
            code_cell(
                """
                # 2. GPU, Python, and disk sanity check.
                !nvidia-smi
                !python --version
                !df -h /content
                """
            ),
            code_cell(
                """
                # 3. Clone the Paper 12 results branch into local SSD.
                %cd /content
                !rm -rf /content/AlphaEarth-System
                !git clone --branch paper12-results-colab-20260619 https://github.com/zhouning/alphaearth-training-system.git /content/AlphaEarth-System
                %cd /content/AlphaEarth-System
                !git rev-parse --abbrev-ref HEAD
                !git rev-parse HEAD
                !git log --oneline -3
                """
            ),
            code_cell(
                """
                # 4. Install the local package and notebook-only helpers.
                %cd /content/AlphaEarth-System
                !pip install -q -e . torchgeo pyyaml huggingface_hub
                """
            ),
            code_cell(
                """
                # 5. Stage Prithvi weights at the path the benchmark expects.
                %cd /content/AlphaEarth-System
                import os
                import shutil
                from huggingface_hub import hf_hub_download

                DRIVE_WEIGHTS = "/content/drive/MyDrive/Prithvi_100M.pt"
                LOCAL_WEIGHTS = "/content/AlphaEarth-System/data/weights/prithvi/Prithvi_100M.pt"
                os.makedirs(os.path.dirname(LOCAL_WEIGHTS), exist_ok=True)

                if os.path.exists(DRIVE_WEIGHTS):
                    shutil.copy(DRIVE_WEIGHTS, LOCAL_WEIGHTS)
                    print("Copied Prithvi weights from Drive.")
                elif not os.path.exists(LOCAL_WEIGHTS):
                    downloaded = hf_hub_download(
                        repo_id="ibm-nasa-geospatial/Prithvi-100M",
                        filename="Prithvi_100M.pt",
                    )
                    shutil.copy(downloaded, LOCAL_WEIGHTS)
                    print("Downloaded Prithvi weights from Hugging Face.")
                else:
                    print("Prithvi weights already present locally.")

                !ls -lh /content/AlphaEarth-System/data/weights/prithvi
                """
            ),
            code_cell(
                """
                # 6. Download the public EuroSAT cache into local SSD and smoke one sample per split.
                %cd /content/AlphaEarth-System
                EUROSAT_ROOT = "/content/AlphaEarth-System/data/eurosat"
                !python scripts/download_public_datasets.py --dataset eurosat --eurosat-root data/eurosat --max-samples 1
                !du -sh /content/AlphaEarth-System/data/eurosat
                """
            ),
            code_cell(
                """
                # 7. Dry-run the full capacity-sweep matrix before training.
                %cd /content/AlphaEarth-System
                !python -m geoadapter.bench.run_benchmark --config geoadapter/bench/configs/eurosat_peft_capacity_sweep.yaml --dry-run
                """
            ),
            code_cell(
                """
                # 8. Run the 10-method x 3-seed EuroSAT capacity sweep.
                # The runner resumes from existing rows in peft_capacity_sweep.json if the Colab session restarts.
                %cd /content/AlphaEarth-System
                !mkdir -p /content/peft_capacity_sweep_runs
                !python -m geoadapter.bench.run_benchmark --config geoadapter/bench/configs/eurosat_peft_capacity_sweep.yaml --output /content/drive/MyDrive/paper12_results/peft_capacity_sweep.json --checkpoint-dir /content/peft_capacity_sweep_runs --checkpoint-every 5
                """
            ),
            code_cell(
                """
                # 9. Verify result counts, aggregate OA and macro-F1 by method, and persist a compact summary JSON to Drive.
                import json
                from collections import defaultdict
                from pathlib import Path
                from statistics import mean, stdev

                results_dir = Path("/content/drive/MyDrive/paper12_results")
                results_path = results_dir / "peft_capacity_sweep.json"
                summary_path = results_dir / "peft_capacity_sweep_summary.json"

                rows = json.loads(results_path.read_text(encoding="utf-8"))
                expected_rows = 30
                assert len(rows) == expected_rows, f"expected {expected_rows} rows, got {len(rows)}"

                grouped = defaultdict(list)
                for row in rows:
                    grouped[row["method"]].append(row)

                summary = {}
                for method, method_rows in sorted(grouped.items()):
                    oa = [float(row["overall_accuracy"]) for row in method_rows]
                    f1 = [float(row["macro_f1"]) for row in method_rows]
                    params = sorted({int(row["trainable_params"]) for row in method_rows})
                    summary[method] = {
                        "trainable_params": params[0] if len(params) == 1 else params,
                        "overall_accuracy_mean": mean(oa),
                        "overall_accuracy_std": stdev(oa) if len(oa) > 1 else 0.0,
                        "macro_f1_mean": mean(f1),
                        "macro_f1_std": stdev(f1) if len(f1) > 1 else 0.0,
                        "seeds": [int(row["seed"]) for row in method_rows],
                    }

                for method, payload in summary.items():
                    print(method, payload)

                summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print("Wrote", summary_path)
                """
            ),
        ]
    )



def landcoverai_decoder_ablation_notebook() -> dict:
    return notebook(
        [
            markdown_cell(
                """
                <a href="https://colab.research.google.com/github/zhouning/alphaearth-training-system/blob/master/colab/paper12_landcoverai_decoder_ablation_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

                # Paper 12 LandCover.ai Decoder Ablation

                This notebook runs the Paper 12 decoder-capacity ablation on LandCover.ai. It uses the existing YAML experiment matrix and compares the original linear segmentation decoder against the optional `conv_lite` decoder for linear probing, LoRA r8, and Houlsby adapters.

                **Required runtime:** Colab Pro L4 or A100. T4 is acceptable only for a one-epoch smoke run.

                **Storage policy:** download LandCover.ai to Colab local SSD under `/content/AlphaEarth-System/data/landcoverai`, persist result JSON and checkpoints to Google Drive, and resume from completed rows if Colab restarts.

                **Outputs written to Drive:**
                - `landcoverai_decoder_ablation.json`
                - `landcoverai_decoder_ablation_summary.json`

                **Expected matrix:** 3 PEFT settings x 2 decoders x 3 seeds = 18 rows.
                """
            ),
            code_cell(
                """
                # 1. Mount Drive and create persistent result/checkpoint directories.
                from google.colab import drive
                drive.mount("/content/drive")

                import os

                RESULTS_DIR = "/content/drive/MyDrive/paper12_results"
                CHECKPOINT_DIR = "/content/drive/MyDrive/paper12_checkpoints/landcoverai_decoder_ablation"
                os.makedirs(RESULTS_DIR, exist_ok=True)
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                print("Drive results directory:", RESULTS_DIR)
                print("Drive checkpoint directory:", CHECKPOINT_DIR)
                """
            ),
            code_cell(
                """
                # 2. GPU, Python, and disk sanity check.
                !nvidia-smi
                !python --version
                !df -h /content /content/drive
                """
            ),
            code_cell(
                """
                # 3. Clone the latest master branch into Colab local SSD.
                %cd /content
                !rm -rf /content/AlphaEarth-System
                !git clone --branch master https://github.com/zhouning/alphaearth-training-system.git /content/AlphaEarth-System
                %cd /content/AlphaEarth-System
                !git pull --ff-only origin master
                !git rev-parse --abbrev-ref HEAD
                !git rev-parse HEAD
                !git log --oneline -3
                """
            ),
            code_cell(
                """
                # 4. Install the local package and benchmark dependencies.
                %cd /content/AlphaEarth-System
                !pip install -q -e . torchgeo pyyaml huggingface_hub
                """
            ),
            code_cell(
                """
                # 5. Stage Prithvi weights and write a Colab-local config with absolute paths.
                %cd /content/AlphaEarth-System
                import os
                import shutil
                import yaml
                from huggingface_hub import hf_hub_download

                DRIVE_WEIGHTS = "/content/drive/MyDrive/Prithvi_100M.pt"
                LOCAL_WEIGHTS = "/content/AlphaEarth-System/data/weights/prithvi/Prithvi_100M.pt"
                os.makedirs(os.path.dirname(LOCAL_WEIGHTS), exist_ok=True)

                if os.path.exists(DRIVE_WEIGHTS):
                    shutil.copy(DRIVE_WEIGHTS, LOCAL_WEIGHTS)
                    print("Copied Prithvi weights from Drive.")
                elif not os.path.exists(LOCAL_WEIGHTS):
                    downloaded = hf_hub_download(
                        repo_id="ibm-nasa-geospatial/Prithvi-100M",
                        filename="Prithvi_100M.pt",
                    )
                    shutil.copy(downloaded, LOCAL_WEIGHTS)
                    print("Downloaded Prithvi weights from Hugging Face.")
                else:
                    print("Prithvi weights already present locally.")

                CONFIG_IN = "/content/AlphaEarth-System/geoadapter/bench/configs/landcoverai_decoder_ablation.yaml"
                CONFIG_COLAB = "/content/AlphaEarth-System/geoadapter/bench/configs/landcoverai_decoder_ablation_colab.yaml"
                with open(CONFIG_IN, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                cfg["experiment"]["dataset_root"] = "/content/AlphaEarth-System/data/landcoverai"
                cfg["experiment"]["allow_synthetic_fallback"] = False
                cfg.setdefault("prithvi", {})["checkpoint"] = LOCAL_WEIGHTS
                with open(CONFIG_COLAB, "w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)

                print("Colab config:", CONFIG_COLAB)
                print("Prithvi checkpoint:", LOCAL_WEIGHTS)
                !ls -lh /content/AlphaEarth-System/data/weights/prithvi
                """
            ),
            code_cell(
                """
                # 6. Download/smoke-test LandCover.ai train and validation splits.
                %cd /content/AlphaEarth-System
                from geoadapter.data.datasets import load_landcoverai

                LANDCOVER_ROOT = "/content/AlphaEarth-System/data/landcoverai"
                for split in ("train", "val"):
                    ds = load_landcoverai(root=LANDCOVER_ROOT, split=split, max_samples=1)
                    image, mask = ds[0]
                    print(
                        split,
                        "len=", len(ds),
                        "image_shape=", tuple(image.shape),
                        "mask_shape=", tuple(mask.shape),
                        "mask_values=", mask.unique()[:20].tolist(),
                    )
                """
            ),
            code_cell(
                """
                # 7. Dry-run the full 18-row decoder-ablation matrix.
                %cd /content/AlphaEarth-System
                !python -m geoadapter.bench.run_benchmark --config {CONFIG_COLAB} --dry-run
                """
            ),
            code_cell(
                """
                # 8. Run the full decoder ablation. The runner resumes completed rows from the Drive JSON.
                %cd /content/AlphaEarth-System
                OUTPUT_JSON = f"{RESULTS_DIR}/landcoverai_decoder_ablation.json"
                !python -m geoadapter.bench.run_benchmark --config {CONFIG_COLAB} --output {OUTPUT_JSON} --checkpoint-dir {CHECKPOINT_DIR} --checkpoint-every 5
                """
            ),
            code_cell(
                """
                # 9. Verify row count and summarize decoder effects by PEFT family.
                import json
                from collections import defaultdict
                from pathlib import Path
                from statistics import mean, stdev

                results_path = Path(RESULTS_DIR) / "landcoverai_decoder_ablation.json"
                summary_path = Path(RESULTS_DIR) / "landcoverai_decoder_ablation_summary.json"
                rows = json.loads(results_path.read_text(encoding="utf-8"))
                expected_rows = 18
                assert len(rows) == expected_rows, f"expected {expected_rows} rows, got {len(rows)}"

                def method_family(method: str) -> str:
                    if method.startswith("linear_probe"):
                        return "linear_probe"
                    if method.startswith("lora_r8"):
                        return "lora_r8"
                    if method.startswith("houlsby"):
                        return "houlsby"
                    return method

                def decoder_name(method: str) -> str:
                    return "conv_lite_d128" if "conv_lite" in method else "linear"

                grouped = defaultdict(list)
                for row in rows:
                    grouped[(method_family(row["method"]), decoder_name(row["method"]))].append(row)

                groups = []
                for (family, decoder), group_rows in sorted(grouped.items()):
                    miou = [float(row["mIoU"]) for row in group_rows]
                    groups.append(
                        {
                            "method_family": family,
                            "decoder": decoder,
                            "mIoU_mean": mean(miou),
                            "mIoU_std": stdev(miou) if len(miou) > 1 else 0.0,
                            "seeds": [int(row["seed"]) for row in group_rows],
                            "trainable_params": sorted({int(row["trainable_params"]) for row in group_rows}),
                        }
                    )

                by_family = defaultdict(dict)
                for group in groups:
                    by_family[group["method_family"]][group["decoder"]] = group["mIoU_mean"]

                deltas = []
                for family, payload in sorted(by_family.items()):
                    if "linear" in payload and "conv_lite_d128" in payload:
                        deltas.append(
                            {
                                "method_family": family,
                                "conv_lite_minus_linear_mIoU": payload["conv_lite_d128"] - payload["linear"],
                            }
                        )

                summary = {
                    "schema": "paper12.landcoverai_decoder_ablation_summary.v1",
                    "row_count": len(rows),
                    "groups": groups,
                    "decoder_deltas": deltas,
                }
                summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print(json.dumps(summary, indent=2))
                print("Wrote", summary_path)
                """
            ),
        ]
    )


def geovlm_prompt_notebook() -> dict:
    return notebook(
        [
            dedented_markdown_cell(
                """
                <a href="https://colab.research.google.com/github/zhouning/alphaearth-training-system/blob/master/colab/paper12_geovlm_prompt_segmentation_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

                # Paper 12 GeoVLM Prompt Segmentation MVP

                This notebook evaluates a bounded English prompt segmentation MVP for `building`, `road`, and `water` on LandCoverAI. It is not a complete or open-vocabulary ArcGIS GeoVLM and does not support captioning, VQA, Chinese prompts, or arbitrary unseen concepts.

                The seed-42 smoke stage must pass before the optional three-seed prompt/baseline matrix is enabled. Real data, Prithvi weights, and SigLIP weights are mandatory; synthetic fallback is forbidden.
                """
            ),
            dedented_code_cell(
                """
                # 1. Mount Drive and prepare resumable result, checkpoint, preview, and cache paths.
                from google.colab import drive
                drive.mount("/content/drive")

                import os
                import shutil
                from pathlib import Path

                DRIVE_RESULTS_DIR = Path("/content/drive/MyDrive/paper12_results")
                CHECKPOINT_DIR = Path("/content/drive/MyDrive/paper12_checkpoints/geovlm_prompt_segmentation")
                PREVIEW_DIR = Path("/content/drive/MyDrive/paper12_previews/geovlm_prompt_segmentation")
                HF_CACHE_DIR = Path("/content/drive/MyDrive/huggingface_cache/paper12_geovlm")
                LOCAL_RESULTS_DIR = Path("/content/paper12_geovlm_results")
                for path in (DRIVE_RESULTS_DIR, CHECKPOINT_DIR, PREVIEW_DIR, HF_CACHE_DIR, LOCAL_RESULTS_DIR):
                    path.mkdir(parents=True, exist_ok=True)

                RAW_JSON = LOCAL_RESULTS_DIR / "geovlm_prompt_segmentation.json"
                SUMMARY_JSON = LOCAL_RESULTS_DIR / "geovlm_prompt_segmentation_summary.json"
                DRIVE_RAW_JSON = DRIVE_RESULTS_DIR / RAW_JSON.name
                DRIVE_SUMMARY_JSON = DRIVE_RESULTS_DIR / SUMMARY_JSON.name
                SIGLIP_REVISION_PIN = HF_CACHE_DIR / "resolved_revision.txt"
                DRIVE_CONFIG_COLAB = DRIVE_RESULTS_DIR / "geovlm_prompt_segmentation_colab.yaml"
                os.environ["HF_HOME"] = str(HF_CACHE_DIR)
                print("Results:", DRIVE_RESULTS_DIR)
                print("Checkpoints:", CHECKPOINT_DIR)
                print("Previews:", PREVIEW_DIR)
                print("Hugging Face cache:", HF_CACHE_DIR)
                """
            ),
            dedented_code_cell(
                """
                # Archive the failed seed-42 artifacts once before running the v2 recovery.
                import json

                ARCHIVE_FAILED_RUN = False
                RESULTS_SCHEMA_V2 = "paper12.geovlm_prompt_results.v2"
                TRAINING_CONTRACT_V2 = "paper12.geovlm_prompt_training.v2"
                REQUIRED_METHODS = (
                    "siglip_film_dense_similarity_houlsby",
                    "no_text_three_binary_heads_houlsby",
                )
                REQUIRED_SEEDS = (42, 123, 456)
                REQUIRED_CLASSES = ("building", "road", "water")
                SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
                FAILED_ARCHIVE_DIR = DRIVE_RESULTS_DIR / "failed_seed42_20260724"
                FAILED_STAGING_DIR = (
                    DRIVE_RESULTS_DIR / ".failed_seed42_20260724.incomplete"
                )
                FAILED_RAW_JSON = DRIVE_RAW_JSON
                FAILED_SUMMARY_JSON = DRIVE_SUMMARY_JSON
                FAILED_CHECKPOINT = (
                    CHECKPOINT_DIR / "siglip_film_dense_similarity_houlsby__seed42.pt"
                )
                FAILED_PREVIEWS = sorted(PREVIEW_DIR.glob("seed42__*.png"))

                compatible_v2_raw = False
                if DRIVE_RAW_JSON.exists():
                    try:
                        drive_raw_payload = json.loads(
                            DRIVE_RAW_JSON.read_text(encoding="utf-8")
                        )
                    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                        print("Drive raw is not compatible recovery state:", exc)
                    else:
                        rows = (
                            drive_raw_payload.get("rows")
                            if isinstance(drive_raw_payload, dict)
                            else None
                        )
                        compatible_v2_raw = (
                            isinstance(drive_raw_payload, dict)
                            and drive_raw_payload.get("schema") == RESULTS_SCHEMA_V2
                            and drive_raw_payload.get("training_contract")
                            == TRAINING_CONTRACT_V2
                            and isinstance(rows, list)
                            and all(
                                isinstance(row, dict)
                                and row.get("training_contract")
                                == TRAINING_CONTRACT_V2
                                and row.get("siglip_model_id") == SIGLIP_MODEL_ID
                                for row in rows
                            )
                        )
                        if compatible_v2_raw:
                            seen = set()
                            classes_by_pair = {}
                            for row in rows:
                                method = row.get("method")
                                seed = row.get("seed")
                                class_name = row.get("class_name")
                                key = (method, seed, class_name)
                                if (
                                    method not in REQUIRED_METHODS
                                    or isinstance(seed, bool)
                                    or not isinstance(seed, int)
                                    or seed not in REQUIRED_SEEDS
                                    or class_name not in REQUIRED_CLASSES
                                    or key in seen
                                ):
                                    compatible_v2_raw = False
                                    break
                                seen.add(key)
                                classes_by_pair.setdefault((method, seed), set()).add(
                                    class_name
                                )
                            if compatible_v2_raw:
                                compatible_v2_raw = all(
                                    classes == set(REQUIRED_CLASSES)
                                    for classes in classes_by_pair.values()
                                )
                            if compatible_v2_raw:
                                revisions = {row.get("siglip_revision") for row in rows}
                                compatible_v2_raw = (
                                    len(revisions) == 1
                                    and all(
                                        isinstance(revision, str) and revision.strip()
                                        for revision in revisions
                                    )
                                )
                            if compatible_v2_raw and SIGLIP_REVISION_PIN.exists():
                                pinned_revision = SIGLIP_REVISION_PIN.read_text(
                                    encoding="utf-8"
                                ).strip()
                                compatible_v2_raw = bool(pinned_revision) and revisions == {
                                    pinned_revision
                                }

                archive_sources = [
                    path
                    for path in (
                        FAILED_RAW_JSON,
                        FAILED_SUMMARY_JSON,
                        FAILED_CHECKPOINT,
                    )
                    if path.exists()
                ] + FAILED_PREVIEWS
                compatible_recovery = (
                    compatible_v2_raw
                    and not FAILED_CHECKPOINT.exists()
                    and not FAILED_STAGING_DIR.exists()
                )
                failed_artifacts = [] if compatible_recovery else archive_sources
                archive_pending = bool(failed_artifacts) or FAILED_STAGING_DIR.exists()
                if archive_pending and not ARCHIVE_FAILED_RUN:
                    pending_paths = failed_artifacts + (
                        [FAILED_STAGING_DIR] if FAILED_STAGING_DIR.exists() else []
                    )
                    raise RuntimeError(
                        "Failed seed-42 artifacts still exist; set ARCHIVE_FAILED_RUN = True "
                        "once to archive it before recovery: "
                        + ", ".join(str(path) for path in pending_paths)
                    )
                if archive_pending:
                    if FAILED_ARCHIVE_DIR.exists():
                        raise RuntimeError(
                            "Failed-run archive already exists while recovery artifacts are "
                            "pending; resolve the archive collision before retrying: "
                            + str(FAILED_ARCHIVE_DIR)
                        )
                    FAILED_STAGING_DIR.mkdir(parents=True, exist_ok=True)
                    for source in failed_artifacts:
                        destination = FAILED_STAGING_DIR / source.name
                        if not source.exists():
                            continue
                        if destination.exists():
                            raise RuntimeError(
                                "Failed artifact source and staged destination both exist; "
                                "refusing to overwrite either path: "
                                + str(source)
                                + " | "
                                + str(destination)
                            )
                        shutil.move(str(source), str(destination))
                        print("Archived", source, "to", destination)
                    FAILED_STAGING_DIR.rename(FAILED_ARCHIVE_DIR)
                    print("Finalized failed-run archive:", FAILED_ARCHIVE_DIR)
                for source, destination in (
                    (DRIVE_RAW_JSON, RAW_JSON),
                    (DRIVE_SUMMARY_JSON, SUMMARY_JSON),
                ):
                    if source.exists():
                        shutil.copy2(source, destination)
                """
            ),
            dedented_code_cell(
                """
                # 2. Record GPU, Python, disk, Torch, and CUDA details.
                !nvidia-smi
                !python --version
                !df -h /content /content/drive

                import torch
                print("torch:", torch.__version__)
                print("cuda available:", torch.cuda.is_available())
                print("torch cuda:", torch.version.cuda)
                if torch.cuda.is_available():
                    print("device:", torch.cuda.get_device_name(0))
                """
            ),
            dedented_code_cell(
                """
                # 3. Clone or fast-forward the latest master branch and record the exact commit.
                %cd /content
                !if [ ! -d /content/AlphaEarth-System/.git ]; then git clone --branch master https://github.com/zhouning/alphaearth-training-system.git /content/AlphaEarth-System; fi
                %cd /content/AlphaEarth-System
                !git fetch origin master
                !git checkout master
                !git pull --ff-only origin master
                !git rev-parse --abbrev-ref HEAD
                !git rev-parse HEAD
                !git log --oneline -3
                """
            ),
            dedented_code_cell(
                """
                # 4. Install the GeoVLM optional dependencies and record exact versions.
                %cd /content/AlphaEarth-System
                !pip install -q -e '.[geovlm]' torchgeo

                from importlib.metadata import version
                for package in ("geoadapter", "torch", "torchvision", "torchgeo", "transformers", "huggingface-hub", "rasterio", "numpy"):
                    try:
                        print(package, version(package))
                    except Exception as exc:
                        print(package, "unavailable", exc)
                """
            ),
            dedented_code_cell(
                """
                # 5. Stage and hash the required Prithvi checkpoint.
                %cd /content/AlphaEarth-System
                import hashlib
                from huggingface_hub import hf_hub_download

                DRIVE_PRITHVI = Path("/content/drive/MyDrive/Prithvi_100M.pt")
                LOCAL_PRITHVI = Path("/content/AlphaEarth-System/data/weights/prithvi/Prithvi_100M.pt")
                LOCAL_PRITHVI.parent.mkdir(parents=True, exist_ok=True)
                if DRIVE_PRITHVI.exists():
                    shutil.copy2(DRIVE_PRITHVI, LOCAL_PRITHVI)
                    print("Copied Prithvi_100M.pt from Drive")
                elif not LOCAL_PRITHVI.exists():
                    downloaded = hf_hub_download(
                        repo_id="ibm-nasa-geospatial/Prithvi-100M",
                        filename="Prithvi_100M.pt",
                    )
                    shutil.copy2(downloaded, LOCAL_PRITHVI)
                    print("Downloaded Prithvi_100M.pt from Hugging Face")

                def sha256(path):
                    digest = hashlib.sha256()
                    with Path(path).open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                    return digest.hexdigest()

                assert LOCAL_PRITHVI.is_file(), LOCAL_PRITHVI
                PRITHVI_SHA256 = sha256(LOCAL_PRITHVI)
                print("Prithvi path:", LOCAL_PRITHVI)
                print("Prithvi SHA-256:", PRITHVI_SHA256)
                """
            ),
            dedented_code_cell(
                """
                # 6. Pre-cache the frozen SigLIP text tower and record its resolved revision.
                from huggingface_hub import model_info, snapshot_download

                SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
                if SIGLIP_REVISION_PIN.exists():
                    SIGLIP_REVISION = SIGLIP_REVISION_PIN.read_text(
                        encoding="utf-8"
                    ).strip()
                    if not SIGLIP_REVISION:
                        raise ValueError("SigLIP revision pin must be non-empty")
                else:
                    SIGLIP_REVISION = getattr(model_info(SIGLIP_MODEL_ID), "sha", None)
                    if not isinstance(SIGLIP_REVISION, str) or not SIGLIP_REVISION.strip():
                        raise ValueError("resolved SigLIP revision must be non-empty")
                    SIGLIP_REVISION = SIGLIP_REVISION.strip()
                    revision_staging = SIGLIP_REVISION_PIN.with_suffix(".txt.tmp")
                    revision_staging.write_text(
                        SIGLIP_REVISION + "\\n", encoding="utf-8"
                    )
                    revision_staging.replace(SIGLIP_REVISION_PIN)
                SIGLIP_CACHE_PATH = snapshot_download(
                    repo_id=SIGLIP_MODEL_ID,
                    revision=SIGLIP_REVISION,
                    cache_dir=str(HF_CACHE_DIR),
                )
                print("SigLIP model:", SIGLIP_MODEL_ID)
                print("SigLIP resolved revision:", SIGLIP_REVISION)
                print("SigLIP cache:", SIGLIP_CACHE_PATH)
                """
            ),
            dedented_code_cell(
                """
                # 7. Download LandCoverAI and verify the official five-class mask contract.
                %cd /content/AlphaEarth-System
                from geoadapter.data.datasets import load_landcoverai

                LANDCOVER_ROOT = "/content/AlphaEarth-System/data/landcoverai"
                ALLOWED_MASK_VALUES = {0, 1, 2, 3, 4}
                datasets = {}
                for split in ("train", "val"):
                    dataset = load_landcoverai(root=LANDCOVER_ROOT, split=split)
                    observed = set()
                    for index in range(len(dataset)):
                        _, mask = dataset[index]
                        observed.update(int(value) for value in mask.unique().tolist())
                    assert observed <= ALLOWED_MASK_VALUES, (split, sorted(observed))
                    datasets[split] = dataset
                    print(split, "samples=", len(dataset), "mask_values=", sorted(observed))
                """
            ),
            dedented_code_cell(
                """
                # 8. Write a Colab-local real-data config with absolute paths and no fallback.
                import yaml

                CONFIG_SOURCE = Path("/content/AlphaEarth-System/geoadapter/bench/configs/geovlm_prompt_segmentation.yaml")
                CONFIG_COLAB = Path("/content/AlphaEarth-System/geoadapter/bench/configs/geovlm_prompt_segmentation_colab.yaml")
                PROMPT_CONFIG = Path("/content/AlphaEarth-System/geoadapter/bench/configs/geovlm_prompts.yaml")
                config = yaml.safe_load(CONFIG_SOURCE.read_text(encoding="utf-8"))
                config["experiment"]["dataset_root"] = LANDCOVER_ROOT
                config["experiment"]["prompt_config"] = str(PROMPT_CONFIG)
                config["experiment"]["allow_synthetic_fallback"] = False
                config["experiment"]["training_contract"] = "paper12.geovlm_prompt_training.v2"
                config["prithvi"]["checkpoint"] = str(LOCAL_PRITHVI)
                config["text_encoder"]["model_id"] = SIGLIP_MODEL_ID
                config["text_encoder"]["revision"] = SIGLIP_REVISION
                config["text_encoder"]["local_files_only"] = True
                config["text_encoder"]["cache_dir"] = str(HF_CACHE_DIR)
                CONFIG_COLAB.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                shutil.copy2(CONFIG_COLAB, DRIVE_CONFIG_COLAB)
                print(CONFIG_COLAB.read_text(encoding="utf-8"))
                print("Persisted config:", DRIVE_CONFIG_COLAB)
                """
            ),
            dedented_code_cell(
                """
                # 9. Run focused offline contract tests before real training.
                %cd /content/AlphaEarth-System
                !python -m pytest tests/test_prompt_segmentation_data.py tests/test_prithvi_position_embeddings.py tests/test_prompt_segmentation_model.py tests/test_prompt_segmentation_engine.py tests/test_geovlm_training.py tests/test_geovlm_prompt_summary.py tests/test_geovlm_prompt_runner.py tests/test_geovlm_prompt_inference.py -v
                """
            ),
            dedented_code_cell(
                """
                # 10. Run seed 42, persist diagnostics, and require every smoke check to pass.
                # Runner stage flag: --stage seed42
                import json
                import subprocess

                seed42_command = [
                    "python", "-m", "geoadapter.bench.run_geovlm_prompt_segmentation",
                    "--config", str(CONFIG_COLAB),
                    "--output", str(RAW_JSON),
                    "--summary-output", str(SUMMARY_JSON),
                    "--checkpoint-dir", str(CHECKPOINT_DIR),
                    "--preview-dir", str(PREVIEW_DIR),
                    "--stage", "seed42",
                ]
                try:
                    subprocess.run(seed42_command, check=True)
                finally:
                    for source, destination in ((RAW_JSON, DRIVE_RAW_JSON), (SUMMARY_JSON, DRIVE_SUMMARY_JSON)):
                        if source.exists():
                            shutil.copy2(source, destination)

                raw_payload = json.loads(RAW_JSON.read_text(encoding="utf-8"))
                seed42_smoke = raw_payload["seed42_smoke"]
                print("seed42 smoke:", json.dumps(seed42_smoke, indent=2))
                assert seed42_smoke["passed"], seed42_smoke["failed_checks"]
                seed42_rows = [
                    row for row in raw_payload["rows"]
                    if row["method"] == "siglip_film_dense_similarity_houlsby" and row["seed"] == 42
                ]
                assert len(seed42_rows) == 3
                assert all(row["checkpoint_reproduced"] for row in seed42_rows)
                checkpoint_path = (
                    CHECKPOINT_DIR
                    / "siglip_film_dense_similarity_houlsby__seed42.best.pt"
                )
                checkpoint_payload = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                assert checkpoint_payload["metadata"]["training_contract"] == (
                    "paper12.geovlm_prompt_training.v2"
                )
                print(
                    "selected epoch:",
                    checkpoint_payload["best_epoch"],
                    "rank:",
                    checkpoint_payload["best_probe_rank"],
                )
                print("reloaded checkpoint metadata:", json.dumps(checkpoint_payload["metadata"], indent=2))
                summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
                print("mvp_status:", summary["mvp_status"])
                print("failed gates:", summary["failed_gates"])
                """
            ),
            dedented_code_cell(
                """
                # 11. Explicitly opt in to the complete two-method x three-seed matrix.
                # Runner stage flag: --stage full
                RUN_FULL_MATRIX = False
                if RUN_FULL_MATRIX:
                    full_command = [
                        "python", "-m", "geoadapter.bench.run_geovlm_prompt_segmentation",
                        "--config", str(CONFIG_COLAB),
                        "--output", str(RAW_JSON),
                        "--summary-output", str(SUMMARY_JSON),
                        "--checkpoint-dir", str(CHECKPOINT_DIR),
                        "--preview-dir", str(PREVIEW_DIR),
                        "--stage", "full",
                    ]
                    try:
                        subprocess.run(full_command, check=True)
                    finally:
                        for source, destination in ((RAW_JSON, DRIVE_RAW_JSON), (SUMMARY_JSON, DRIVE_SUMMARY_JSON)):
                            if source.exists():
                                shutil.copy2(source, destination)
                else:
                    print("Full matrix disabled. Set RUN_FULL_MATRIX = True only after seed42 passes.")
                """
            ),
            dedented_code_cell(
                """
                # 12. Validate pair count, rebuild gates with 1,000 bootstraps, and persist artifacts.
                raw_payload = json.loads(RAW_JSON.read_text(encoding="utf-8"))
                rows = raw_payload["rows"]
                method_seed_pairs = sorted({(row["method"], int(row["seed"])) for row in rows})
                print("expected method/seed pairs = 6")
                print("observed method/seed pairs:", len(method_seed_pairs), method_seed_pairs)
                if RUN_FULL_MATRIX:
                    assert len(method_seed_pairs) == 6, method_seed_pairs

                summary_command = [
                    "python", "-m", "geoadapter.bench.geovlm_prompt_summary",
                    "--input", str(RAW_JSON),
                    "--output", str(SUMMARY_JSON),
                    "--bootstrap-iterations", "1000",
                ]
                subprocess.run(summary_command, check=True)
                summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
                print("mvp_status:", summary["mvp_status"])
                for gate, passed in summary.get("gates", {}).items():
                    print(gate, passed)
                print("failed gates:", summary.get("failed_gates", []))
                print("incomplete reasons:", summary.get("incomplete_reasons", []))
                for source, destination in ((RAW_JSON, DRIVE_RAW_JSON), (SUMMARY_JSON, DRIVE_SUMMARY_JSON)):
                    shutil.copy2(source, destination)
                print("checkpoints:", sorted(str(path) for path in CHECKPOINT_DIR.glob("*.pt")))
                print("previews:", sorted(str(path) for path in PREVIEW_DIR.glob("*.png")))
                print("Drive raw:", DRIVE_RAW_JSON)
                print("Drive summary:", DRIVE_SUMMARY_JSON)
                """
            ),
        ]
    )


def main() -> None:
    COLAB_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        GEOVLM_PROMPT_OUT: geovlm_prompt_notebook(),
        LANDCOVER_DECODER_OUT: landcoverai_decoder_ablation_notebook(),
        SECOND_BACKBONE_OUT: second_backbone_notebook(),
        CAPACITY_OUT: capacity_sweep_notebook(),
        LOVE_OUT: loveda_notebook(),
        EURO_OUT: eurosat_notebook(),
    }
    for path, payload in outputs.items():
        rendered = json.dumps(payload, indent=1)
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            if existing_notebook_matches(existing, rendered):
                print(f"[ok] unchanged {path}")
                continue
            if has_execution_artifacts(existing):
                print(f"[ok] preserved executed notebook {path}")
                continue
        try:
            path.write_text(rendered, encoding="utf-8")
            print(f"[ok] wrote {path}")
        except PermissionError:
            if path == CAPACITY_OUT or not path.exists():
                raise
            print(f"[warn] skipped locked existing notebook {path}")


if __name__ == "__main__":
    main()
