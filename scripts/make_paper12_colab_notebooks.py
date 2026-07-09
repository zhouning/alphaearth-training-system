from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLAB_DIR = ROOT / "colab"
LOVE_OUT = COLAB_DIR / "paper12_loveda_full_finetune_colab.ipynb"
EURO_OUT = COLAB_DIR / "paper12_eurosat_channel_bridge_colab.ipynb"
CAPACITY_OUT = COLAB_DIR / "paper12_peft_capacity_sweep_colab.ipynb"
SECOND_BACKBONE_OUT = COLAB_DIR / "paper12_second_backbone_eurosat_colab.ipynb"
LANDCOVER_DECODER_OUT = COLAB_DIR / "paper12_landcoverai_decoder_ablation_colab.ipynb"
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
def main() -> None:
    COLAB_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        LANDCOVER_DECODER_OUT: landcoverai_decoder_ablation_notebook(),
        SECOND_BACKBONE_OUT: second_backbone_notebook(),
        CAPACITY_OUT: capacity_sweep_notebook(),
        LOVE_OUT: loveda_notebook(),
        EURO_OUT: eurosat_notebook(),
    }
    for path, payload in outputs.items():
        rendered = json.dumps(payload, indent=1)
        if path.exists() and path.read_text(encoding="utf-8") == rendered:
            print(f"[ok] unchanged {path}")
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
