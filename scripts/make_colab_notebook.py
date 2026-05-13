"""Generate colab/train_linhe.ipynb from cell-by-cell strings.

Run once: python scripts/make_colab_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "colab" / "train_linhe.ipynb"


CELLS = [
    ("markdown", """# AlphaEarth-System — Linhe PEFT Training on Colab

Trains Prithvi + PEFT on the Linhe RGB patch subset packaged by
`scripts/package_colab_data.py`. Saves checkpoints back to Google Drive so a
mid-session Colab kill can resume from the last epoch.

**Before running:**
1. Upload `linhe_<quarters>.tar.gz` + its `.sha256` to Drive (MyDrive root).
2. Runtime → Change runtime type → L4 GPU (or T4 for smoke).
3. Run cells top-to-bottom.
"""),
    ("code", """# 1. Mount Drive, pick archive
from google.colab import drive
drive.mount('/content/drive')

ARCHIVE_NAME = 'linhe_2025Q1_2025Q4.tar.gz'   # change if you packaged a different subset
CHECKPOINT_DIR = '/content/drive/MyDrive/linhe_ckpt'
RESULTS_DIR = '/content/drive/MyDrive/linhe_results'

import os
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
print('Drive mounted; checkpoint dir:', CHECKPOINT_DIR)"""),
    ("code", """# 2. GPU check
!nvidia-smi | head -20"""),
    ("code", """# 3. Clone repo
%cd /content
!git clone https://github.com/zhouning/AlphaEarth-System.git 2>/dev/null || (cd AlphaEarth-System && git pull)
%cd /content/AlphaEarth-System
!git log --oneline -5"""),
    ("code", """# 4. Install deps
!pip install -q -e .
!pip install -q pyarrow scikit-learn matplotlib"""),
    ("code", """# 5. Copy + verify + extract data archive
import shutil, subprocess, os
src = f'/content/drive/MyDrive/{ARCHIVE_NAME}'
sha_src = src + '.sha256'
dst = f'/content/{ARCHIVE_NAME}'
assert os.path.exists(src), f'Upload {ARCHIVE_NAME} to Drive first'
shutil.copy(src, dst)
shutil.copy(sha_src, '/content/')
print('copied, verifying sha256...')
subprocess.run(['sha256sum', '-c', f'/content/{os.path.basename(sha_src)}'], cwd='/content', check=True)
print('extracting...')
subprocess.run(['tar', 'xzf', dst, '-C', '/content/AlphaEarth-System/'], check=True)
!ls -la data/linhe_patches/ | head -10
!cat data/linhe_patches/_colab_manifest.txt"""),
    ("code", """# 6. Download Prithvi weights from HuggingFace (350 MB)
import os
os.makedirs('data/weights/prithvi', exist_ok=True)
if not os.path.exists('data/weights/prithvi/Prithvi_100M.pt'):
    !wget -q https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/resolve/main/Prithvi_100M.pt -O data/weights/prithvi/Prithvi_100M.pt
!ls -la data/weights/prithvi/"""),
    ("code", """# 7. GPU throughput sanity check (~30 s)
!python scripts/bench_prithvi_throughput.py --steps 20 --batch-size 16"""),
    ("code", """# 8. Run benchmark with checkpoint persistence to Drive
# Resume is automatic: existing results JSON skips done experiments, and per-epoch
# checkpoints under CHECKPOINT_DIR resume a mid-trained experiment to the last
# saved epoch. Safe to re-run this cell after a session kill.
CONFIG = 'geoadapter/bench/configs/linhe_buildings.yaml'   # or linhe_buildings_smoke.yaml for a fast test
OUTPUT = f'{RESULTS_DIR}/linhe_buildings.json'
!python -m geoadapter.bench.run_benchmark \\
    --config {CONFIG} \\
    --output {OUTPUT} \\
    --checkpoint-dir {CHECKPOINT_DIR} \\
    --checkpoint-every 2"""),
    ("code", """# 9. Peek results
import json
with open(OUTPUT) as f:
    results = json.load(f)
import pandas as pd
df = pd.DataFrame(results)
print(df[['method','modality','seed','trainable_params','mIoU']].sort_values('mIoU', ascending=False).to_string())"""),
    ("markdown", """## Usage notes

- **Out-of-time kill**: Colab may cut the session after ~24 h / idle. Re-running
  cell 8 resumes from the last checkpoint + skips completed experiments.
- **Budget**: one full run of 68-scene 20-epoch × 2 method × 3 seed ≈ 15 L4
  GPU-hours ≈ 75 compute units.
- **Change subset**: repackage locally with
  `python scripts/package_colab_data.py --quarters 2025Q3 2025Q4` and upload
  the new tar.gz; update `ARCHIVE_NAME` in cell 1.
- **Checkpoint cleanup**: after a successful full run, delete
  `/content/drive/MyDrive/linhe_ckpt/` to free Drive space.
"""),
]


def build() -> dict:
    nbcells = []
    for kind, src in CELLS:
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": src.strip().split("\n"),
        }
        # ensure every line ends with a newline except the last
        cell["source"] = [l + "\n" for l in cell["source"][:-1]] + [cell["source"][-1]]
        if kind == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        nbcells.append(cell)
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
        },
        "cells": nbcells,
    }


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(f"[ok] wrote {OUT}")
