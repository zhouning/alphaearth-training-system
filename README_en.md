# Geo-MLOps: PEFT Benchmark Platform for Geospatial Foundation Models

**English** | [简体中文](README.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/tests-39%20passed-brightgreen.svg)]()
[![Experiments](https://img.shields.io/badge/experiments-75%20completed-blue.svg)]()

A systematic evaluation platform for Parameter-Efficient Fine-Tuning (PEFT) of geospatial foundation models. We benchmark 5 PEFT methods across 5 modality configurations on Prithvi-100M (75 experiments total), revealing practical insights for real-world deployment.

## Key Results (EuroSAT, Overall Accuracy)

| Method | s2_full (10ch) | rgb (3ch) | rgb_sar (4ch) | gf2 (4ch) | sar_only (2ch) | Params |
|---|---|---|---|---|---|---|
| Linear Probe | 0.657 | 0.556 | 0.553 | 0.649 | 0.387 | 7.7K |
| BitFit | 0.702 | 0.608 | 0.605 | 0.701 | 0.449 | 111K |
| LoRA (r=8) | 0.658 | 0.556 | 0.553 | 0.650 | 0.388 | 155K |
| **Houlsby (d=64)** | **0.821** | **0.727** | **0.738** | **0.820** | **0.615** | **1.2M** |

**Key findings**: LoRA fails on Prithvi's fused-QKV architecture. Houlsby dominates (+16-23%). Modality selection matters more than PEFT method choice.

## Quick Start

```bash
pip install -e .
python -m pytest tests/ -v
cd ae_backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8087
```

## License

MIT License
