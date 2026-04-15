# Geo-MLOps: PEFT Benchmark Platform for Geospatial Foundation Models

[English](README_en.md) | **简体中文**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/tests-39%20passed-brightgreen.svg)]()
[![Experiments](https://img.shields.io/badge/experiments-75%20completed-blue.svg)]()

一个面向遥感基础模型的参数高效微调（PEFT）系统性评估平台。在 Prithvi-100M 上跨 5 种 PEFT 方法、5 种模态配置完成了 75 组实验，发现了若干对实际部署有指导意义的结论。

## 核心实验结果

在 EuroSAT（10 类场景分类）上的 Overall Accuracy：

| 方法 | s2_full (10ch) | rgb (3ch) | rgb_sar (4ch) | gf2 (4ch) | sar_only (2ch) | 参数量 |
|---|---|---|---|---|---|---|
| Linear Probe | 0.657 | 0.556 | 0.553 | 0.649 | 0.387 | 7.7K |
| BitFit | 0.702 | 0.608 | 0.605 | 0.701 | 0.449 | 111K |
| LoRA (r=8) | 0.658 | 0.556 | 0.553 | 0.650 | 0.388 | 155K |
| **Houlsby (d=64)** | **0.821** | **0.727** | **0.738** | **0.820** | **0.615** | **1.2M** |

**关键发现**：
1. LoRA 在 Prithvi 上完全失效（delta < 0.001），可能与 fused QKV attention 结构有关
2. Houlsby adapter 在所有模态上碾压其他方法（+16-23%）
3. Houlsby 能隐式利用 zero-pad 通道（rgb_sar > rgb），说明骨干内部适配比输入端适配更有效
4. 模态选择（10% 级差异）比 PEFT 方法选择（5% 级差异）对性能影响更大

详细分析见 [实验结果分析](docs/Experiment_Results_Analysis.md)。

## 架构

```
geoadapter/          <- 独立 Python 包（Colab 可用）
├── adapters/        <- 5 种 PEFT: Linear Probe / BitFit / LoRA / Houlsby / GeoAdapter
├── models/          <- Prithvi-100M 完整 12 层 ViT (149/149 权重加载)
├── engine/          <- 统一训练引擎 + 评估器
├── data/            <- EuroSAT 加载 + 波段选择
├── bench/           <- Benchmark Runner (YAML 配置)
└── viz/             <- t-SNE/UMAP + Attention 热力图

ae_backend/          <- FastAPI 平台（多 PEFT 方法切换）
ae_frontend/         <- Vue 3 + ECharts 实时训练监控
notebooks/           <- Colab A100 实验 Notebook
results/             <- 75 组实验原始数据 (JSON)
```

## 快速开始

```bash
pip install -e .
python -m pytest tests/ -v          # 39 tests, ~15s
cd ae_backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8087
```

## 文档

- [实验结果分析](docs/Experiment_Results_Analysis.md) — 75 组实验的完整数据和发现
- [AlphaEarth vs Prithvi 技术对比](docs/AlphaEarth_vs_Prithvi_Technical_Analysis.md)
- [系统验证指南](docs/GeoAdapter_Verification_Guide.md)

## 许可证

MIT License
