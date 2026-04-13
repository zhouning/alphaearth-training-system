# GeoAdapter + Geo-MLOps Platform

[English](README_en.md) | **简体中文**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/tests-37%20passed-brightgreen.svg)]()

**GeoAdapter** 是一个面向遥感基础模型的模态感知参数高效微调（PEFT）框架。它解决了一个实际问题：Prithvi-100M 等基础模型预训练在固定的 6 波段 HLS 数据上，但真实部署场景的输入是异构的——4 波段商业光学（GF-2）、2 波段 SAR（Sentinel-1）、10 波段 Sentinel-2，或任意组合。

核心贡献：一个仅 ~150 参数的三层适配器（Channel Projection + SE Attention + Spatial Refinement），插在冻结的 Prithvi 骨干前端，让模型接受任意通道组合而不退化。配套的 Geo-MLOps 平台提供从数据融合到训练监控到模型注册的端到端自动化。

## 架构

```
geoadapter/          ← 独立 Python 包（零 FastAPI 依赖，Colab 可用）
├── adapters/        ← GeoAdapter + ZeroPad + LoRA + BitFit + Houlsby
├── models/          ← Prithvi-100M 完整 12 层 ViT 骨干 + 任务 Heads
├── engine/          ← 统一 PEFT 训练引擎 + 评估器
├── data/            ← 数据集加载 + 波段选择 + 归一化
├── bench/           ← Benchmark Runner（5 方法 × 5 模态 × 3 任务）
└── viz/             ← t-SNE/UMAP + Channel Attention 热力图

ae_backend/          ← FastAPI 平台（调用 geoadapter 包）
ae_frontend/         ← Vue 3 + ECharts 实时训练监控大屏
notebooks/           ← Colab Pro+ A100 实验 Notebook
```

## 快速开始

```bash
# 安装 geoadapter 包
pip install -e .

# 运行测试（37 个，约 12 秒）
python -m pytest tests/ -v

# 启动平台
cd ae_backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8087
```

## GeoAdapter 核心用法

```python
from geoadapter.models.prithvi import PrithviBackbone
from geoadapter.adapters.geo_adapter import GeoAdapter
from geoadapter.models.heads import ClassificationHead
from geoadapter.engine.trainer import PEFTTrainer

# 冻结的 Prithvi-100M（86M 参数，12 层 ViT）
backbone = PrithviBackbone(pretrained=True, checkpoint_path="weights/Prithvi_100M.pt")

# GeoAdapter：4 波段 GF-2 → 6 波段 Prithvi 输入（仅 147 参数）
adapter = GeoAdapter(in_channels=4, out_channels=6)

# 下游任务 Head
head = ClassificationHead(in_dim=768, num_classes=10)

# 统一训练（支持 CosineAnnealingLR + 差异化学习率）
trainer = PEFTTrainer(backbone, adapter, head, lr=1e-3, device="cuda")
loss = trainer.train_step(images, labels)
```

## Benchmark 实验

5 种 PEFT 方法 × 5 种模态配置 × 3 个下游任务 = 75 组实验：

| 方法 | 可训练参数 | 输入适配 |
|---|---|---|
| Linear Probe | ~7.7K (head only) | Zero-pad |
| BitFit | ~103K (bias only) | Zero-pad |
| LoRA (r=8) | ~147K | Zero-pad |
| Houlsby (d=64) | ~1.19M | Zero-pad |
| **GeoAdapter** | **~150 + head** | **自适应映射** |

```bash
# Dry-run 查看实验矩阵
python -m geoadapter.bench.run_benchmark --config geoadapter/bench/configs/eurosat_default.yaml --dry-run

# 在 Colab A100 上运行
python -m geoadapter.bench.run_benchmark --config geoadapter/bench/configs/eurosat_default.yaml --epochs 50
```

## 平台功能

三个页面覆盖完整 MLOps 流程：

- **数据源与预处理**：行政区搜索 → 卫星选择 → GEE 下载 + 本地影像融合 → 128×128 张量切片
- **训练监控驾驶舱**：WebSocket 实时 Loss 曲线 + PCA 特征空间演化 + GPU 指标
- **模型资产库**：评测（KMeans + Silhouette Score）→ 激活默认模型 → OBS 云端同步

## 技术栈

- **核心框架**：PyTorch 2.x + timm
- **基础模型**：Prithvi-100M (IBM/NASA, Apache 2.0)
- **后端**：FastAPI + SQLAlchemy + PostgreSQL/PostGIS
- **前端**：Vue 3 + Tailwind CSS + ECharts
- **GIS**：Rasterio + GeoPandas + Google Earth Engine
- **云存储**：华为云 OBS

## 文档

- [系统验证指南](docs/GeoAdapter_Verification_Guide.md)
- [AlphaEarth vs Prithvi 技术对比](docs/AlphaEarth_vs_Prithvi_Technical_Analysis.md)
- [设计规格](docs/superpowers/specs/2026-04-13-geoadapter-design.md)
- [实施计划](docs/superpowers/plans/2026-04-13-geoadapter-implementation.md)

## 许可证

MIT License
