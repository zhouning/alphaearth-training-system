# GeoAdapter + Geo-MLOps Platform

**English** | [简体中文](README.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/tests-37%20passed-brightgreen.svg)]()

**GeoAdapter** is a modality-aware Parameter-Efficient Fine-Tuning (PEFT) framework for geospatial foundation models. It solves a practical problem: models like Prithvi-100M are pre-trained on fixed 6-band HLS data, but real-world inputs are heterogeneous — 4-band commercial optical (GF-2), 2-band SAR (Sentinel-1), 10-band Sentinel-2, or arbitrary combinations.

Core contribution: a ~150-parameter three-layer adapter (Channel Projection + SE Attention + Spatial Refinement) inserted before the frozen Prithvi backbone, enabling the model to accept arbitrary channel combinations without degradation. The companion Geo-MLOps platform provides end-to-end automation from data fusion to training monitoring to model registry.

## Quick Start

```bash
pip install -e .
python -m pytest tests/ -v          # 37 tests, ~12s
cd ae_backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8087
```

## Core Usage

```python
from geoadapter.models.prithvi import PrithviBackbone
from geoadapter.adapters.geo_adapter import GeoAdapter
from geoadapter.models.heads import ClassificationHead
from geoadapter.engine.trainer import PEFTTrainer

backbone = PrithviBackbone(pretrained=True, checkpoint_path="weights/Prithvi_100M.pt")
adapter = GeoAdapter(in_channels=4, out_channels=6)  # 4-band GF-2 -> 6-band Prithvi
head = ClassificationHead(in_dim=768, num_classes=10)
trainer = PEFTTrainer(backbone, adapter, head, lr=1e-3, device="cuda")
```

## Benchmark

5 PEFT methods x 5 modality configs x 3 downstream tasks = 75 experiments.

```bash
python -m geoadapter.bench.run_benchmark --config geoadapter/bench/configs/eurosat_default.yaml --dry-run
```

## License

MIT License
