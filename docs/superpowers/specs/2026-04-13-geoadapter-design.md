# GeoAdapter: Modality-Aware PEFT for Geospatial Foundation Models — Design Spec

## 1. Problem Statement

Geospatial foundation models (GeoFMs) like Prithvi-100M are pre-trained on fixed band configurations (e.g., HLS 6-band). Real-world deployment faces heterogeneous inputs: 4-band commercial optical (GF-2), 2-band SAR (Sentinel-1 VV/VH), 10-band Sentinel-2, or arbitrary mixtures. Current PEFT methods (LoRA, Adapter, BitFit) all assume input modality matches pre-training and use zero-padding for mismatches — a lossy, uninformed workaround.

## 2. Proposed Solution

### 2.1 GeoAdapter Module

A lightweight, learnable input-stage adapter that maps arbitrary C_in channels to the C_pre channels expected by the frozen backbone. Three-layer design:

```
Input [B, C_in, H, W]
  |
  v
Layer 1: Channel Projection — nn.Conv2d(C_in, 6, kernel_size=1)
  |  Learnable linear mapping from arbitrary bands to pre-training band space
  v
Layer 2: Channel Attention — SE-style Squeeze-Excitation (r=2)
  |  GAP -> FC(6, 3) -> ReLU -> FC(3, 6) -> Sigmoid -> scale
  |  Learns which projected channels matter most for the backbone
  v
Layer 3: Spatial Refinement — DepthwiseConv2d(6, 6, kernel_size=3, padding=1) + BN + GELU
  |  Recalibrates local spatial correlations across mixed-resolution modalities
  v
Output [B, 6, H, W] -> Frozen Prithvi Patch Embedding
```

Total trainable parameters: ~500 (+ task head). This is 60x fewer than LoRA (r=8).

### 2.2 Geo-MLOps Platform

End-to-end MLOps framework wrapping GeoAdapter for automated experimentation:
- Data acquisition: GEE public data + local private rasters
- Preprocessing: PostGIS boundary query -> rasterio crop -> normalization -> tensor generation
- Training: Unified PEFT engine supporting all 5 methods
- Evaluation: Standard metrics (OA, F1, mAP, mIoU) on public benchmarks
- Visualization: t-SNE/UMAP embedding plots, channel attention heatmaps

## 3. Architecture

### 3.1 Code Structure

```
AlphaEarth-System/
├── geoadapter/                    # Independent Python package (core algorithms)
│   ├── adapters/
│   │   ├── base.py               # ModalityAdapter base class
│   │   ├── geo_adapter.py        # GeoAdapter (Proj + SE + DWConv)
│   │   ├── zero_pad.py           # Zero-padding baseline
│   │   ├── linear_probe.py       # Linear Probing
│   │   ├── lora.py               # LoRA (r=4,8,16)
│   │   ├── bitfit.py             # BitFit (bias-only tuning)
│   │   └── houlsby.py            # Houlsby Adapter
│   ├── models/
│   │   ├── prithvi.py            # Full Prithvi-100M loading (12-layer ViT)
│   │   └── heads.py              # Task heads (classification, multilabel, segmentation)
│   ├── data/
│   │   ├── fusion.py             # DataFusionPipeline (enhanced from ae_backend)
│   │   ├── datasets.py           # EuroSAT / BigEarthNet / DynamicEarthNet loaders
│   │   └── transforms.py         # Normalization, augmentation, channel recombination
│   ├── engine/
│   │   ├── trainer.py            # Unified training loop (all PEFT methods)
│   │   └── evaluator.py          # Unified evaluation (OA/F1/mAP/mIoU)
│   ├── bench/
│   │   ├── run_benchmark.py      # Entry point for full 25-experiment sweep
│   │   └── configs/              # YAML experiment configs
│   └── viz/
│       ├── embedding_viz.py      # t-SNE / UMAP visualization
│       └── attention_viz.py      # Channel attention weight heatmaps
│
├── notebooks/                     # Colab notebooks (run on A100)
│   ├── 01_benchmark_eurosat.ipynb
│   ├── 02_benchmark_bigearthnet.ipynb
│   ├── 03_benchmark_dynamicearthnet.ipynb
│   ├── 04_private_case_study.ipynb
│   └── 05_system_evaluation.ipynb
│
├── ae_backend/                    # Existing FastAPI platform (thin wrapper)
│   └── app/services/
│       ├── trainer.py            # Calls geoadapter.engine.trainer
│       └── data_fusion.py        # Calls geoadapter.data.fusion
│
├── ae_frontend/                   # Existing Vue 3 dashboard (unchanged)
└── pyproject.toml                 # Package definition for `pip install -e .`
```

### 3.2 Prithvi-100M Integration

Load full 12-layer ViT from HuggingFace checkpoint (`ibm-nasa-geospatial/Prithvi-100M`):
- Conv3d(6, 768, kernel_size=(1,16,16)) patch embedding — squeeze temporal dim for single-timestep
- 12 Transformer encoder layers (d=768, heads=12, mlp=3072)
- Freeze all backbone parameters
- GeoAdapter inserted before patch embedding; task head appended after GAP

### 3.3 PEFT Method Integration Points

| Method | Where it modifies | Input handling | Trainable params |
|---|---|---|---|
| Linear Probe | Task head only | Zero-pad to 6ch | ~4K |
| BitFit | All bias parameters | Zero-pad to 6ch | ~1K |
| Houlsby Adapter | Bottleneck after each ViT layer | Zero-pad to 6ch | ~50K |
| LoRA (r=8) | Q/V matrices low-rank decomposition | Zero-pad to 6ch | ~30K |
| **GeoAdapter** | **Input-stage adapter + task head** | **Adaptive mapping** | **~500 + head** |

Key comparison: all baselines use zero-padding for channel mismatch. Only GeoAdapter learns the mapping.

## 4. Experiment Design

### 4.1 Downstream Tasks

| Task | Dataset | Scale | Metric |
|---|---|---|---|
| Scene classification | EuroSAT | 27K images, 10 classes | OA, Macro F1 |
| Multi-label classification | BigEarthNet-S2 | 590K images, 19 classes | mAP |
| Semantic segmentation | DynamicEarthNet | Monthly, 7 classes | mIoU |

### 4.2 Modality Configurations

| ID | Input config | C_in | Purpose |
|---|---|---|---|
| M1 | Sentinel-2 full (B2-B12) | 10 | Superset: more bands than pre-training |
| M2 | RGB only (B4,B3,B2) | 3 | Subset: fewer bands than pre-training |
| M3 | RGB + SAR VV | 4 | Cross-modal: optical + radar mix |
| M4 | GF-2 (B,G,R,NIR) | 4 | Cross-sensor: different optical sensor |
| M5 | SAR only (VV+VH) | 2 | Extreme OOD: pure radar input |

### 4.3 Experiment Matrix

5 PEFT methods x 5 modality configs x 3 tasks = 75 experiment cells.
Each cell: 3 random seeds, report mean +/- std.
Priority: EuroSAT (all 25 combos first), then BigEarthNet (top 3 methods x 5 modalities), then DynamicEarthNet (top 3 methods x 3 modalities).
Estimated compute: 5-7 days on Colab Pro+ A100.

### 4.4 Ablation Study

GeoAdapter three-layer ablation:
- Full GeoAdapter (Proj + SE + DWConv)
- Proj + SE only (remove spatial refinement)
- Proj only (= current 1x1 Conv baseline)
- Zero-pad (no adapter at all)

### 4.5 Private Data Case Study (Qualitative)

- Load trained GeoAdapter model
- Extract 64-dim embeddings from GF-2 private imagery
- t-SNE/UMAP visualization: GeoAdapter vs zero-pad feature distributions
- Channel attention weight heatmap: what did SE learn for optical+SAR vs pure optical?

### 4.6 System Evaluation

- End-to-end latency: manual script vs Geo-MLOps pipeline
- Throughput: patches/sec (disk vs in-memory)
- Platform capability matrix: TorchGeo vs manual scripts vs Geo-MLOps

## 5. Training Configuration

- Optimizer: AdamW
- Learning rates: 1e-3 (GeoAdapter/head), 1e-4 (LoRA/Adapter internals)
- Scheduler: CosineAnnealingLR, 50 epochs
- Batch sizes: 64 (EuroSAT), 32 (BigEarthNet), 16 (DynamicEarthNet)
- Hardware: Google Colab Pro+ A100 (40GB)

## 6. Paper Structure

```
1. Introduction — heterogeneous input problem + contributions
2. Related Work — GeoFMs, PEFT in vision, RS data fusion
3. Method
   3.1 Problem formalization (C_in != C_pre)
   3.2 GeoAdapter module (three-layer architecture + math)
   3.3 Geo-MLOps platform architecture
   3.4 Unified PEFT evaluation framework
4. Experiments
   4.1 Setup (datasets, metrics, hyperparams, hardware)
   4.2 Main results: 5 PEFT x 5 modalities x 3 tasks
   4.3 Ablation: GeoAdapter layer-by-layer
   4.4 Efficiency: params vs accuracy Pareto
   4.5 Case study: private GF-2 embedding visualization
   4.6 System evaluation: Geo-MLOps latency + throughput
5. Discussion — why GeoAdapter wins on cross-modal, attention analysis
6. Conclusion
```

Target: JAG / GIScience & Remote Sensing / CVPR EarthVision Workshop

## 7. Key Figures and Tables

| Figure/Table | Content | Source |
|---|---|---|
| Fig.1 | GeoAdapter architecture diagram | Design |
| Fig.2 | Geo-MLOps platform system architecture | Design |
| Fig.3 | Main results heatmap (method x modality x task) | Colab experiments |
| Fig.4 | Params vs accuracy Pareto scatter | Colab experiments |
| Fig.5 | Three-layer ablation bar chart | Colab experiments |
| Fig.6 | t-SNE: GeoAdapter vs zero-pad on private GF-2 | Colab + local data |
| Fig.7 | Channel attention weight heatmap | Colab experiments |
| Table 1 | Main numerical results (OA/F1/mAP/mIoU) | Colab experiments |
| Table 2 | Efficiency comparison (params, time, memory) | Colab experiments |
| Table 3 | Geo-MLOps vs manual vs TorchGeo capability matrix | System analysis |

## 8. Relationship to Other Work

- **Paper 5 (completed)**: Uses frozen AlphaEarth embeddings for temporal dynamics (Route C). GeoAdapter addresses a different problem: adapting the model itself to heterogeneous inputs (Route B). Complementary, not overlapping.
- **Original AlphaEarth System**: The platform infrastructure (FastAPI + Vue 3 + GEE + OBS) is preserved. GeoAdapter replaces the demo-level STP encoder with production-grade Prithvi-100M + learned adapter.

## 9. Constraints and Risks

- **Compute**: Colab Pro+ A100 sessions have time limits. Experiment configs must support checkpointing and resumption.
- **Prithvi weights**: Must verify HuggingFace checkpoint loads correctly with Conv3d->Conv2d temporal squeeze.
- **BigEarthNet scale**: 590K images may require multi-session training. Consider subset sampling if needed.
- **No private data labels**: Private GF-2 case study is qualitative only (t-SNE). Cannot report quantitative metrics on private data.
