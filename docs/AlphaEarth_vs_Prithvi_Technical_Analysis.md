# AlphaEarth Foundations vs Prithvi-100M: Technical Comparison & Ecosystem Analysis

## 1. Architecture Comparison

### 1.1 Prithvi-100M (IBM/NASA)

**Paper**: [arXiv:2310.18660](https://arxiv.org/abs/2310.18660)

**Architecture**: Standard Temporal ViT + Masked AutoEncoder (MAE)

| Parameter | Value |
|---|---|
| Model type | Encoder-only ViT with MAE pre-training |
| Parameters | ~100M (encoder 12 layers + decoder 8 layers) |
| embed_dim | 768 |
| depth (encoder) | 12 |
| num_heads | 12 |
| decoder_embed_dim | 512 |
| decoder_depth | 8 |
| decoder_num_heads | 16 |
| mlp_ratio | 4 |
| patch_size | (1, 16, 16) — Conv3d with temporal tubelet |
| img_size | 224 x 224 |
| in_chans | 6 (Blue, Green, Red, Narrow NIR, SWIR1, SWIR2) |
| num_frames | 3 (temporal) |
| mask_ratio | 0.75 |

**Pre-training methodology**:
- Self-supervised MAE: mask 75% of 3D patches, reconstruct pixels
- Single MSE reconstruction loss
- Training data: Contiguous US HLS (Harmonized Landsat Sentinel-2) dataset only
- Single-source optical data (no SAR, no elevation, no climate)

**Key characteristics**:
- All patches processed at single resolution (14x14 = 196 tokens per frame)
- 3D positional embedding for spatial + temporal dimensions
- Outputs 768-dim token features — requires task head for downstream use
- No geometric constraints on embedding space
- Weights fully open-source (Apache 2.0) via HuggingFace

### 1.2 AlphaEarth Foundations (Google DeepMind)

**Paper**: [arXiv:2507.22291](https://arxiv.org/abs/2507.22291)

**Architecture**: Custom STP (Space-Time-Precision) multi-resolution encoder

| Parameter | Value |
|---|---|
| Model type | Custom multi-resolution pyramid (NOT standard ViT) |
| Parameters | ~480M (selected) / ~1B (largest variant) |
| Embedding dimension | 64 (8 float32 values) |
| Spatial resolution | 10m globally |
| Embedding constraint | L2-normalized on S^63 unit hypersphere |
| Quantization | 8-bit (4x compression, negligible quality loss) |

**STP Architecture — three simultaneous operators per block**:

| Operator | Resolution | Mechanism | Purpose |
|---|---|---|---|
| Space | 1/16 L | ViT-like spatial self-attention | Global context |
| Time | 1/8 L | Time-axial self-attention + sinusoidal timecodes | Temporal dynamics |
| Precision | 1/2 L | 3x3 convolutions | Local detail |

Each block terminates with learned Laplacian pyramid rescaling, enabling cross-resolution state passing. Final output at precision operator resolution (1/2 L).

**Pre-training methodology**:
- Teacher-Student framework with shared parameters
  - Teacher video embedding model with implicit decoders
  - Student video embedding model (identical architecture)
  - Text alignment model
- **4 loss functions** (vs Prithvi's 1):
  1. Reconstruction: per-source conditional decoding (sensor-specific)
  2. Batch uniformity: minimizes absolute dot products between batch-rotated embeddings
  3. Consistency: temporal/spatial coherence across observations
  4. Text-contrastive: aligns satellite embeddings with geotagged Wikipedia articles

**Training data**: ~3 billion observations across ~1.1% of Earth's land surface, from **9+ gridded datasets**:

| Type | Dataset | Bands/Channels |
|---|---|---|
| Optical | Sentinel-2 L1C | 5 bands |
| Optical/Thermal | Landsat 8/9 L1C | 7 bands |
| C-band SAR | Sentinel-1 GRD | 5 channels |
| L-band SAR | ALOS PALSAR-2 | 3 channels |
| Elevation | Copernicus DEM GLO-30 | 1 |
| LiDAR | GEDI L2A | 1 |
| Climate | ERA5-Land monthly | multiple |
| Gravity | GRACE monthly | multiple |
| Land cover | NLCD 2019/2021 | 1 |
| Text | Wikipedia geocoded articles | N/A |
| Text | GBIF research-grade observations | N/A |

**Key characteristics**:
- Multi-resolution simultaneous pathways (1/16L, 1/8L, 1/2L) vs Prithvi's single resolution
- 64-dim output constrained to unit hypersphere — directly usable for k-NN/linear probe
- Handles 9+ heterogeneous data sources natively (no channel mismatch problem)
- Produces pre-computed global annual embedding datasets (2017-2024) via GEE
- Model weights NOT open-sourced — only embeddings are public

## 2. Side-by-Side Comparison

| Dimension | Prithvi-100M | AlphaEarth Foundations |
|---|---|---|
| **Architecture** | Standard ViT + MAE (single resolution) | Custom STP multi-resolution pyramid (NOT ViT) |
| **Parameters** | 100M | 480M - 1B |
| **Pre-training** | MAE self-supervised (mask & reconstruct pixels) | Teacher-Student distillation + 4-loss joint training |
| **Loss functions** | 1 (MSE reconstruction) | 4 (reconstruction + uniformity + consistency + text-contrastive) |
| **Training data** | US-only HLS (Landsat+S2), single optical source | Global 3B observations, 9+ sources (optical+SAR+LiDAR+climate+gravity+text) |
| **Input format** | Fixed 6 bands, (B,6,T,224,224) | 9+ heterogeneous sources, any combination |
| **Temporal** | 3D positional embedding, T frames stacked | Dedicated Time Operator: time-axial self-attention + sinusoidal timecodes |
| **Multi-resolution** | None — all patches equal | Three-path pyramid with Laplacian rescaling |
| **Output** | 768-dim token sequence (needs task head) | 64-dim L2-normalized embedding (directly usable) |
| **Embedding constraint** | None | Unit hypersphere S^63 (von Mises-Fisher) |
| **Text alignment** | None | Yes — geotagged Wikipedia contrastive learning |
| **Weights open** | Yes (Apache 2.0, HuggingFace) | No — only pre-computed embeddings via GEE |
| **Fine-tunable** | Yes — load weights, add head, PEFT/full tune | No — can only consume pre-computed embeddings |
| **Global coverage** | No pre-computed results | 2017-2024 annual 10m global embedding dataset |
| **Geographic bias** | US-centric (trained on CONUS only) | Global (but 1.1% land surface) |

## 3. Has Anyone Reproduced AlphaEarth?

### 3.1 Reproduction attempts

**Only one known attempt**: [Brayden-Zhang/alphaearth-foundations](https://github.com/Brayden-Zhang/alphaearth-foundations) (207 GitHub stars)

This unofficial PyTorch implementation:
- Implements the STP three-path architecture
- Implements the Teacher-Student-Text framework
- **BUT** trained on only 1/40th of the Landsat subset
- **BUT** used batch size 16 (paper uses 256)
- **BUT** far from reaching paper's training scale (100K steps)

**Verdict**: Architecture verification only, not a true reproduction. No other team has attempted a full reproduction.

### 3.2 Why no one has reproduced it

**Barrier 1 — Data**: 3 billion observations from 9+ sources requires PB-scale storage and massive GEE compute quotas. Academic teams cannot access this at scale.

**Barrier 2 — Compute**: 480M-1B parameter model trained at 256 batch size x 100K steps requires large GPU/TPU clusters. Google DeepMind-level infrastructure. Not feasible with a few A100s.

**Barrier 3 — Architecture complexity**: STP three-path pyramid + Learned Laplacian Pyramid + Teacher-Student distillation + 4 losses = far more complex than standard ViT-MAE. Even if implemented, without sufficient data and compute, the trained model won't match Google's results.

### 3.3 Papers using AlphaEarth embeddings (without reproducing the model)

Instead of reproducing the model, researchers are consuming the pre-computed embedding dataset:

- [Wetland Vegetation Mapping with AlphaEarth Embeddings](https://www.mdpi.com/2072-4292/18/2/293) — uses embeddings for wetland classification, comparable accuracy to traditional methods with minimal preprocessing
- [Land Cover Classification with AlphaEarth Embeddings](https://www.preprints.org/manuscript/202511.2172) — evaluates pixel- and object-based classification using the embedding dataset in GEE
- Our own Paper 5 (world model) — uses frozen AlphaEarth embeddings as state space for LULC change prediction

**Pattern**: The community treats AlphaEarth as an "embedding service" — consume the outputs, don't replicate the model.

## 4. Is AlphaEarth's Architecture Revolutionary?

### 4.1 What IS genuinely novel

| Innovation | Why it matters |
|---|---|
| Multi-resolution STP three-path processing (1/16L + 1/8L + 1/2L) | Standard ViT processes all patches at one resolution. STP captures macro context and pixel detail simultaneously. |
| S^63 hypersphere embedding constraint | Standard embeddings have no geometric structure. AlphaEarth's embeddings are directly comparable via cosine distance without additional training. |
| Native 9-source heterogeneous fusion | Prithvi/SatMAE can only handle fixed-band inputs. AlphaEarth handles optical+SAR+LiDAR+climate natively at the architecture level. |
| Vision-language alignment | First geospatial FM to do satellite-text contrastive learning, enabling semantic interpretability. |
| Global pre-computed embedding dataset | Other models release weights and expect users to run inference. AlphaEarth releases the results directly, lowering the barrier to zero. |

### 4.2 What is NOT revolutionary (just scale)

| Aspect | Assessment |
|---|---|
| MAE-style reconstruction loss | Standard, used by SatMAE/Prithvi |
| Teacher-Student distillation | Standard in SSL (BYOL, DINO) |
| Uniformity loss | Known technique from contrastive learning literature |
| Transformer backbone | Ubiquitous |

### 4.3 Why traditional methods still dominate

Despite AlphaEarth's innovations, most real-world projects still use traditional approaches (ResNet/UNet + supervised training):

1. **When you have labels, you don't need a foundation model**. Most operational remote sensing tasks (e.g., crop classification in a known region) have sufficient labeled data. A simple UNet trained on 10K labeled patches often beats a foundation model with a linear probe.

2. **Inference cost**. Running a 480M model per-pixel at 10m resolution globally is expensive. A lightweight CNN is orders of magnitude cheaper.

3. **Foundation model advantage appears in specific scenarios**: few-shot learning (very few labels), cross-domain transfer (train on US, deploy in China), and embedding-based retrieval (find similar locations). If you don't need these, traditional methods suffice.

## 5. Implications for Our Project

### 5.1 Why we chose Prithvi over AlphaEarth for GeoAdapter

| Reason | Detail |
|---|---|
| Prithvi weights are open-source | Can load, freeze, and insert GeoAdapter in front. AlphaEarth weights are closed. |
| Prithvi has the "fixed input" problem | 6-band HLS input lock-in = exactly the problem GeoAdapter solves |
| AlphaEarth has no input problem | 9-source native fusion means GeoAdapter is irrelevant to AlphaEarth |
| Prithvi is fine-tunable | Supports PEFT, which is the basis of our benchmark |
| AlphaEarth is consumption-only | Can only use pre-computed embeddings, cannot fine-tune |

### 5.2 How our two papers cover the ecosystem

| Paper | Foundation Model | Approach | Contribution |
|---|---|---|---|
| Paper 5 (world model) | AlphaEarth (frozen embeddings) | Consume embedding dataset, learn temporal dynamics on top | Application layer — "what can you do with FM embeddings?" |
| GeoAdapter paper | Prithvi-100M (frozen backbone) | Adapt the model itself to accept heterogeneous inputs via PEFT | Infrastructure layer — "how to make FMs work with real-world data?" |

Together, these two papers provide a complete story: one shows how to efficiently use closed FM outputs (AlphaEarth), the other shows how to adapt open FM internals to real deployment constraints (Prithvi + GeoAdapter). This is a strong, non-overlapping portfolio.
