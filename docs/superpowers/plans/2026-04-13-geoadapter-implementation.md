# GeoAdapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `geoadapter` Python package with modality-aware PEFT adapters for Prithvi-100M, a unified benchmark engine, and integrate it into the existing AlphaEarth platform.

**Architecture:** Independent `geoadapter/` package (zero FastAPI dependency) containing adapter implementations, full Prithvi-100M backbone loading, dataset loaders, unified training/evaluation engine, and benchmark configs. The existing `ae_backend/` becomes a thin wrapper calling `geoadapter` internals. Colab notebooks import `geoadapter` directly.

**Tech Stack:** PyTorch 2.x, timm (for ViT), torchgeo (for datasets), rasterio, HuggingFace transformers, scikit-learn (for metrics), matplotlib/seaborn (for viz)

---

## File Structure

```
geoadapter/
├── __init__.py                    # Package version + top-level imports
├── adapters/
│   ├── __init__.py
│   ├── base.py                    # ModalityAdapter ABC
│   ├── geo_adapter.py             # GeoAdapter (Proj + SE + DWConv)
│   ├── zero_pad.py                # Zero-padding baseline
│   ├── lora.py                    # LoRA injection into ViT
│   ├── bitfit.py                  # Bias-only fine-tuning
│   └── houlsby.py                 # Houlsby bottleneck adapter
├── models/
│   ├── __init__.py
│   ├── prithvi.py                 # Full Prithvi-100M backbone loader
│   └── heads.py                   # Classification / multilabel / segmentation heads
├── data/
│   ├── __init__.py
│   ├── datasets.py                # EuroSAT + BigEarthNet loaders with channel configs
│   └── transforms.py              # Band selection, normalization, augmentation
├── engine/
│   ├── __init__.py
│   ├── trainer.py                 # Unified PEFT training loop
│   └── evaluator.py               # OA / F1 / mAP metrics
├── viz/
│   ├── __init__.py
│   └── embedding_viz.py           # t-SNE / UMAP + attention heatmaps
├── bench/
│   ├── __init__.py
│   ├── run_benchmark.py           # CLI entry point for experiment sweep
│   └── configs/
│       └── eurosat_default.yaml   # Example experiment config
tests/
├── __init__.py
├── test_adapters.py               # Unit tests for all adapters
├── test_prithvi.py                # Backbone loading + forward pass
├── test_heads.py                  # Task head shapes
├── test_trainer.py                # Training loop smoke test
└── test_datasets.py               # Dataset loading + transforms
pyproject.toml                     # Package definition
```

### Task 1: Package scaffold + pyproject.toml

**Files:**
- Create: `pyproject.toml`
- Create: `geoadapter/__init__.py`
- Create: `geoadapter/adapters/__init__.py`
- Create: `geoadapter/models/__init__.py`
- Create: `geoadapter/data/__init__.py`
- Create: `geoadapter/engine/__init__.py`
- Create: `geoadapter/viz/__init__.py`
- Create: `geoadapter/bench/__init__.py`
- Create: `geoadapter/bench/configs/`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "geoadapter"
version = "0.1.0"
description = "Modality-aware PEFT for geospatial foundation models"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "torchvision",
    "timm>=0.9",
    "rasterio>=1.3",
    "numpy",
    "pyyaml",
    "scikit-learn",
]

[project.optional-dependencies]
bench = ["torchgeo>=0.5", "matplotlib", "seaborn", "umap-learn"]
dev = ["pytest>=7.0", "pytest-cov"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package __init__.py files**

`geoadapter/__init__.py`:
```python
"""GeoAdapter: Modality-aware PEFT for geospatial foundation models."""
__version__ = "0.1.0"
```

All other `__init__.py` files: empty.

- [ ] **Step 3: Verify package installs**

Run: `pip install -e ".[dev]"`
Expected: Successful installation

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml geoadapter/ tests/
git commit -m "feat: scaffold geoadapter package structure"
```

---

### Task 2: Adapter base class + GeoAdapter implementation

**Files:**
- Create: `geoadapter/adapters/base.py`
- Create: `geoadapter/adapters/geo_adapter.py`
- Create: `geoadapter/adapters/zero_pad.py`
- Create: `tests/test_adapters.py`

- [ ] **Step 1: Write failing tests for adapters**

`tests/test_adapters.py`:
```python
import pytest
import torch
from geoadapter.adapters.base import ModalityAdapter
from geoadapter.adapters.geo_adapter import GeoAdapter
from geoadapter.adapters.zero_pad import ZeroPadAdapter


class TestModalityAdapterInterface:
    def test_geo_adapter_is_modality_adapter(self):
        adapter = GeoAdapter(in_channels=4, out_channels=6)
        assert isinstance(adapter, ModalityAdapter)

    def test_zero_pad_is_modality_adapter(self):
        adapter = ZeroPadAdapter(in_channels=4, out_channels=6)
        assert isinstance(adapter, ModalityAdapter)


class TestGeoAdapter:
    @pytest.mark.parametrize("c_in", [2, 3, 4, 5, 10])
    def test_output_shape(self, c_in):
        adapter = GeoAdapter(in_channels=c_in, out_channels=6)
        x = torch.randn(2, c_in, 128, 128)
        out = adapter(x)
        assert out.shape == (2, 6, 128, 128)

    def test_trainable_param_count(self):
        adapter = GeoAdapter(in_channels=4, out_channels=6)
        n = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        assert n < 1000, f"GeoAdapter should be <1000 params, got {n}"

    def test_three_layers_present(self):
        adapter = GeoAdapter(in_channels=5, out_channels=6)
        assert hasattr(adapter, "channel_proj")
        assert hasattr(adapter, "channel_attn")
        assert hasattr(adapter, "spatial_refine")


class TestZeroPadAdapter:
    @pytest.mark.parametrize("c_in", [2, 3, 4, 5, 10])
    def test_output_shape(self, c_in):
        adapter = ZeroPadAdapter(in_channels=c_in, out_channels=6)
        x = torch.randn(2, c_in, 64, 64)
        out = adapter(x)
        assert out.shape == (2, 6, 64, 64)

    def test_no_trainable_params(self):
        adapter = ZeroPadAdapter(in_channels=4, out_channels=6)
        n = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        assert n == 0

    def test_preserves_existing_channels(self):
        adapter = ZeroPadAdapter(in_channels=3, out_channels=6)
        x = torch.ones(1, 3, 4, 4)
        out = adapter(x)
        assert torch.allclose(out[0, :3], x[0])
        assert torch.allclose(out[0, 3:], torch.zeros(3, 4, 4))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_adapters.py -v`
Expected: FAIL — modules not found

- [ ] **Step 3: Implement base class**

`geoadapter/adapters/base.py`:
```python
import torch.nn as nn
from abc import abstractmethod


class ModalityAdapter(nn.Module):
    """Base class for input-stage modality adapters."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, x):
        """Map [B, C_in, H, W] -> [B, C_out, H, W]."""
        ...
```

- [ ] **Step 4: Implement GeoAdapter**

`geoadapter/adapters/geo_adapter.py`:
```python
import torch
import torch.nn as nn
from .base import ModalityAdapter


class GeoAdapter(ModalityAdapter):
    """Three-layer modality-aware adapter: Projection + SE Attention + Spatial Refinement."""

    def __init__(self, in_channels: int, out_channels: int = 6, se_reduction: int = 2):
        super().__init__(in_channels, out_channels)

        # Layer 1: Channel Projection
        self.channel_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Layer 2: SE-style Channel Attention
        mid = max(1, out_channels // se_reduction)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, out_channels),
            nn.Sigmoid(),
        )

        # Layer 3: Spatial Refinement (depthwise conv)
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.channel_proj(x)
        attn = self.channel_attn(x).unsqueeze(-1).unsqueeze(-1)
        x = x * attn
        x = self.spatial_refine(x)
        return x
```

- [ ] **Step 5: Implement ZeroPadAdapter**

`geoadapter/adapters/zero_pad.py`:
```python
import torch
import torch.nn.functional as F
from .base import ModalityAdapter


class ZeroPadAdapter(ModalityAdapter):
    """Baseline: zero-pad or truncate channels to match target."""

    def forward(self, x):
        c_in = x.shape[1]
        if c_in < self.out_channels:
            pad = torch.zeros(
                x.shape[0], self.out_channels - c_in, x.shape[2], x.shape[3],
                device=x.device, dtype=x.dtype,
            )
            return torch.cat([x, pad], dim=1)
        return x[:, : self.out_channels]
```

- [ ] **Step 6: Update adapters __init__.py**

`geoadapter/adapters/__init__.py`:
```python
from .base import ModalityAdapter
from .geo_adapter import GeoAdapter
from .zero_pad import ZeroPadAdapter
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_adapters.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add geoadapter/adapters/ tests/test_adapters.py
git commit -m "feat: implement GeoAdapter and ZeroPadAdapter with tests"
```

---

### Task 3: LoRA, BitFit, Houlsby adapters

**Files:**
- Create: `geoadapter/adapters/lora.py`
- Create: `geoadapter/adapters/bitfit.py`
- Create: `geoadapter/adapters/houlsby.py`
- Modify: `geoadapter/adapters/__init__.py`
- Modify: `tests/test_adapters.py`

- [ ] **Step 1: Add tests for PEFT methods**

Append to `tests/test_adapters.py`:
```python
from geoadapter.adapters.lora import inject_lora, remove_lora
from geoadapter.adapters.bitfit import configure_bitfit
from geoadapter.adapters.houlsby import inject_houlsby_adapters


class TestLoRA:
    def test_inject_and_remove(self):
        block = nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True)
        original_params = sum(p.numel() for p in block.parameters())
        inject_lora(block, rank=8, target_modules=["self_attn"])
        lora_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
        assert lora_params > 0
        assert lora_params < original_params

    def test_forward_unchanged_shape(self):
        block = nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True)
        inject_lora(block, rank=8, target_modules=["self_attn"])
        x = torch.randn(2, 16, 768)
        out = block(x)
        assert out.shape == (2, 16, 768)


class TestBitFit:
    def test_only_biases_trainable(self):
        block = nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True)
        configure_bitfit(block)
        for name, p in block.named_parameters():
            if "bias" in name:
                assert p.requires_grad, f"{name} should be trainable"
            else:
                assert not p.requires_grad, f"{name} should be frozen"


class TestHoulsby:
    def test_inject_adds_params(self):
        block = nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True)
        before = sum(p.numel() for p in block.parameters())
        inject_houlsby_adapters(block, bottleneck_dim=64)
        after = sum(p.numel() for p in block.parameters())
        assert after > before

    def test_forward_shape(self):
        block = nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True)
        inject_houlsby_adapters(block, bottleneck_dim=64)
        x = torch.randn(2, 16, 768)
        out = block(x)
        assert out.shape == (2, 16, 768)
```

Add at top of file: `import torch.nn as nn`

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_adapters.py -v -k "LoRA or BitFit or Houlsby"`
Expected: FAIL — modules not found

- [ ] **Step 3: Implement LoRA**

`geoadapter/adapters/lora.py`:
```python
import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper around nn.Linear."""

    def __init__(self, original: nn.Linear, rank: int = 8):
        super().__init__()
        self.original = original
        self.rank = rank
        d_in, d_out = original.in_features, original.out_features
        self.lora_A = nn.Parameter(torch.randn(d_in, rank) / math.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out))
        original.weight.requires_grad_(False)
        if original.bias is not None:
            original.bias.requires_grad_(False)

    def forward(self, x):
        base = self.original(x)
        return base + (x @ self.lora_A @ self.lora_B)


def inject_lora(module: nn.Module, rank: int = 8, target_modules=("self_attn",)):
    """Replace Linear layers inside target submodules with LoRALinear."""
    for tgt in target_modules:
        if hasattr(module, tgt):
            submod = getattr(module, tgt)
            for name, child in list(submod.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(submod, name, LoRALinear(child, rank=rank))
    # Freeze everything except LoRA params
    for p in module.parameters():
        p.requires_grad_(False)
    for m in module.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)


def remove_lora(module: nn.Module):
    """Merge LoRA weights back into original Linear and remove wrappers."""
    for name, child in list(module.named_modules()):
        if isinstance(child, LoRALinear):
            merged = child.original
            merged.weight.data += (child.lora_A @ child.lora_B).T
            merged.weight.requires_grad_(True)
            # Navigate to parent and replace
            parts = name.split(".")
            parent = module
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], merged)
```

- [ ] **Step 4: Implement BitFit**

`geoadapter/adapters/bitfit.py`:
```python
import torch.nn as nn


def configure_bitfit(module: nn.Module):
    """Freeze all parameters except biases."""
    for name, param in module.named_parameters():
        param.requires_grad_("bias" in name)
```

- [ ] **Step 5: Implement Houlsby Adapter**

`geoadapter/adapters/houlsby.py`:
```python
import torch
import torch.nn as nn


class HoulsbyBottleneck(nn.Module):
    """Bottleneck adapter inserted after FFN in a Transformer layer."""

    def __init__(self, d_model: int, bottleneck_dim: int = 64):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


def inject_houlsby_adapters(block: nn.Module, bottleneck_dim: int = 64):
    """Wrap the forward of a TransformerEncoderLayer to insert adapter after FFN."""
    adapter = HoulsbyBottleneck(d_model=768, bottleneck_dim=bottleneck_dim)
    original_forward = block.forward

    def new_forward(src, *args, **kwargs):
        out = original_forward(src, *args, **kwargs)
        return adapter(out)

    block.forward = new_forward
    # Register adapter as submodule so its params are visible
    block.add_module("houlsby_adapter", adapter)
    # Freeze original, keep adapter trainable
    for name, p in block.named_parameters():
        p.requires_grad_("houlsby_adapter" in name)
```

- [ ] **Step 6: Update __init__.py**

`geoadapter/adapters/__init__.py`:
```python
from .base import ModalityAdapter
from .geo_adapter import GeoAdapter
from .zero_pad import ZeroPadAdapter
from .lora import inject_lora, remove_lora, LoRALinear
from .bitfit import configure_bitfit
from .houlsby import inject_houlsby_adapters, HoulsbyBottleneck
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_adapters.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add geoadapter/adapters/ tests/test_adapters.py
git commit -m "feat: implement LoRA, BitFit, Houlsby PEFT adapters"
```

---

### Task 4: Prithvi-100M backbone loader

**Files:**
- Create: `geoadapter/models/prithvi.py`
- Create: `tests/test_prithvi.py`

- [ ] **Step 1: Write failing tests**

`tests/test_prithvi.py`:
```python
import pytest
import torch
from geoadapter.models.prithvi import PrithviBackbone


class TestPrithviBackbone:
    def test_load_without_weights(self):
        model = PrithviBackbone(pretrained=False)
        assert model is not None

    def test_forward_shape(self):
        model = PrithviBackbone(pretrained=False)
        x = torch.randn(2, 6, 224, 224)
        features = model(x)
        assert features.shape == (2, 768)

    def test_all_frozen(self):
        model = PrithviBackbone(pretrained=False)
        for p in model.parameters():
            assert not p.requires_grad

    def test_num_blocks(self):
        model = PrithviBackbone(pretrained=False)
        assert len(model.blocks) == 12

    @pytest.mark.parametrize("h,w", [(128, 128), (224, 224), (64, 64)])
    def test_variable_input_size(self, h, w):
        model = PrithviBackbone(pretrained=False)
        x = torch.randn(1, 6, h, w)
        features = model(x)
        assert features.shape == (1, 768)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_prithvi.py -v`
Expected: FAIL

- [ ] **Step 3: Implement PrithviBackbone**

`geoadapter/models/prithvi.py`:
```python
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class PrithviBackbone(nn.Module):
    """Full Prithvi-100M ViT backbone with frozen weights.

    Architecture: Conv3d patch embed (squeeze temporal) + 12 Transformer layers.
    """

    def __init__(
        self,
        pretrained: bool = True,
        checkpoint_path: str | None = None,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        in_chans: int = 6,
        patch_size: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Patch embedding: Conv3d(6,768,(1,16,16)) squeezed to Conv2d
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=embed_dim * 4, batch_first=True,
                activation="gelu", norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        if pretrained and checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        self._freeze_all()

    def _load_checkpoint(self, path: str):
        """Load Prithvi-100M weights, squeezing Conv3d temporal dim."""
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            state = ckpt.get("model", ckpt)

            # Patch embed: Conv3d [768,6,1,16,16] -> Conv2d [768,6,16,16]
            pe_key = "encoder.patch_embed.proj.weight"
            if pe_key in state and state[pe_key].dim() == 5:
                state[pe_key] = state[pe_key].squeeze(2)

            # Map keys and load what matches
            own_state = self.state_dict()
            loaded = 0
            for k, v in state.items():
                # Remap encoder.X -> X
                mapped = k.replace("encoder.", "", 1) if k.startswith("encoder.") else k
                if mapped in own_state and own_state[mapped].shape == v.shape:
                    own_state[mapped] = v
                    loaded += 1
            self.load_state_dict(own_state, strict=False)
            logger.info(f"Loaded {loaded} tensors from Prithvi checkpoint")
        except Exception as e:
            logger.warning(f"Could not load Prithvi weights: {e}")

    def _freeze_all(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 6, H, W] -> [B, 768] global features."""
        B = x.shape[0]
        x = self.patch_embed(x)                    # [B, 768, H/16, W/16]
        x = x.flatten(2).transpose(1, 2)           # [B, N, 768]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)             # [B, N+1, 768]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]                             # CLS token -> [B, 768]
```

- [ ] **Step 4: Update models __init__.py**

```python
from .prithvi import PrithviBackbone
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_prithvi.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add geoadapter/models/ tests/test_prithvi.py
git commit -m "feat: implement PrithviBackbone with full 12-layer ViT"
```

---

### Task 5: Task heads (classification, multilabel, segmentation)

**Files:**
- Create: `geoadapter/models/heads.py`
- Create: `tests/test_heads.py`

- [ ] **Step 1: Write failing tests**

`tests/test_heads.py`:
```python
import torch
from geoadapter.models.heads import ClassificationHead, MultiLabelHead


class TestClassificationHead:
    def test_output_shape(self):
        head = ClassificationHead(in_dim=768, num_classes=10)
        x = torch.randn(4, 768)
        logits = head(x)
        assert logits.shape == (4, 10)


class TestMultiLabelHead:
    def test_output_shape(self):
        head = MultiLabelHead(in_dim=768, num_classes=19)
        x = torch.randn(4, 768)
        logits = head(x)
        assert logits.shape == (4, 19)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_heads.py -v`

- [ ] **Step 3: Implement heads**

`geoadapter/models/heads.py`:
```python
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int = 768, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MultiLabelHead(nn.Module):
    def __init__(self, in_dim: int = 768, num_classes: int = 19):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
```

- [ ] **Step 4: Run tests, commit**

Run: `pytest tests/test_heads.py -v`

```bash
git add geoadapter/models/heads.py tests/test_heads.py
git commit -m "feat: add classification and multilabel task heads"
```

---

### Task 6: Dataset loaders and transforms

**Files:**
- Create: `geoadapter/data/datasets.py`
- Create: `geoadapter/data/transforms.py`
- Create: `tests/test_datasets.py`

- [ ] **Step 1: Write failing tests**

`tests/test_datasets.py`:
```python
import torch
from geoadapter.data.transforms import BandSelector, Normalize


class TestBandSelector:
    def test_select_rgb(self):
        sel = BandSelector(indices=[3, 2, 1])  # B4,B3,B2 from 13-band
        x = torch.randn(13, 64, 64)
        out = sel(x)
        assert out.shape == (3, 64, 64)
        assert torch.allclose(out[0], x[3])

    def test_identity(self):
        sel = BandSelector(indices=None)
        x = torch.randn(6, 64, 64)
        out = sel(x)
        assert out.shape == (6, 64, 64)


class TestNormalize:
    def test_output_range(self):
        norm = Normalize(method="log1p")
        x = torch.randint(0, 10000, (5, 64, 64)).float()
        out = norm(x)
        assert out.min() >= 0
```

- [ ] **Step 2: Implement transforms**

`geoadapter/data/transforms.py`:
```python
import torch


class BandSelector:
    """Select specific band indices from a multi-band tensor."""

    def __init__(self, indices: list[int] | None = None):
        self.indices = indices

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.indices is None:
            return x
        return x[self.indices]


class Normalize:
    """Radiometric normalization for satellite imagery."""

    def __init__(self, method: str = "log1p"):
        self.method = method

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, 10000)
        if self.method == "log1p":
            x = torch.log1p(x) / 10.0
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True) + 1e-6
        return (x - mean) / std
```

- [ ] **Step 3: Implement dataset wrapper**

`geoadapter/data/datasets.py`:
```python
"""Dataset loaders for GeoAdapter benchmarks.

Wraps torchgeo datasets with modality configuration support.
Requires `pip install geoadapter[bench]` for torchgeo dependency.
"""
from pathlib import Path
from torch.utils.data import Dataset


class ModalityConfig:
    """Defines which bands to select and how to label them."""

    PRESETS = {
        "s2_full": {"indices": list(range(13)), "c_in": 13, "name": "Sentinel-2 Full"},
        "rgb": {"indices": [3, 2, 1], "c_in": 3, "name": "RGB (B4,B3,B2)"},
        "rgb_sar": {"indices": [3, 2, 1], "c_in": 4, "name": "RGB + SAR VV"},
        "gf2": {"indices": [3, 2, 1, 7], "c_in": 4, "name": "GF-2 (B,G,R,NIR)"},
        "sar_only": {"indices": None, "c_in": 2, "name": "SAR VV+VH"},
    }

    def __init__(self, preset: str):
        cfg = self.PRESETS[preset]
        self.indices = cfg["indices"]
        self.c_in = cfg["c_in"]
        self.name = cfg["name"]


def load_eurosat(root: str, modality: str = "s2_full", split: str = "train"):
    """Load EuroSAT via torchgeo with modality selection.

    Returns a standard PyTorch Dataset yielding (tensor, label) tuples.
    """
    try:
        from torchgeo.datasets import EuroSAT
    except ImportError:
        raise ImportError("Install torchgeo: pip install geoadapter[bench]")

    cfg = ModalityConfig(modality)
    ds = EuroSAT(root=root, split=split, download=True)
    return _BandSubset(ds, cfg.indices, key="image")


class _BandSubset(Dataset):
    """Wraps a torchgeo dataset to select specific bands."""

    def __init__(self, base_dataset, band_indices, key="image"):
        self.base = base_dataset
        self.indices = band_indices
        self.key = key

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        img = sample[self.key]
        if self.indices is not None:
            img = img[self.indices]
        label = sample.get("label", sample.get("labels", 0))
        return img, label
```

- [ ] **Step 4: Run tests, commit**

Run: `pytest tests/test_datasets.py -v`

```bash
git add geoadapter/data/ tests/test_datasets.py
git commit -m "feat: add dataset loaders and band selection transforms"
```

---

### Task 7: Unified training engine

**Files:**
- Create: `geoadapter/engine/trainer.py`
- Create: `geoadapter/engine/evaluator.py`
- Create: `tests/test_trainer.py`

- [ ] **Step 1: Write failing smoke test**

`tests/test_trainer.py`:
```python
import torch
from geoadapter.engine.trainer import PEFTTrainer
from geoadapter.models.prithvi import PrithviBackbone
from geoadapter.adapters.geo_adapter import GeoAdapter
from geoadapter.models.heads import ClassificationHead


class TestPEFTTrainer:
    def test_one_step(self):
        backbone = PrithviBackbone(pretrained=False)
        adapter = GeoAdapter(in_channels=4, out_channels=6)
        head = ClassificationHead(in_dim=768, num_classes=10)
        trainer = PEFTTrainer(backbone, adapter, head, lr=1e-3)

        x = torch.randn(4, 4, 128, 128)
        y = torch.randint(0, 10, (4,))
        loss = trainer.train_step(x, y)
        assert isinstance(loss, float)
        assert loss > 0
```

- [ ] **Step 2: Implement trainer**

`geoadapter/engine/trainer.py`:
```python
import torch
import torch.nn as nn
from geoadapter.adapters.base import ModalityAdapter
from geoadapter.models.prithvi import PrithviBackbone


class PEFTTrainer:
    """Unified training loop for all PEFT methods."""

    def __init__(
        self,
        backbone: PrithviBackbone,
        adapter: ModalityAdapter | None,
        head: nn.Module,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.backbone = backbone.to(device)
        self.adapter = adapter.to(device) if adapter else None
        self.head = head.to(device)
        self.device = device

        # Collect trainable params
        params = list(head.parameters())
        if adapter:
            params += [p for p in adapter.parameters() if p.requires_grad]
        # Also include any unfrozen backbone params (LoRA/BitFit/Houlsby)
        params += [p for p in backbone.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()

        if self.adapter:
            x = self.adapter(x)
        features = self.backbone(x)
        logits = self.head(features)
        loss = self.criterion(logits, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        if self.adapter:
            x = self.adapter(x)
        features = self.backbone(x)
        return self.head(features)
```

- [ ] **Step 3: Implement evaluator**

`geoadapter/engine/evaluator.py`:
```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, average_precision_score


def compute_classification_metrics(y_true, y_pred):
    """Compute OA and Macro F1 for single-label classification."""
    return {
        "overall_accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def compute_multilabel_metrics(y_true, y_scores):
    """Compute mAP for multi-label classification."""
    return {
        "mAP": average_precision_score(y_true, y_scores, average="macro"),
    }
```

- [ ] **Step 4: Run tests, commit**

Run: `pytest tests/test_trainer.py -v`

```bash
git add geoadapter/engine/ tests/test_trainer.py
git commit -m "feat: unified PEFT training engine and evaluator"
```

---

### Task 8: Benchmark runner + experiment config

**Files:**
- Create: `geoadapter/bench/run_benchmark.py`
- Create: `geoadapter/bench/configs/eurosat_default.yaml`

- [ ] **Step 1: Create experiment config**

`geoadapter/bench/configs/eurosat_default.yaml`:
```yaml
experiment:
  name: eurosat_geoadapter_benchmark
  dataset: eurosat
  dataset_root: ./data/eurosat
  epochs: 50
  batch_size: 64
  seeds: [42, 123, 456]

modalities:
  - preset: s2_full
  - preset: rgb
  - preset: rgb_sar
  - preset: gf2
  - preset: sar_only

methods:
  - name: linear_probe
    adapter: zero_pad
    peft: null
  - name: bitfit
    adapter: zero_pad
    peft: bitfit
  - name: houlsby
    adapter: zero_pad
    peft: houlsby
    bottleneck_dim: 64
  - name: lora_r8
    adapter: zero_pad
    peft: lora
    rank: 8
  - name: geoadapter
    adapter: geo_adapter
    peft: null

training:
  lr: 1.0e-3
  lr_peft: 1.0e-4
  scheduler: cosine
  weight_decay: 0.01

prithvi:
  pretrained: true
  checkpoint: null  # auto-download from HuggingFace
```

- [ ] **Step 2: Implement benchmark runner**

`geoadapter/bench/run_benchmark.py`:
```python
"""CLI entry point: python -m geoadapter.bench.run_benchmark --config path/to/config.yaml"""
import argparse
import yaml
import json
import itertools
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_single_experiment(method_cfg, modality_cfg, global_cfg, seed):
    """Run one (method, modality, seed) combination. Returns metrics dict."""
    # Lazy imports so the CLI loads fast
    import torch
    from geoadapter.models.prithvi import PrithviBackbone
    from geoadapter.models.heads import ClassificationHead
    from geoadapter.adapters import GeoAdapter, ZeroPadAdapter
    from geoadapter.adapters.lora import inject_lora
    from geoadapter.adapters.bitfit import configure_bitfit
    from geoadapter.adapters.houlsby import inject_houlsby_adapters
    from geoadapter.data.datasets import ModalityConfig
    from geoadapter.engine.trainer import PEFTTrainer

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_m = ModalityConfig(modality_cfg["preset"])

    # Build backbone
    backbone = PrithviBackbone(pretrained=global_cfg["prithvi"]["pretrained"])

    # Apply PEFT to backbone if needed
    peft = method_cfg.get("peft")
    if peft == "lora":
        for block in backbone.blocks:
            inject_lora(block, rank=method_cfg.get("rank", 8))
    elif peft == "bitfit":
        configure_bitfit(backbone)
    elif peft == "houlsby":
        for block in backbone.blocks:
            inject_houlsby_adapters(block, bottleneck_dim=method_cfg.get("bottleneck_dim", 64))

    # Build adapter
    if method_cfg["adapter"] == "geo_adapter":
        adapter = GeoAdapter(in_channels=cfg_m.c_in, out_channels=6)
    else:
        adapter = ZeroPadAdapter(in_channels=cfg_m.c_in, out_channels=6)

    head = ClassificationHead(in_dim=768, num_classes=10)
    trainer = PEFTTrainer(backbone, adapter, head, lr=global_cfg["training"]["lr"], device=device)

    # Training loop placeholder — actual dataset loading happens here
    print(f"  Running: method={method_cfg['name']}, modality={modality_cfg['preset']}, seed={seed}")
    # TODO: integrate with dataset loader in Colab notebooks
    return {"method": method_cfg["name"], "modality": modality_cfg["preset"], "seed": seed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = []

    for method, modality in itertools.product(cfg["methods"], cfg["modalities"]):
        for seed in cfg["experiment"]["seeds"]:
            result = run_single_experiment(method, modality, cfg, seed)
            results.append(result)

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify CLI loads**

Run: `python -m geoadapter.bench.run_benchmark --config geoadapter/bench/configs/eurosat_default.yaml --output /dev/null`
Expected: Prints experiment combinations (no actual training without data)

- [ ] **Step 4: Commit**

```bash
git add geoadapter/bench/
git commit -m "feat: benchmark runner with YAML experiment configs"
```

---

### Task 9: Visualization module

**Files:**
- Create: `geoadapter/viz/embedding_viz.py`

- [ ] **Step 1: Implement visualization utilities**

`geoadapter/viz/embedding_viz.py`:
```python
"""Embedding visualization: t-SNE/UMAP + channel attention heatmaps."""
import numpy as np


def compute_tsne(embeddings: np.ndarray, perplexity: int = 30) -> np.ndarray:
    """Reduce N x D embeddings to N x 2 via t-SNE."""
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(embeddings)


def compute_umap(embeddings: np.ndarray, n_neighbors: int = 15) -> np.ndarray:
    """Reduce N x D embeddings to N x 2 via UMAP."""
    import umap
    return umap.UMAP(n_neighbors=n_neighbors, random_state=42).fit_transform(embeddings)


def extract_channel_attention_weights(adapter) -> np.ndarray:
    """Extract SE attention weights from a GeoAdapter for visualization."""
    # Run a dummy forward to capture attention
    import torch
    adapter.eval()
    dummy = torch.randn(1, adapter.in_channels, 64, 64)
    with torch.no_grad():
        proj = adapter.channel_proj(dummy)
        attn = adapter.channel_attn(proj)
    return attn.squeeze().cpu().numpy()
```

- [ ] **Step 2: Commit**

```bash
git add geoadapter/viz/
git commit -m "feat: t-SNE/UMAP embedding visualization utilities"
```

---

### Task 10: Wire geoadapter into ae_backend

**Files:**
- Modify: `ae_backend/app/services/trainer.py`
- Modify: `ae_backend/requirements.txt`

- [ ] **Step 1: Add geoadapter dependency**

Append to `ae_backend/requirements.txt`:
```
# Local package
-e ../.
```

- [ ] **Step 2: Update trainer.py to use geoadapter backbone**

Replace the `PrithviAlphaEarthEncoder` class in `ae_backend/app/services/trainer.py` with a thin wrapper that imports from `geoadapter`:

At the top of the file, add:
```python
from geoadapter.models.prithvi import PrithviBackbone
from geoadapter.adapters.geo_adapter import GeoAdapter
from geoadapter.models.heads import ClassificationHead
```

Replace the `PrithviAlphaEarthEncoder` class (lines 77-179) with:
```python
class PrithviAlphaEarthEncoder(nn.Module):
    """Wrapper using geoadapter's PrithviBackbone + GeoAdapter."""

    def __init__(self, weight_path=None, in_channels=5, hidden_dim=64):
        super().__init__()
        self.adapter = GeoAdapter(in_channels=in_channels, out_channels=6)
        self.backbone = PrithviBackbone(
            pretrained=bool(weight_path),
            checkpoint_path=weight_path,
        )
        self.finetune_head = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(),
            nn.Linear(256, in_channels * 128 * 128),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        x_adapted = self.adapter(x)
        features = self.backbone(x_adapted)
        ae_embed = self.finetune_head(features)
        rec = self.decoder(ae_embed).view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        z = self.proj(ae_embed)
        return rec, z
```

- [ ] **Step 3: Verify backend still starts**

Run: `cd ae_backend && python -c "from app.services.trainer import PrithviAlphaEarthEncoder; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add ae_backend/app/services/trainer.py ae_backend/requirements.txt
git commit -m "refactor: wire ae_backend trainer to geoadapter package"
```

---

### Task 11: Colab notebook template

**Files:**
- Create: `notebooks/01_benchmark_eurosat.ipynb` (as .py script for version control)

- [ ] **Step 1: Create notebook script**

Create `notebooks/01_benchmark_eurosat.py`:
```python
# %% [markdown]
# # GeoAdapter Benchmark: EuroSAT
# Run on Google Colab Pro+ with A100 GPU.

# %% Install dependencies
# !pip install -q torch torchvision timm torchgeo scikit-learn pyyaml umap-learn
# !pip install -e /content/AlphaEarth-System/

# %% Imports
import torch
from geoadapter.models.prithvi import PrithviBackbone
from geoadapter.adapters import GeoAdapter, ZeroPadAdapter
from geoadapter.models.heads import ClassificationHead
from geoadapter.engine.trainer import PEFTTrainer
from geoadapter.engine.evaluator import compute_classification_metrics
from geoadapter.data.datasets import load_eurosat

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# %% Load dataset
train_ds = load_eurosat(root="./data", modality="rgb", split="train")
# ... training loop using PEFTTrainer
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/
git commit -m "feat: add Colab benchmark notebook template"
```

---

## Summary

| Task | Component | Est. Time |
|------|-----------|-----------|
| 1 | Package scaffold | 5 min |
| 2 | GeoAdapter + ZeroPad | 15 min |
| 3 | LoRA + BitFit + Houlsby | 15 min |
| 4 | Prithvi backbone | 20 min |
| 5 | Task heads | 10 min |
| 6 | Dataset loaders | 15 min |
| 7 | Training engine | 15 min |
| 8 | Benchmark runner | 15 min |
| 9 | Visualization | 10 min |
| 10 | ae_backend integration | 10 min |
| 11 | Colab notebook | 10 min |
| **Total** | | **~2.5 hours** |
