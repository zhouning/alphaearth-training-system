# Paper12 Second-Backbone Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline-testable SatMAE-compatible second-backbone validation track for Paper12 and prepare the Colab experiment that produces the 18-row EuroSAT evidence.

**Architecture:** Add a small backbone factory so the existing benchmark runner can select either the current Prithvi path or a SatMAE-compatible ViT backbone. Keep local tests offline by validating config, factory behavior, dry-run matrix shape, and summary schema; put the real pretrained-weight experiment in a generated Colab notebook that requires an explicit Drive-staged SatMAE checkpoint.

**Tech Stack:** Python, PyTorch, YAML, pytest, Jupyter notebook JSON, existing `geoadapter` benchmark runner and PEFT adapters.

---

## File Structure

- Create `geoadapter/models/satmae.py`
  - Defines `SatMAEBackbone`, a frozen ViT-style backbone with 10-channel patch embedding and PyTorch `TransformerEncoderLayer` blocks compatible with existing LoRA/Houlsby injection helpers.
  - Loads a local SatMAE-compatible checkpoint with strict file presence and lenient key mapping.
- Create `geoadapter/models/backbone_factory.py`
  - Defines `BackboneSpec`.
  - Builds existing Prithvi metadata from legacy configs.
  - Builds the SatMAE-compatible backbone from the new `backbone:` config block.
- Modify `geoadapter/bench/run_benchmark.py`
  - Uses `build_backbone(global_cfg)` instead of hard-coded `PrithviBackbone`.
  - Uses backbone metadata for adapter output channels and head input dimension.
  - Emits `backbone` in every result row.
- Create `geoadapter/bench/configs/eurosat_second_backbone.yaml`
  - Defines the 18-row EuroSAT SatMAE matrix.
- Create `geoadapter/bench/second_backbone_summary.py`
  - Validates raw second-backbone rows and writes grouped summary JSON.
- Modify `scripts/make_paper12_colab_notebooks.py`
  - Generates `colab/paper12_second_backbone_eurosat_colab.ipynb`.
- Create or modify tests:
  - Create `tests/test_backbone_factory.py`.
  - Modify `tests/test_benchmark_runner.py`.
  - Modify `tests/test_paper12_colab_notebooks.py`.
  - Create `tests/test_second_backbone_summary.py`.
- Modify status documents:
  - `submission/paper12_isprs_jprs_20260606/REQUIRED_EXPERIMENTS_ISPRS.md`.
  - `submission/paper12_isprs_jprs_20260606/00_ACTION_REQUIRED.md`.
  - `paper12/README.md`.

## Task 1: Backbone Factory and SatMAE-Compatible Backbone

**Files:**
- Create: `tests/test_backbone_factory.py`
- Create: `geoadapter/models/satmae.py`
- Create: `geoadapter/models/backbone_factory.py`

- [ ] **Step 1: Write failing backbone factory tests**

Create `tests/test_backbone_factory.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
import torch


def test_build_backbone_preserves_legacy_prithvi_defaults():
    from geoadapter.models.backbone_factory import build_backbone
    from geoadapter.models.prithvi import PrithviBackbone

    spec = build_backbone(
        {
            "prithvi": {
                "pretrained": False,
                "checkpoint": None,
            }
        }
    )

    assert spec.name == "prithvi"
    assert isinstance(spec.model, PrithviBackbone)
    assert spec.feature_dim == 768
    assert spec.input_channels == 6
    assert spec.blocks is spec.model.blocks
    assert len(spec.blocks) == 12


def test_build_backbone_constructs_satmae_without_weights_for_unit_tests():
    from geoadapter.models.backbone_factory import build_backbone
    from geoadapter.models.satmae import SatMAEBackbone

    spec = build_backbone(
        {
            "backbone": {
                "name": "satmae_vit_base",
                "family": "satmae",
                "pretrained": False,
                "checkpoint": None,
                "input_channels": 10,
                "embed_dim": 128,
                "depth": 2,
                "num_heads": 4,
                "patch_size": 16,
            }
        }
    )

    assert spec.name == "satmae_vit_base"
    assert isinstance(spec.model, SatMAEBackbone)
    assert spec.feature_dim == 128
    assert spec.input_channels == 10
    assert len(spec.blocks) == 2

    x = torch.randn(2, 10, 64, 64)
    features = spec.model(x)
    assert features.shape == (2, 128)
    assert all(not param.requires_grad for param in spec.model.parameters())


def test_satmae_pretrained_requires_existing_checkpoint(tmp_path: Path):
    from geoadapter.models.backbone_factory import build_backbone

    missing = tmp_path / "missing_satmae.pth"
    with pytest.raises(FileNotFoundError, match="SatMAE checkpoint not found"):
        build_backbone(
            {
                "backbone": {
                    "name": "satmae_vit_base",
                    "family": "satmae",
                    "pretrained": True,
                    "checkpoint": str(missing),
                    "input_channels": 10,
                    "embed_dim": 128,
                    "depth": 2,
                    "num_heads": 4,
                    "patch_size": 16,
                }
            }
        )


def test_unknown_backbone_family_fails_clearly():
    from geoadapter.models.backbone_factory import build_backbone

    with pytest.raises(ValueError, match="unsupported backbone family"):
        build_backbone(
            {
                "backbone": {
                    "name": "x",
                    "family": "not_a_backbone",
                    "pretrained": False,
                }
            }
        )
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run:

```powershell
python -m pytest tests/test_backbone_factory.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `geoadapter.models.backbone_factory`.

- [ ] **Step 3: Implement `SatMAEBackbone`**

Create `geoadapter/models/satmae.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class SatMAEBackbone(nn.Module):
    """SatMAE-compatible frozen ViT backbone for Paper12 second-backbone checks."""

    _KEY_MAP = {
        "attn.qkv.weight": "self_attn.in_proj_weight",
        "attn.qkv.bias": "self_attn.in_proj_bias",
        "attn.proj.weight": "self_attn.out_proj.weight",
        "attn.proj.bias": "self_attn.out_proj.bias",
        "mlp.fc1.weight": "linear1.weight",
        "mlp.fc1.bias": "linear1.bias",
        "mlp.fc2.weight": "linear2.weight",
        "mlp.fc2.bias": "linear2.bias",
        "patch_embed.proj.weight": "patch_embed.weight",
        "patch_embed.proj.bias": "patch_embed.bias",
    }

    def __init__(
        self,
        *,
        pretrained: bool = True,
        checkpoint_path: str | None = None,
        in_chans: int = 10,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.patch_size = patch_size

        self.patch_embed = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        if pretrained:
            if not checkpoint_path:
                raise FileNotFoundError("SatMAE checkpoint not found: no checkpoint path configured")
            self._load_checkpoint(checkpoint_path)

        self._freeze_all()

    def _freeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)

    def _checkpoint_state(self, payload: Any) -> dict[str, torch.Tensor]:
        if isinstance(payload, dict):
            for key in ("model", "state_dict", "checkpoint"):
                value = payload.get(key)
                if isinstance(value, dict):
                    return value
            if all(torch.is_tensor(value) for value in payload.values()):
                return payload
        raise ValueError("SatMAE checkpoint does not contain a tensor state dict")

    def _normalize_key(self, key: str) -> str:
        for prefix in ("module.", "encoder.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix) :]

        for satmae_suffix, torch_suffix in self._KEY_MAP.items():
            if key.endswith(satmae_suffix):
                key = key[: -len(satmae_suffix)] + torch_suffix
                break
        return key

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"SatMAE checkpoint not found: {path}")

        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = self._checkpoint_state(payload)
        own_state = self.state_dict()
        loaded = 0

        for raw_key, tensor in state.items():
            key = self._normalize_key(raw_key)
            if key in own_state and own_state[key].shape == tensor.shape:
                own_state[key] = tensor
                loaded += 1

        if loaded == 0:
            raise ValueError(f"No compatible SatMAE tensors loaded from {path}")
        self.load_state_dict(own_state, strict=False)

    def forward(self, x: torch.Tensor, return_spatial: bool = False):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        if return_spatial:
            return x[:, 1:], (h, w)
        return x[:, 0]
```

- [ ] **Step 4: Implement the backbone factory**

Create `geoadapter/models/backbone_factory.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from geoadapter.models.prithvi import PrithviBackbone
from geoadapter.models.satmae import SatMAEBackbone


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    model: nn.Module
    feature_dim: int
    input_channels: int
    blocks: nn.ModuleList


def _legacy_prithvi_config(global_cfg: dict[str, Any]) -> dict[str, Any]:
    return global_cfg.get("prithvi") or {"pretrained": True, "checkpoint": None}


def _build_prithvi(global_cfg: dict[str, Any]) -> BackboneSpec:
    cfg = _legacy_prithvi_config(global_cfg)
    model = PrithviBackbone(
        pretrained=bool(cfg.get("pretrained", True)),
        checkpoint_path=cfg.get("checkpoint"),
        embed_dim=int(cfg.get("embed_dim", 768)),
        depth=int(cfg.get("depth", 12)),
        num_heads=int(cfg.get("num_heads", 12)),
        in_chans=int(cfg.get("input_channels", cfg.get("in_chans", 6))),
        patch_size=int(cfg.get("patch_size", 16)),
    )
    return BackboneSpec(
        name="prithvi",
        model=model,
        feature_dim=model.embed_dim,
        input_channels=int(cfg.get("input_channels", cfg.get("in_chans", 6))),
        blocks=model.blocks,
    )


def _build_satmae(backbone_cfg: dict[str, Any]) -> BackboneSpec:
    model = SatMAEBackbone(
        pretrained=bool(backbone_cfg.get("pretrained", True)),
        checkpoint_path=backbone_cfg.get("checkpoint"),
        in_chans=int(backbone_cfg.get("input_channels", 10)),
        embed_dim=int(backbone_cfg.get("embed_dim", 768)),
        depth=int(backbone_cfg.get("depth", 12)),
        num_heads=int(backbone_cfg.get("num_heads", 12)),
        patch_size=int(backbone_cfg.get("patch_size", 16)),
    )
    return BackboneSpec(
        name=str(backbone_cfg.get("name", "satmae_vit_base")),
        model=model,
        feature_dim=model.embed_dim,
        input_channels=model.in_chans,
        blocks=model.blocks,
    )


def build_backbone(global_cfg: dict[str, Any]) -> BackboneSpec:
    backbone_cfg = global_cfg.get("backbone")
    if not backbone_cfg:
        return _build_prithvi(global_cfg)

    family = str(backbone_cfg.get("family", backbone_cfg.get("name", "prithvi"))).lower()
    if family == "prithvi":
        merged = dict(global_cfg)
        merged["prithvi"] = backbone_cfg
        return _build_prithvi(merged)
    if family == "satmae":
        return _build_satmae(backbone_cfg)
    raise ValueError(f"unsupported backbone family: {family}")
```

- [ ] **Step 5: Run the backbone factory tests**

Run:

```powershell
python -m pytest tests/test_backbone_factory.py -q
```

Expected: `4 passed`.

- [ ] **Step 6: Commit Task 1**

Run:

```powershell
git add tests/test_backbone_factory.py geoadapter/models/satmae.py geoadapter/models/backbone_factory.py
git commit -m "feat: add satmae backbone factory"
```

## Task 2: Runner Integration and Second-Backbone Config

**Files:**
- Modify: `geoadapter/bench/run_benchmark.py`
- Create: `geoadapter/bench/configs/eurosat_second_backbone.yaml`
- Modify: `tests/test_benchmark_runner.py`
- Modify: `tests/test_paper12_colab_notebooks.py`

- [ ] **Step 1: Add failing config and runner tests**

Append to `tests/test_paper12_colab_notebooks.py`:

```python
def test_paper12_second_backbone_config_contract():
    cfg = yaml.safe_load(
        (CONFIG_DIR / "eurosat_second_backbone.yaml").read_text(encoding="utf-8")
    )

    assert cfg["experiment"]["name"] == "eurosat_second_backbone"
    assert cfg["experiment"]["dataset"] == "eurosat"
    assert cfg["experiment"]["dataset_root"] == "./data/eurosat"
    assert cfg["experiment"]["epochs"] == 50
    assert cfg["experiment"]["batch_size"] == 64
    assert cfg["experiment"]["seeds"] == [42, 123, 456]
    assert cfg["experiment"]["allow_synthetic_fallback"] is False

    assert cfg["backbone"] == {
        "name": "satmae_vit_base",
        "family": "satmae",
        "pretrained": True,
        "checkpoint": "data/weights/satmae/satmae_vit_base.pth",
        "input_channels": 10,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "patch_size": 16,
    }
    assert cfg["modalities"] == [{"preset": "s2_full"}, {"preset": "rgb"}]
    assert [method["name"] for method in cfg["methods"]] == [
        "satmae_linear_probe",
        "satmae_lora_split_qkv_r8",
        "satmae_houlsby_d64",
    ]

    matrix_size = (
        len(cfg["modalities"])
        * len(cfg["methods"])
        * len(cfg["experiment"]["seeds"])
    )
    assert matrix_size == 18
```

Append to `tests/test_benchmark_runner.py`:

```python
def test_run_single_experiment_uses_backbone_metadata(monkeypatch):
    from dataclasses import dataclass

    import geoadapter.bench.run_benchmark as runner

    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList()
            self.weight = nn.Parameter(torch.zeros(1), requires_grad=False)

        def forward(self, x, return_spatial=False):
            features = torch.zeros(x.shape[0], 32, device=x.device)
            if return_spatial:
                return features.unsqueeze(1), (1, 1)
            return features

    @dataclass(frozen=True)
    class TinySpec:
        name: str
        model: nn.Module
        feature_dim: int
        input_channels: int
        blocks: nn.ModuleList

    monkeypatch.setattr(
        runner,
        "build_backbone",
        lambda cfg: TinySpec(
            name="tiny_backbone",
            model=TinyBackbone(),
            feature_dim=32,
            input_channels=4,
            blocks=nn.ModuleList(),
        ),
    )

    cfg = {
        "experiment": {
            "dataset": "eurosat",
            "dataset_root": "missing",
            "epochs": 0,
            "batch_size": 8,
            "allow_synthetic_fallback": True,
        },
        "training": {"lr": 1e-3},
        "backbone": {"name": "tiny_backbone", "family": "satmae", "pretrained": False},
    }

    result = runner.run_single_experiment(
        {"name": "linear_probe", "adapter": "zero_pad", "peft": None},
        {"preset": "rgb"},
        cfg,
        seed=42,
    )

    assert result["backbone"] == "tiny_backbone"
    assert result["trainable_params"] == 330
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```powershell
python -m pytest tests/test_paper12_colab_notebooks.py::test_paper12_second_backbone_config_contract tests/test_benchmark_runner.py::test_run_single_experiment_uses_backbone_metadata -q
```

Expected: config test fails because `eurosat_second_backbone.yaml` is missing; runner test fails because `build_backbone` is not imported at module level.

- [ ] **Step 3: Add the second-backbone config**

Create `geoadapter/bench/configs/eurosat_second_backbone.yaml`:

```yaml
experiment:
  name: eurosat_second_backbone
  dataset: eurosat
  dataset_root: ./data/eurosat
  epochs: 50
  batch_size: 64
  seeds: [42, 123, 456]
  allow_synthetic_fallback: false

backbone:
  name: satmae_vit_base
  family: satmae
  pretrained: true
  checkpoint: data/weights/satmae/satmae_vit_base.pth
  input_channels: 10
  embed_dim: 768
  depth: 12
  num_heads: 12
  patch_size: 16

modalities:
  - preset: s2_full
  - preset: rgb

methods:
  - name: satmae_linear_probe
    adapter: zero_pad
    peft: null
  - name: satmae_lora_split_qkv_r8
    adapter: zero_pad
    peft: lora_split_qkv
    rank: 8
  - name: satmae_houlsby_d64
    adapter: zero_pad
    peft: houlsby
    bottleneck_dim: 64

training:
  lr: 1.0e-3
  lr_peft: 1.0e-4
  scheduler: cosine
  weight_decay: 0.01
```

- [ ] **Step 4: Integrate the factory into `run_benchmark.py`**

Modify the imports near the top of `geoadapter/bench/run_benchmark.py`:

```python
from geoadapter.models.backbone_factory import build_backbone
```

Inside `run_single_experiment`, remove:

```python
from geoadapter.models.prithvi import PrithviBackbone
```

Replace the hard-coded backbone construction:

```python
    backbone = PrithviBackbone(
        pretrained=global_cfg["prithvi"]["pretrained"],
        checkpoint_path=global_cfg["prithvi"].get("checkpoint"),
    )
```

with:

```python
    backbone_spec = build_backbone(global_cfg)
    backbone = backbone_spec.model
```

Replace every loop over `backbone.blocks` for PEFT injection with `backbone_spec.blocks`:

```python
    if peft == "lora":
        for block in backbone_spec.blocks:
            inject_lora(block, rank=method_cfg.get("rank", 8))
    elif peft == "bitfit":
        configure_bitfit(backbone)
    elif peft == "houlsby":
        for block in backbone_spec.blocks:
            inject_houlsby_adapters(block, bottleneck_dim=method_cfg.get("bottleneck_dim", 64))
    elif peft == "full_finetune":
        for p in backbone.parameters():
            p.requires_grad_(True)
    elif peft == "lora_split_qkv":
        from geoadapter.adapters.lora import split_qkv_and_inject_lora
        for block in backbone_spec.blocks:
            split_qkv_and_inject_lora(block, rank=method_cfg.get("rank", 8))
```

Replace adapter construction:

```python
    adapter = build_adapter(method_cfg["adapter"], in_channels=cfg_m.c_in, out_channels=6)
```

with:

```python
    adapter = build_adapter(
        method_cfg["adapter"],
        in_channels=cfg_m.c_in,
        out_channels=backbone_spec.input_channels,
    )
```

Replace head construction dimensions:

```python
        head = MultiLabelHead(in_dim=768, num_classes=num_classes)
...
        head = SegmentationHead(in_dim=768, num_classes=num_classes, patch_size=16)
...
        head = ClassificationHead(in_dim=768, num_classes=num_classes)
```

with:

```python
        head = MultiLabelHead(in_dim=backbone_spec.feature_dim, num_classes=num_classes)
...
        head = SegmentationHead(
            in_dim=backbone_spec.feature_dim,
            num_classes=num_classes,
            patch_size=global_cfg.get("backbone", {}).get("patch_size", 16),
        )
...
        head = ClassificationHead(in_dim=backbone_spec.feature_dim, num_classes=num_classes)
```

Replace metrics initialization:

```python
    metrics = {"method": method_cfg["name"], "modality": modality_cfg["preset"],
               "seed": seed, "trainable_params": n_trainable}
```

with:

```python
    metrics = {
        "backbone": backbone_spec.name,
        "method": method_cfg["name"],
        "modality": modality_cfg["preset"],
        "seed": seed,
        "trainable_params": n_trainable,
    }
```

Update the status print:

```python
    print(f"  [{tag}] device={device}, trainable_params={n_trainable:,}")
```

to:

```python
    print(
        f"  [{tag}] backbone={backbone_spec.name}, "
        f"device={device}, trainable_params={n_trainable:,}"
    )
```

- [ ] **Step 5: Run focused runner/config tests**

Run:

```powershell
python -m pytest tests/test_benchmark_runner.py tests/test_paper12_colab_notebooks.py::test_paper12_second_backbone_config_contract -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Run a dry-run matrix**

Run:

```powershell
python -m geoadapter.bench.run_benchmark --config geoadapter/bench/configs/eurosat_second_backbone.yaml --dry-run
```

Expected output includes:

```text
Total experiments: 18
  satmae_linear_probe x s2_full x seed=42
  satmae_lora_split_qkv_r8 x s2_full x seed=42
  satmae_houlsby_d64 x rgb x seed=456
```

- [ ] **Step 7: Commit Task 2**

Run:

```powershell
git add geoadapter/bench/run_benchmark.py geoadapter/bench/configs/eurosat_second_backbone.yaml tests/test_benchmark_runner.py tests/test_paper12_colab_notebooks.py
git commit -m "feat: add paper12 second-backbone benchmark config"
```

## Task 3: Second-Backbone Summary Builder

**Files:**
- Create: `tests/test_second_backbone_summary.py`
- Create: `geoadapter/bench/second_backbone_summary.py`

- [ ] **Step 1: Write failing summary tests**

Create `tests/test_second_backbone_summary.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _rows():
    rows = []
    for modality in ("s2_full", "rgb"):
        for seed in (42, 123, 456):
            rows.append(
                {
                    "backbone": "satmae_vit_base",
                    "method": "satmae_linear_probe",
                    "modality": modality,
                    "seed": seed,
                    "trainable_params": 7690,
                    "overall_accuracy": 0.70 if modality == "s2_full" else 0.60,
                    "macro_f1": 0.69 if modality == "s2_full" else 0.59,
                }
            )
            rows.append(
                {
                    "backbone": "satmae_vit_base",
                    "method": "satmae_lora_split_qkv_r8",
                    "modality": modality,
                    "seed": seed,
                    "trainable_params": 155146,
                    "overall_accuracy": 0.72 if modality == "s2_full" else 0.62,
                    "macro_f1": 0.71 if modality == "s2_full" else 0.61,
                }
            )
            rows.append(
                {
                    "backbone": "satmae_vit_base",
                    "method": "satmae_houlsby_d64",
                    "modality": modality,
                    "seed": seed,
                    "trainable_params": 1197322,
                    "overall_accuracy": 0.80 if modality == "s2_full" else 0.68,
                    "macro_f1": 0.79 if modality == "s2_full" else 0.67,
                }
            )
    return rows


def test_build_second_backbone_summary_groups_and_ranks_methods():
    from geoadapter.bench.second_backbone_summary import build_second_backbone_summary

    summary = build_second_backbone_summary(_rows())

    assert summary["schema"] == "paper12.second_backbone_eurosat_summary.v1"
    assert summary["row_count"] == 18
    assert len(summary["groups"]) == 6

    by_key = {
        (item["method"], item["modality"]): item
        for item in summary["groups"]
    }
    houlsby_s2 = by_key[("satmae_houlsby_d64", "s2_full")]
    assert houlsby_s2["overall_accuracy_mean"] == pytest.approx(0.80)
    assert houlsby_s2["macro_f1_mean"] == pytest.approx(0.79)
    assert houlsby_s2["rank_by_overall_accuracy"] == 1
    assert houlsby_s2["seeds"] == [42, 123, 456]

    linear_rgb = by_key[("satmae_linear_probe", "rgb")]
    assert linear_rgb["rank_by_overall_accuracy"] == 3


def test_build_second_backbone_summary_requires_fields():
    from geoadapter.bench.second_backbone_summary import build_second_backbone_summary

    bad = _rows()
    bad[0] = dict(bad[0])
    bad[0].pop("macro_f1")

    with pytest.raises(ValueError, match="missing required fields"):
        build_second_backbone_summary(bad)


def test_write_second_backbone_summary_round_trips(tmp_path: Path):
    from geoadapter.bench.second_backbone_summary import write_second_backbone_summary

    raw_path = tmp_path / "raw.json"
    summary_path = tmp_path / "summary.json"
    raw_path.write_text(json.dumps(_rows()), encoding="utf-8")

    summary = write_second_backbone_summary(raw_path, summary_path)

    assert summary_path.exists()
    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded == summary
    assert loaded["row_count"] == 18
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
python -m pytest tests/test_second_backbone_summary.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `geoadapter.bench.second_backbone_summary`.

- [ ] **Step 3: Implement the summary builder**

Create `geoadapter/bench/second_backbone_summary.py`:

```python
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


REQUIRED_FIELDS = {
    "backbone",
    "method",
    "modality",
    "seed",
    "trainable_params",
    "overall_accuracy",
    "macro_f1",
}


def _require_fields(row: dict[str, Any], index: int) -> None:
    missing = REQUIRED_FIELDS - set(row)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"row {index} missing required fields: {names}")


def _std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def build_second_backbone_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        _require_fields(row, index)
        key = (str(row["backbone"]), str(row["method"]), str(row["modality"]))
        grouped[key].append(row)

    groups = []
    for (backbone, method, modality), group_rows in sorted(grouped.items()):
        oa = [float(row["overall_accuracy"]) for row in group_rows]
        f1 = [float(row["macro_f1"]) for row in group_rows]
        params = sorted({int(row["trainable_params"]) for row in group_rows})
        seeds = sorted(int(row["seed"]) for row in group_rows)
        groups.append(
            {
                "backbone": backbone,
                "method": method,
                "modality": modality,
                "trainable_params": params[0] if len(params) == 1 else params,
                "overall_accuracy_mean": mean(oa),
                "overall_accuracy_std": _std(oa),
                "macro_f1_mean": mean(f1),
                "macro_f1_std": _std(f1),
                "seeds": seeds,
            }
        )

    by_modality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_modality[group["modality"]].append(group)

    for modality_groups in by_modality.values():
        ranked = sorted(
            modality_groups,
            key=lambda item: item["overall_accuracy_mean"],
            reverse=True,
        )
        for rank, group in enumerate(ranked, start=1):
            group["rank_by_overall_accuracy"] = rank

    return {
        "schema": "paper12.second_backbone_eurosat_summary.v1",
        "row_count": len(rows),
        "groups": groups,
    }


def write_second_backbone_summary(
    input_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("second-backbone raw result file must contain a JSON list")
    summary = build_second_backbone_summary(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_second_backbone_summary(args.input, args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run summary tests**

Run:

```powershell
python -m pytest tests/test_second_backbone_summary.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit Task 3**

Run:

```powershell
git add tests/test_second_backbone_summary.py geoadapter/bench/second_backbone_summary.py
git commit -m "feat: summarize paper12 second-backbone results"
```

## Task 4: Generated Colab Notebook

**Files:**
- Modify: `tests/test_paper12_colab_notebooks.py`
- Modify: `scripts/make_paper12_colab_notebooks.py`
- Create: `colab/paper12_second_backbone_eurosat_colab.ipynb`

- [ ] **Step 1: Add failing notebook contract test**

Append to `tests/test_paper12_colab_notebooks.py`:

```python
def test_paper12_second_backbone_eurosat_colab_notebook_contract():
    path = COLAB_DIR / "paper12_second_backbone_eurosat_colab.ipynb"
    text = read_notebook_text(path)

    assert f"blob/{PAPER12_RESULTS_BRANCH}/colab/paper12_second_backbone_eurosat_colab.ipynb" in text
    assert f"--branch {PAPER12_RESULTS_BRANCH}" in text
    assert "git rev-parse --abbrev-ref HEAD" in text
    assert "Colab Pro L4" in text
    assert "/content/AlphaEarth-System/data/eurosat" in text
    assert "/content/drive/MyDrive/paper12_results" in text
    assert "/content/second_backbone_eurosat_runs" in text
    assert "scripts/download_public_datasets.py --dataset eurosat" in text
    assert "eurosat_second_backbone.yaml" in text
    assert "second_backbone_eurosat.json" in text
    assert "second_backbone_eurosat_summary.json" in text
    assert "satmae_vit_base.pth" in text
    assert "python -m geoadapter.bench.run_benchmark" in text
    assert "python -m geoadapter.bench.second_backbone_summary" in text
    assert "expected_rows = 18" in text
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```powershell
python -m pytest tests/test_paper12_colab_notebooks.py::test_paper12_second_backbone_eurosat_colab_notebook_contract -q
```

Expected: FAIL because `colab/paper12_second_backbone_eurosat_colab.ipynb` does not exist.

- [ ] **Step 3: Extend notebook generator constants**

In `scripts/make_paper12_colab_notebooks.py`, add:

```python
SECOND_BACKBONE_OUT = COLAB_DIR / "paper12_second_backbone_eurosat_colab.ipynb"
```

- [ ] **Step 4: Add the notebook generator function**

Add this function to `scripts/make_paper12_colab_notebooks.py`:

```python
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
```

- [ ] **Step 5: Add notebook to generator outputs**

In `main()` in `scripts/make_paper12_colab_notebooks.py`, add the new output:

```python
    outputs = {
        SECOND_BACKBONE_OUT: second_backbone_notebook(),
        CAPACITY_OUT: capacity_sweep_notebook(),
        LOVE_OUT: loveda_notebook(),
        EURO_OUT: eurosat_notebook(),
    }
```

- [ ] **Step 6: Run the notebook generator**

Run:

```powershell
python scripts\make_paper12_colab_notebooks.py
```

Expected output includes:

```text
[ok] wrote D:\adk\AlphaEarth-System\colab\paper12_second_backbone_eurosat_colab.ipynb
```

- [ ] **Step 7: Run notebook contract tests**

Run:

```powershell
python -m pytest tests/test_paper12_colab_notebooks.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 4**

Run:

```powershell
git add scripts/make_paper12_colab_notebooks.py colab/paper12_second_backbone_eurosat_colab.ipynb tests/test_paper12_colab_notebooks.py
git commit -m "feat: add paper12 second-backbone colab notebook"
```

## Task 5: Status Documents

**Files:**
- Modify: `submission/paper12_isprs_jprs_20260606/REQUIRED_EXPERIMENTS_ISPRS.md`
- Modify: `submission/paper12_isprs_jprs_20260606/00_ACTION_REQUIRED.md`
- Modify: `paper12/README.md`

- [ ] **Step 1: Update required experiments status**

In `submission/paper12_isprs_jprs_20260606/REQUIRED_EXPERIMENTS_ISPRS.md`, under `### 1. Second-backbone validation`, add this status block immediately after the Purpose paragraph:

```markdown
Current status:

- Prepared as an explicit SatMAE-compatible EuroSAT validation track.
- Config: `geoadapter/bench/configs/eurosat_second_backbone.yaml`.
- Notebook: `colab/paper12_second_backbone_eurosat_colab.ipynb`.
- Expected output files:
  - `/content/drive/MyDrive/paper12_results/second_backbone_eurosat.json`
  - `/content/drive/MyDrive/paper12_results/second_backbone_eurosat_summary.json`
- The matrix is 18 rows: 1 backbone x 3 methods x 2 modalities x 3 seeds.
- The manuscript must not use this evidence until both JSON files are mirrored
  into `paper12_results/` and audited.
```

- [ ] **Step 2: Update action-required technical checks**

In `submission/paper12_isprs_jprs_20260606/00_ACTION_REQUIRED.md`, under `## Technical Checks`, add:

```markdown
- Run or intentionally defer the second-backbone EuroSAT notebook:
  `colab/paper12_second_backbone_eurosat_colab.ipynb`.
- If completed, mirror `second_backbone_eurosat.json` and
  `second_backbone_eurosat_summary.json` from Drive into `paper12_results/` and
  the supplementary result directory before making manuscript claims from them.
```

- [ ] **Step 3: Update Paper12 README**

In `paper12/README.md`, under `## Data sources`, add:

```markdown
- `paper12_results/second_backbone_eurosat.json` and
  `paper12_results/second_backbone_eurosat_summary.json` - SatMAE-compatible
  second-backbone EuroSAT validation outputs. These files are expected after the
  Colab notebook run and should be treated as absent evidence until mirrored
  locally.
```

Under `## Status`, append:

```markdown
Second-backbone validation is prepared but not manuscript evidence until the
18-row Colab run is completed and result JSON files are mirrored into
`paper12_results/`.
```

- [ ] **Step 4: Verify status text**

Run:

```powershell
rg -n "second_backbone_eurosat|eurosat_second_backbone|paper12_second_backbone_eurosat_colab" submission\paper12_isprs_jprs_20260606 paper12\README.md
```

Expected output includes all three edited documents.

- [ ] **Step 5: Commit Task 5**

Run:

```powershell
git add submission/paper12_isprs_jprs_20260606/REQUIRED_EXPERIMENTS_ISPRS.md submission/paper12_isprs_jprs_20260606/00_ACTION_REQUIRED.md paper12/README.md
git commit -m "docs: record paper12 second-backbone experiment status"
```

## Task 6: Verification Pass

**Files:**
- Verify only.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
python -m pytest tests/test_backbone_factory.py tests/test_benchmark_runner.py tests/test_second_backbone_summary.py tests/test_paper12_colab_notebooks.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run existing Paper12 result contract tests**

Run:

```powershell
python -m pytest tests/test_paper12_public_dataset_results.py tests/test_paper12_review_audit.py -q
```

Expected: all selected tests pass, confirming existing Paper12 evidence contracts were not regressed.

- [ ] **Step 3: Run second-backbone dry-run**

Run:

```powershell
python -m geoadapter.bench.run_benchmark --config geoadapter/bench/configs/eurosat_second_backbone.yaml --dry-run
```

Expected: `Total experiments: 18`.

- [ ] **Step 4: Check whitespace**

Run:

```powershell
git diff --check
```

Expected: no output and exit code 0.

- [ ] **Step 5: Inspect final status**

Run:

```powershell
git status --short
git log -5 --oneline
```

Expected: working tree clean after the task commits.

## Self-Review

- Spec coverage:
  - Backbone factory: Task 1.
  - Preserve Prithvi path: Task 1 and Task 2 tests.
  - 18-row EuroSAT config: Task 2.
  - Generated Colab notebook: Task 4.
  - Summary schema: Task 3.
  - Status documents before manuscript edits: Task 5.
  - No manuscript claim changes before JSON evidence: Task 5.
  - Local offline tests: Tasks 1 through 4.
- Placeholder scan:
  - The plan contains concrete file paths, commands, and code snippets for every code-changing task.
  - The implementation target is concretely SatMAE-compatible.
  - The real pretrained checkpoint is an explicit Drive-staged file path, so unit tests do not download or depend on network access.
- Type consistency:
  - `BackboneSpec.name`, `model`, `feature_dim`, `input_channels`, and `blocks` are defined in Task 1 and consumed in Task 2.
  - Raw result fields `backbone`, `method`, `modality`, `seed`, `trainable_params`, `overall_accuracy`, and `macro_f1` are emitted in Task 2 and consumed in Task 3.
  - Notebook output filenames match the spec and status-document text.
