# Paper12 GeoVLM Prompt Segmentation MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a real SigLIP-conditioned Prithvi prompt-segmentation MVP for LandCoverAI buildings, roads, and water, with reproducible Colab training, offline inference, and automated acceptance gates.

**Architecture:** Keep the historical Paper12 segmentation path unchanged. Add an opt-in Prithvi positional-embedding path, a frozen SigLIP text tower, a FiLM plus dense-similarity binary decoder, and a separate prompt-training runner that compares against a no-text three-head baseline. Real-data results are accepted only through class IoU, held-out phrasing, counterfactual prompt sensitivity, paired bootstrap, and checkpoint-reproduction gates.

**Tech Stack:** Python 3.10+, PyTorch, torchvision, TorchGeo LandCoverAI, Hugging Face Transformers/SigLIP, PyYAML, NumPy, rasterio, Pillow, pytest, Google Colab.

---

## File Map

Create these focused modules:

- `geoadapter/data/prompt_segmentation.py`: official taxonomy, prompt YAML validation, binary-mask conversion, prompt sampling, and dataset positive-weight estimation.
- `geoadapter/models/text_encoder.py`: replaceable text-encoder interface and frozen SigLIP implementation.
- `geoadapter/models/prompt_segmentation.py`: FiLM/dense-similarity prompt model and no-text three-head baseline.
- `geoadapter/engine/prompt_segmentation.py`: BCE plus Dice objective and prompt-model training/prediction wrapper.
- `geoadapter/bench/geovlm_prompt_summary.py`: IoU/Dice aggregation, paired bootstrap, acceptance gates, and summary CLI.
- `geoadapter/bench/run_geovlm_prompt_segmentation.py`: real LandCoverAI experiment, checkpoint/resume, previews, manifests, and append-safe raw results.
- `geoadapter/inference/__init__.py` and `geoadapter/inference/prompt_segmentation.py`: checkpoint reconstruction and offline image/GeoTIFF inference.
- `scripts/run_geovlm_prompt_segmentation.py`: thin CLI entry point.
- `geoadapter/bench/configs/geovlm_prompt_segmentation.yaml`: one-seed/full-matrix training contract.
- `geoadapter/bench/configs/geovlm_prompts.yaml`: versioned disjoint prompt sets.
- `colab/paper12_geovlm_prompt_segmentation_colab.ipynb`: generated real experiment notebook.

Modify only these existing files:

- `geoadapter/models/prithvi.py`: add default-off checkpoint positional embeddings.
- `geoadapter/models/__init__.py`: export new model classes without importing Transformers eagerly.
- `pyproject.toml`: add a `geovlm` optional dependency group.
- `scripts/make_paper12_colab_notebooks.py`: generate the new notebook.
- `tests/test_paper12_colab_notebooks.py`: assert notebook/config contract.
- `paper12/README.md`: document the bounded GeoVLM experiment and commands without claiming success before results exist.

Add focused tests:

- `tests/test_prompt_segmentation_data.py`
- `tests/test_prithvi_position_embeddings.py`
- `tests/test_prompt_segmentation_model.py`
- `tests/test_prompt_segmentation_engine.py`
- `tests/test_geovlm_prompt_summary.py`
- `tests/test_geovlm_prompt_runner.py`
- `tests/test_geovlm_prompt_inference.py`

Do not edit existing Paper12 result JSON, manuscript claims, backend code, or frontend code in this implementation.

## Task 1: LandCoverAI Taxonomy and Prompt Data Contract

**Files:**
- Create: `geoadapter/data/prompt_segmentation.py`
- Create: `geoadapter/bench/configs/geovlm_prompts.yaml`
- Create: `tests/test_prompt_segmentation_data.py`
- Modify: `geoadapter/data/datasets.py:111-124`

- [ ] **Step 1: Write failing tests for the official five-class mapping and binary masks**

```python
import torch

from geoadapter.data.prompt_segmentation import (
    LANDCOVERAI_CLASSES,
    PROMPT_TARGET_CLASS_IDS,
    multiclass_to_binary,
    normalize_landcoverai_image,
)


def test_landcoverai_taxonomy_matches_torchgeo():
    assert LANDCOVERAI_CLASSES == (
        "background", "building", "woodland", "water", "road"
    )
    assert PROMPT_TARGET_CLASS_IDS == {"building": 1, "water": 3, "road": 4}


def test_multiclass_to_binary_uses_requested_official_index():
    mask = torch.tensor([[0, 1, 3], [4, 2, 1]])
    assert multiclass_to_binary(mask, 1).tolist() == [
        [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
    ]
    assert multiclass_to_binary(mask, 3).sum().item() == 1
    assert multiclass_to_binary(mask, 4).sum().item() == 1


def test_landcoverai_image_normalization_maps_bytes_to_unit_interval():
    image = torch.tensor([0.0, 127.5, 255.0]).reshape(1, 1, 3)
    normalized = normalize_landcoverai_image(image)
    assert torch.allclose(normalized, torch.tensor([0.0, 0.5, 1.0]).reshape(1, 1, 3))


def test_landcoverai_image_normalization_rejects_out_of_range_values():
    with pytest.raises(ValueError, match="0..255"):
        normalize_landcoverai_image(torch.tensor([[[300.0]]]))
```

- [ ] **Step 2: Run the tests and verify the module is missing**

Run: `python -m pytest tests/test_prompt_segmentation_data.py -v`

Expected: FAIL during import with `ModuleNotFoundError: geoadapter.data.prompt_segmentation`.

- [ ] **Step 3: Implement taxonomy validation and binary-mask conversion**

```python
# geoadapter/data/prompt_segmentation.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import yaml


LANDCOVERAI_CLASSES = ("background", "building", "woodland", "water", "road")
PROMPT_TARGET_CLASS_IDS = {"building": 1, "water": 3, "road": 4}


def validate_landcoverai_mask(mask: torch.Tensor) -> None:
    values = {int(value) for value in torch.unique(mask).tolist()}
    invalid = sorted(values - set(range(len(LANDCOVERAI_CLASSES))))
    if invalid:
        raise ValueError(f"LandCoverAI mask contains invalid class ids: {invalid}")


def multiclass_to_binary(mask: torch.Tensor, class_id: int) -> torch.Tensor:
    if class_id not in PROMPT_TARGET_CLASS_IDS.values():
        raise ValueError(f"unsupported prompt target class id: {class_id}")
    validate_landcoverai_mask(mask)
    return mask.eq(class_id).to(dtype=torch.float32)


def normalize_landcoverai_image(image: torch.Tensor) -> torch.Tensor:
    image = image.to(dtype=torch.float32)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("LandCoverAI image must have shape [3,H,W]")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 255:
        raise ValueError("LandCoverAI image values must be finite and in 0..255")
    return image / 255.0
```

Also correct only the inaccurate loader docstring in `geoadapter/data/datasets.py` from “6-class” to “5-class”; do not change existing historical configs in this task.

- [ ] **Step 4: Add failing prompt-config isolation tests**

```python
from pathlib import Path

import pytest

from geoadapter.data.prompt_segmentation import load_prompt_config


CONFIG = Path("geoadapter/bench/configs/geovlm_prompts.yaml")


def test_prompt_config_has_disjoint_seen_and_held_out_prompts():
    config = load_prompt_config(CONFIG)
    assert set(config.classes) == {"building", "road", "water"}
    for prompt_class in config.classes.values():
        assert prompt_class.training
        assert prompt_class.held_out
        assert set(prompt_class.training).isdisjoint(prompt_class.held_out)


def test_prompt_config_rejects_overlap(tmp_path):
    path = tmp_path / "prompts.yaml"
    path.write_text(
        "schema: paper12.geovlm_prompts.v1\nclasses:\n"
        "  building:\n    class_id: 1\n"
        "    training: ['find buildings']\n"
        "    held_out: ['find buildings']\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlap"):
        load_prompt_config(path)
```

- [ ] **Step 5: Create the exact prompt YAML and loader dataclasses**

```yaml
# geoadapter/bench/configs/geovlm_prompts.yaml
schema: paper12.geovlm_prompts.v1
language: en
classes:
  building:
    class_id: 1
    training:
      - segment all buildings
      - find the buildings
      - map building footprints
      - show built structures
    held_out:
      - extract every building visible in this aerial image
      - identify roofed structures
  road:
    class_id: 4
    training:
      - segment all roads
      - find the roads
      - map road surfaces
      - show the road network
    held_out:
      - extract the visible transportation routes
      - identify paved routes
  water:
    class_id: 3
    training:
      - segment all water bodies
      - find the water
      - map surface water
      - show lakes and rivers
    held_out:
      - extract visible aquatic areas
      - identify open water
```

```python
@dataclass(frozen=True)
class PromptClass:
    class_id: int
    training: tuple[str, ...]
    held_out: tuple[str, ...]


@dataclass(frozen=True)
class PromptConfig:
    schema: str
    language: str
    classes: Mapping[str, PromptClass]


def load_prompt_config(path: str | Path) -> PromptConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "paper12.geovlm_prompts.v1":
        raise ValueError("unsupported prompt schema")
    if payload.get("language") != "en":
        raise ValueError("MVP prompt language must be en")
    classes = {}
    for name, expected_id in PROMPT_TARGET_CLASS_IDS.items():
        raw = payload.get("classes", {}).get(name)
        if raw is None or int(raw.get("class_id", -1)) != expected_id:
            raise ValueError(f"invalid class mapping for {name}")
        training = tuple(str(value).strip() for value in raw.get("training", ()))
        held_out = tuple(str(value).strip() for value in raw.get("held_out", ()))
        if not training or not held_out or any(not value for value in training + held_out):
            raise ValueError(f"empty prompt set for {name}")
        if set(training) & set(held_out):
            raise ValueError(f"training/held-out prompt overlap for {name}")
        classes[name] = PromptClass(expected_id, training, held_out)
    return PromptConfig(payload["schema"], payload["language"], classes)
```

- [ ] **Step 6: Add failing tests for balanced prompt-batch sampling and empty cap**

```python
from geoadapter.data.prompt_segmentation import sample_prompt_batch


def test_sample_prompt_batch_builds_binary_targets_and_prompts():
    masks = torch.stack([
        torch.tensor([[1, 0], [0, 0]]),
        torch.tensor([[3, 3], [0, 0]]),
        torch.tensor([[4, 0], [4, 0]]),
    ])
    config = load_prompt_config(CONFIG)
    batch = sample_prompt_batch(
        masks, config, generator=torch.Generator().manual_seed(7), empty_cap=0.25
    )
    assert batch.class_ids.shape == (3,)
    assert batch.targets.shape == (3, 2, 2)
    assert len(batch.prompts) == 3
    assert batch.targets.dtype == torch.float32
    assert batch.voluntary_empty_count <= int(len(masks) * 0.25)


def test_sample_prompt_batch_rejects_invalid_empty_cap():
    config = load_prompt_config(CONFIG)
    with pytest.raises(ValueError, match="empty_cap"):
        sample_prompt_batch(torch.zeros(1, 2, 2), config, empty_cap=1.5)
```

- [ ] **Step 7: Implement the sampled batch contract**

```python
@dataclass(frozen=True)
class PromptBatch:
    class_ids: torch.Tensor
    class_names: tuple[str, ...]
    prompts: tuple[str, ...]
    targets: torch.Tensor
    empty_count: int
    voluntary_empty_count: int


def sample_prompt_batch(
    masks: torch.Tensor,
    config: PromptConfig,
    *,
    generator: torch.Generator | None = None,
    empty_cap: float = 0.25,
) -> PromptBatch:
    if masks.ndim != 3:
        raise ValueError("masks must have shape [B,H,W]")
    if not 0.0 <= empty_cap <= 1.0:
        raise ValueError("empty_cap must be between 0 and 1")
    names = tuple(PROMPT_TARGET_CLASS_IDS)
    batch_size = int(masks.shape[0])
    repeated = (names * ((batch_size + len(names) - 1) // len(names)))[:batch_size]
    permutation = torch.randperm(batch_size, generator=generator).tolist()
    proposed = [repeated[index] for index in permutation]
    voluntary_empty_budget = int(batch_size * empty_cap)
    voluntary_empty_count = 0
    selected_names: list[str] = []
    prompts: list[str] = []
    targets: list[torch.Tensor] = []
    for mask, proposed_name in zip(masks, proposed):
        validate_landcoverai_mask(mask)
        present = [name for name in names if mask.eq(PROMPT_TARGET_CLASS_IDS[name]).any()]
        forced_empty = not present
        proposed_is_empty = proposed_name not in present
        if proposed_is_empty and not forced_empty and voluntary_empty_count >= voluntary_empty_budget:
            index = int(torch.randint(len(present), (), generator=generator))
            name = present[index]
        else:
            name = proposed_name
            if proposed_is_empty and not forced_empty:
                voluntary_empty_count += 1
        prompt_values = config.classes[name].training
        prompt_index = int(torch.randint(len(prompt_values), (), generator=generator))
        selected_names.append(name)
        prompts.append(prompt_values[prompt_index])
        targets.append(multiclass_to_binary(mask, PROMPT_TARGET_CLASS_IDS[name]))
    target_tensor = torch.stack(targets)
    return PromptBatch(
        class_ids=torch.tensor(
            [PROMPT_TARGET_CLASS_IDS[name] for name in selected_names], dtype=torch.long
        ),
        class_names=tuple(selected_names),
        prompts=tuple(prompts),
        targets=target_tensor,
        empty_count=int(target_tensor.flatten(1).sum(dim=1).eq(0).sum()),
        voluntary_empty_count=voluntary_empty_count,
    )
```

- [ ] **Step 8: Run focused tests and commit**

Run: `python -m pytest tests/test_prompt_segmentation_data.py tests/test_datasets.py -v`

Expected: PASS.

```bash
git add geoadapter/data/prompt_segmentation.py geoadapter/data/datasets.py geoadapter/bench/configs/geovlm_prompts.yaml tests/test_prompt_segmentation_data.py
git commit -m "feat: add GeoVLM prompt data contract"
```

## Task 2: Opt-In Prithvi Checkpoint Position Embeddings

**Files:**
- Modify: `geoadapter/models/prithvi.py`
- Create: `tests/test_prithvi_position_embeddings.py`
- Test: `tests/test_prithvi.py`

- [ ] **Step 1: Write failing parsing/interpolation tests**

```python
import pytest
import torch

from geoadapter.models.prithvi import PrithviBackbone


def _position_tensor(embed_dim=8):
    return torch.arange(589 * embed_dim, dtype=torch.float32).reshape(1, 589, embed_dim)


def test_checkpoint_positions_reduce_three_temporal_grids_and_interpolate():
    model = PrithviBackbone(
        pretrained=False, embed_dim=8, depth=1, num_heads=2,
        use_checkpoint_position_embeddings=True,
    )
    model.set_checkpoint_position_embedding(_position_tensor())
    cls, patch = model.interpolate_checkpoint_positions((8, 10))
    assert cls.shape == (1, 1, 8)
    assert patch.shape == (1, 80, 8)
    assert torch.equal(cls, _position_tensor()[:, :1])


def test_checkpoint_positions_reject_unexpected_token_count():
    model = PrithviBackbone(
        pretrained=False, embed_dim=8, depth=1, num_heads=2,
        use_checkpoint_position_embeddings=True,
    )
    with pytest.raises(ValueError, match="589"):
        model.set_checkpoint_position_embedding(torch.zeros(1, 590, 8))


def test_default_path_remains_position_free():
    model = PrithviBackbone(pretrained=False, embed_dim=8, depth=1, num_heads=2)
    assert model.use_checkpoint_position_embeddings is False
```

- [ ] **Step 2: Run tests to verify constructor incompatibility**

Run: `python -m pytest tests/test_prithvi_position_embeddings.py -v`

Expected: FAIL because `use_checkpoint_position_embeddings` and position methods do not exist.

- [ ] **Step 3: Add a default-off buffer, parser, and rectangular interpolation**

Add `import math` and `import torch.nn.functional as F`, then extend the constructor and class:

```python
def __init__(
    self,
    pretrained: bool = True,
    checkpoint_path: str | None = None,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    in_chans: int = 6,
    patch_size: int = 16,
    use_checkpoint_position_embeddings: bool = False,
):
    super().__init__()
    self.embed_dim = embed_dim
    self.use_checkpoint_position_embeddings = use_checkpoint_position_embeddings
    self.register_buffer("checkpoint_cls_position", torch.empty(0), persistent=False)
    self.register_buffer("checkpoint_patch_positions", torch.empty(0), persistent=False)
    # Preserve the existing patch embedding, blocks, checkpoint load, and freeze code.


def set_checkpoint_position_embedding(self, position: torch.Tensor) -> None:
    expected_tokens = 1 + 3 * 14 * 14
    if tuple(position.shape) != (1, expected_tokens, self.embed_dim):
        raise ValueError(
            f"expected Prithvi position embedding [1,589,{self.embed_dim}], "
            f"got {list(position.shape)}"
        )
    self.checkpoint_cls_position = position[:, :1].detach().clone()
    temporal = position[:, 1:].reshape(1, 3, 14, 14, self.embed_dim)
    self.checkpoint_patch_positions = temporal.mean(dim=1).permute(0, 3, 1, 2).contiguous()


def interpolate_checkpoint_positions(
    self, spatial_dims: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    if self.checkpoint_patch_positions.numel() == 0:
        raise RuntimeError("checkpoint position embedding was not loaded")
    patch = F.interpolate(
        self.checkpoint_patch_positions,
        size=spatial_dims,
        mode="bilinear",
        align_corners=False,
    )
    patch = patch.flatten(2).transpose(1, 2)
    return self.checkpoint_cls_position, patch
```

In `_load_checkpoint`, capture `state.get("encoder.pos_embed")` before key remapping. If the opt-in flag is true, require it and call `set_checkpoint_position_embedding`; if false, preserve current behavior exactly.

In `forward`, after concatenating CLS and patch tokens, add the interpolated CLS and patch positions only when the flag is true:

```python
if self.use_checkpoint_position_embeddings:
    cls_position, patch_positions = self.interpolate_checkpoint_positions((h, w))
    x = x + torch.cat([cls_position, patch_positions], dim=1).to(
        device=x.device, dtype=x.dtype
    )
```

- [ ] **Step 4: Add a checkpoint-load regression test**

Build a tiny checkpoint containing `encoder.pos_embed`, matching patch weights,
CLS, norm, and one block state. Assert the opt-in model populates both buffers,
while a default model keeps both empty. Use a temporary file and do not load the
real 454 MB checkpoint in unit tests.

- [ ] **Step 5: Run legacy and new tests, then commit**

Run: `python -m pytest tests/test_prithvi.py tests/test_prithvi_position_embeddings.py tests/test_backbone_factory.py -v`

Expected: PASS, including the legacy default path.

```bash
git add geoadapter/models/prithvi.py tests/test_prithvi_position_embeddings.py
git commit -m "feat: add opt-in Prithvi position embeddings"
```

## Task 3: Frozen SigLIP Text Encoder and Prompt Models

**Files:**
- Create: `geoadapter/models/text_encoder.py`
- Create: `geoadapter/models/prompt_segmentation.py`
- Modify: `geoadapter/models/__init__.py`
- Modify: `pyproject.toml`
- Create: `tests/test_prompt_segmentation_model.py`

- [ ] **Step 1: Write a fake encoder and failing model tests**

```python
import torch
import torch.nn as nn

from geoadapter.models.prompt_segmentation import (
    PromptSegmentationModel,
    ThreeHeadSegmentationBaseline,
)


class FakeTextEncoder(nn.Module):
    output_dim = 6

    def forward(self, prompts):
        rows = []
        for prompt in prompts:
            code = sum(ord(char) for char in prompt)
            rows.append(torch.tensor([(code + i * 17) % 31 for i in range(6)]).float())
        return torch.stack(rows).to(next(self.parameters(), torch.empty(0)).device)


class TinySpatialBackbone(nn.Module):
    embed_dim = 8

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(6, 8, 4, stride=4)

    def forward(self, images, return_spatial=False):
        features = self.conv(images)
        h, w = features.shape[-2:]
        tokens = features.flatten(2).transpose(1, 2)
        return (tokens, (h, w)) if return_spatial else tokens.mean(dim=1)


def test_prompt_model_output_shape_and_prompt_dependence():
    torch.manual_seed(2)
    model = PromptSegmentationModel(
        TinySpatialBackbone(), FakeTextEncoder(), visual_dim=8,
        text_dim=6, condition_dim=5, decoder_dim=7, patch_size=4,
    )
    images = torch.randn(2, 6, 16, 20)
    first = model(images, ["find buildings", "find buildings"])
    second = model(images, ["find water", "find water"])
    assert first.shape == (2, 16, 20)
    assert not torch.allclose(first, second)


def test_three_head_baseline_selects_requested_target_channel():
    model = ThreeHeadSegmentationBaseline(
        TinySpatialBackbone(), visual_dim=8, decoder_dim=7, patch_size=4
    )
    logits = model(torch.randn(2, 6, 16, 20), torch.tensor([1, 4]))
    assert logits.shape == (2, 16, 20)
```

- [ ] **Step 2: Verify the new model imports fail**

Run: `python -m pytest tests/test_prompt_segmentation_model.py -v`

Expected: FAIL with missing module/classes.

- [ ] **Step 3: Implement a lazy, frozen SigLIP wrapper**

```python
# geoadapter/models/text_encoder.py
from __future__ import annotations

import torch
import torch.nn as nn


class SiglipTextEncoder(nn.Module):
    def __init__(
        self,
        model_id: str = "google/siglip-base-patch16-224",
        *,
        revision: str | None = None,
        local_files_only: bool = False,
    ):
        super().__init__()
        try:
            from transformers import AutoTokenizer, SiglipTextModel
        except ImportError as exc:
            raise ImportError(
                "Install GeoVLM dependencies with pip install -e '.[geovlm]'"
            ) from exc
        self.model_id = model_id
        self.revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, local_files_only=local_files_only
        )
        self.model = SiglipTextModel.from_pretrained(
            model_id, revision=revision, local_files_only=local_files_only
        )
        self.output_dim = int(self.model.config.hidden_size)
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, prompts: list[str] | tuple[str, ...]) -> torch.Tensor:
        if not prompts or any(not prompt.strip() for prompt in prompts):
            raise ValueError("prompts must contain non-empty text")
        device = next(self.model.parameters()).device
        tokens = self.tokenizer(list(prompts), padding=True, return_tensors="pt")
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs.pooler_output
```

- [ ] **Step 4: Implement FiLM, dense similarity, and the baseline**

```python
# geoadapter/models/prompt_segmentation.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


TARGET_CLASS_TO_CHANNEL = {1: 0, 3: 1, 4: 2}


class _BinaryDecoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    def forward(self, features, output_size):
        logits = self.net(features)
        return F.interpolate(
            logits, size=output_size, mode="bilinear", align_corners=False
        )[:, 0]


class PromptSegmentationModel(nn.Module):
    def __init__(
        self, backbone, text_encoder, *, visual_dim=768, text_dim=768,
        condition_dim=256, decoder_dim=128, patch_size=16,
    ):
        super().__init__()
        self.backbone = backbone
        self.text_encoder = text_encoder
        self.patch_size = patch_size
        self.visual_similarity = nn.Linear(visual_dim, condition_dim)
        self.text_projection = nn.Linear(text_dim, condition_dim)
        self.visual_decoder = nn.Linear(visual_dim, decoder_dim)
        self.film = nn.Sequential(
            nn.Linear(condition_dim, decoder_dim * 2), nn.GELU(),
            nn.Linear(decoder_dim * 2, decoder_dim * 2),
        )
        self.logit_scale = nn.Parameter(torch.tensor(0.0))
        self.decoder = _BinaryDecoder(decoder_dim + 1, decoder_dim)

    def forward(self, images, prompts):
        if len(prompts) != images.shape[0]:
            raise ValueError("one prompt is required per image")
        tokens, (h, w) = self.backbone(images, return_spatial=True)
        text = self.text_encoder(prompts).to(tokens.device)
        visual_semantic = F.normalize(self.visual_similarity(tokens), dim=-1)
        text_semantic = F.normalize(self.text_projection(text), dim=-1)
        similarity = torch.einsum("bnd,bd->bn", visual_semantic, text_semantic)
        similarity = similarity * self.logit_scale.clamp(max=4.6052).exp()
        visual = self.visual_decoder(tokens)
        gamma, beta = self.film(text_semantic).chunk(2, dim=-1)
        visual = visual * (1.0 + gamma[:, None]) + beta[:, None]
        visual = visual.transpose(1, 2).reshape(images.shape[0], -1, h, w)
        similarity = similarity.reshape(images.shape[0], 1, h, w)
        return self.decoder(torch.cat([visual, similarity], dim=1), images.shape[-2:])


class ThreeHeadSegmentationBaseline(nn.Module):
    def __init__(self, backbone, *, visual_dim=768, decoder_dim=128, patch_size=16):
        super().__init__()
        self.backbone = backbone
        self.decoder = nn.Sequential(
            nn.Conv2d(visual_dim, decoder_dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(decoder_dim, 3, 1),
        )

    def forward(self, images, conditions):
        class_ids = conditions
        tokens, (h, w) = self.backbone(images, return_spatial=True)
        features = tokens.transpose(1, 2).reshape(images.shape[0], -1, h, w)
        logits = F.interpolate(
            self.decoder(features), size=images.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        channels = torch.tensor(
            [TARGET_CLASS_TO_CHANNEL[int(value)] for value in class_ids],
            device=logits.device,
        )
        return logits[torch.arange(logits.shape[0], device=logits.device), channels]
```

Both model classes use `forward(images, conditions)`: prompt conditions are a
sequence of strings and baseline conditions are a class-id tensor. The runner
will apply `ZeroPadAdapter` before either model, so these classes receive
six-channel images and stay independent of modality bridging.

- [ ] **Step 5: Add the optional dependency group without eager imports**

```toml
geovlm = [
    "transformers>=4.46,<5",
    "sentencepiece>=0.2",
    "safetensors>=0.4",
    "huggingface-hub>=0.26",
]
```

Export `PromptSegmentationModel` and `ThreeHeadSegmentationBaseline` from
`geoadapter/models/__init__.py`, but do not export or instantiate
`SiglipTextEncoder` there; this keeps core imports working without Transformers.

- [ ] **Step 6: Add gradient/freeze assertions**

Extend the model test to backpropagate `model(images, prompts).mean()` and assert
gradients exist on `visual_similarity`, `text_projection`, `film`,
`logit_scale`, and `decoder`. Add a fake text encoder containing a frozen
parameter and assert it has no gradient. A separate mocked Transformers test
must assert every `SiglipTextEncoder.model` parameter has
`requires_grad=False` and the wrapped tower remains in eval mode after
`encoder.train()`.

- [ ] **Step 7: Run focused tests and commit**

Run: `python -m pytest tests/test_prompt_segmentation_model.py tests/test_heads.py -v`

Expected: PASS without downloading SigLIP.

```bash
git add geoadapter/models/text_encoder.py geoadapter/models/prompt_segmentation.py geoadapter/models/__init__.py pyproject.toml tests/test_prompt_segmentation_model.py
git commit -m "feat: add SigLIP-conditioned segmentation model"
```

## Task 4: Prompt Loss and Training Engine

**Files:**
- Create: `geoadapter/engine/prompt_segmentation.py`
- Modify: `geoadapter/engine/__init__.py`
- Create: `tests/test_prompt_segmentation_engine.py`

- [ ] **Step 1: Write failing BCE/Dice tests**

```python
import torch

from geoadapter.engine.prompt_segmentation import PromptSegmentationLoss


def test_prompt_loss_is_lower_for_correct_logits():
    targets = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    correct = torch.tensor([[[5.0, -5.0], [-5.0, 5.0]]])
    wrong = -correct
    loss = PromptSegmentationLoss()
    assert loss(correct, targets, torch.tensor([2.0])) < loss(
        wrong, targets, torch.tensor([2.0])
    )


def test_prompt_loss_is_finite_for_empty_target():
    value = PromptSegmentationLoss()(
        torch.zeros(2, 4, 4), torch.zeros(2, 4, 4), torch.ones(2)
    )
    assert torch.isfinite(value)
```

- [ ] **Step 2: Implement per-example positive weighting and Dice**

```python
class PromptSegmentationLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, epsilon=1e-6):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.epsilon = float(epsilon)

    def forward(self, logits, targets, positive_weights):
        if logits.shape != targets.shape:
            raise ValueError("logits and targets must have identical shapes")
        targets = targets.to(dtype=logits.dtype)
        pixel_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        weights = torch.where(
            targets > 0.5,
            positive_weights.to(logits.device)[:, None, None],
            torch.ones_like(targets),
        )
        bce = (pixel_loss * weights).mean()
        probabilities = logits.sigmoid().flatten(1)
        target_flat = targets.flatten(1)
        intersection = (probabilities * target_flat).sum(dim=1)
        dice = (2 * intersection + self.epsilon) / (
            probabilities.sum(dim=1) + target_flat.sum(dim=1) + self.epsilon
        )
        return self.bce_weight * bce + self.dice_weight * (1 - dice).mean()
```

- [ ] **Step 3: Write a failing tiny end-to-end trainer test**

Use the fake text encoder and tiny spatial backbone from Task 3. Train four
deterministic geometric examples for 20 steps and assert:

```python
trainer = PromptSegmentationTrainer(model, lr=1e-2, epochs=20, device="cpu")
first = trainer.train_step(images, prompts, targets, positive_weights)
for _ in range(19):
    last = trainer.train_step(images, prompts, targets, positive_weights)
assert last < first

before = trainer.predict(images, ["find buildings"] * len(images))
state = trainer.state_dict(epoch=20, metadata={"schema": "test"})
clone = PromptSegmentationTrainer(clone_model, lr=1e-2, epochs=20, device="cpu")
clone.load_state_dict(state)
after = clone.predict(images, ["find buildings"] * len(images))
assert torch.allclose(before, after, atol=1e-6)
```

- [ ] **Step 4: Implement the isolated trainer**

```python
class PromptSegmentationTrainer:
    def __init__(
        self, model, *, lr=1e-3, lr_peft=1e-4, epochs=50, device="cpu",
        loss=None,
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = loss or PromptSegmentationLoss()
        prompt_params, peft_params = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            (peft_params if "houlsby_adapter" in name else prompt_params).append(parameter)
        groups = [{"params": prompt_params, "lr": lr}]
        if peft_params:
            groups.append({"params": peft_params, "lr": lr_peft})
        self.optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

    def train_step(self, images, conditions, targets, positive_weights):
        self.model.train()
        images = images.to(self.device)
        targets = targets.to(self.device)
        positive_weights = positive_weights.to(self.device)
        self.optimizer.zero_grad()
        logits = self.model(images, conditions)
        loss = self.criterion(logits, targets, positive_weights)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())

    @torch.no_grad()
    def predict(self, images, conditions):
        self.model.eval()
        return self.model(images.to(self.device), conditions)

    def state_dict(self, *, epoch, metadata):
        return {
            "epoch": int(epoch),
            "trainable_model": {
                name: parameter.detach().cpu()
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            },
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "metadata": dict(metadata),
        }

    def load_state_dict(self, state):
        missing, unexpected = self.model.load_state_dict(
            state["trainable_model"], strict=False
        )
        if unexpected:
            raise ValueError(f"unexpected checkpoint parameters: {unexpected}")
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        return int(state["epoch"]), dict(state["metadata"])
```

Use the same trainer for both methods: pass prompt strings to the prompt model
and class-id tensors to the baseline. Do not add baseline branches to the prompt
model itself. The lightweight checkpoint intentionally saves only trainable
parameters; frozen Prithvi and SigLIP tensors are reconstructed from the
metadata identifiers and hashes.

- [ ] **Step 5: Run focused tests and commit**

Run: `python -m pytest tests/test_prompt_segmentation_engine.py tests/test_prompt_segmentation_model.py -v`

Expected: PASS.

```bash
git add geoadapter/engine/prompt_segmentation.py geoadapter/engine/__init__.py tests/test_prompt_segmentation_engine.py
git commit -m "feat: add prompt segmentation training engine"
```

## Task 5: Evaluation Metrics, Counterfactual Bootstrap, and Gates

**Files:**
- Create: `geoadapter/bench/geovlm_prompt_summary.py`
- Create: `tests/test_geovlm_prompt_summary.py`

- [ ] **Step 1: Write failing metric and gate tests**

```python
import pytest

from geoadapter.bench.geovlm_prompt_summary import build_summary


def _passing_rows():
    rows = []
    for seed in (42, 123, 456):
        for name, iou in (("building", 0.50), ("road", 0.40), ("water", 0.45)):
            rows.append({
                "method": "siglip_film_dense_similarity_houlsby",
                "seed": seed,
                "class_name": name,
                "seen_iou": iou,
                "held_out_iou": iou * 0.95,
                "correct_iou_by_sample": [iou, iou + 0.02, iou - 0.01],
                "wrong_iou_by_sample": [iou - 0.20, iou - 0.18, iou - 0.22],
                "prompt_probability_change_by_sample": [0.08, 0.09, 0.07],
            })
    return rows


def test_summary_passes_all_confirmed_mvp_gates():
    summary = build_summary(_passing_rows(), bootstrap_iterations=1000, seed=7)
    assert summary["schema"] == "paper12.geovlm_prompt_summary.v1"
    assert summary["mvp_status"] == "passed"
    assert summary["failed_gates"] == []


def test_summary_reports_failed_class_gate():
    rows = _passing_rows()
    for row in rows:
        if row["class_name"] == "road":
            row["seen_iou"] = 0.20
            row["held_out_iou"] = 0.19
    summary = build_summary(rows, bootstrap_iterations=100, seed=7)
    assert summary["mvp_status"] == "failed"
    assert "class_iou:road<0.25" in summary["failed_gates"]
```

- [ ] **Step 2: Implement foreground IoU/Dice and deterministic paired bootstrap**

```python
def binary_metrics(target, prediction):
    target = np.asarray(target, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    intersection = np.logical_and(target, prediction).sum()
    union = np.logical_or(target, prediction).sum()
    iou = 1.0 if union == 0 else float(intersection / union)
    denom = target.sum() + prediction.sum()
    dice = 1.0 if denom == 0 else float(2 * intersection / denom)
    return {"foreground_iou": iou, "dice": dice}


def paired_bootstrap_delta(correct, wrong, *, iterations=1000, seed=0):
    correct = np.asarray(correct, dtype=float)
    wrong = np.asarray(wrong, dtype=float)
    if correct.shape != wrong.shape or correct.size == 0:
        raise ValueError("paired bootstrap inputs must be non-empty and aligned")
    deltas = correct - wrong
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=float)
    for index in range(iterations):
        draw = rng.integers(0, deltas.size, size=deltas.size)
        samples[index] = deltas[draw].mean()
    return {
        "mean_delta": float(deltas.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }
```

- [ ] **Step 3: Implement exact acceptance gates and JSON CLI**

`build_summary` must group only the prompt method rows across seeds and classes,
compute mean seen and held-out IoU, concatenate paired per-sample arrays, and
produce these exact gates:

```python
gates = {
    "mean_foreground_iou": mean_iou >= 0.40,
    "each_class_iou": all(value >= 0.25 for value in class_iou.values()),
    "held_out_retention": held_out_iou >= 0.90 * seen_iou,
    "correct_minus_wrong_iou": bootstrap["mean_delta"] >= 0.10,
    "counterfactual_ci_positive": bootstrap["ci95_low"] > 0.0,
    "prompt_probability_change": probability_change >= 0.05,
}
```

Emit stable failed reasons, including class names and thresholds. Add a CLI:

```bash
python -m geoadapter.bench.geovlm_prompt_summary \
  --input paper12_results/geovlm_prompt_segmentation.json \
  --output paper12_results/geovlm_prompt_segmentation_summary.json \
  --bootstrap-iterations 1000
```

Validate unique `(method, seed, class_name)` rows, supported seed/class values,
finite values, aligned per-sample arrays, and no `synthetic_fallback=true` row.
Missing required seeds or classes are reported as stable incomplete reasons
such as `missing_seed:123` or `missing_class:water`; this lets the seed-42 stage
produce an honest `mvp_status: incomplete` summary instead of crashing before
the full matrix exists. Once all required rows exist, `mvp_status` is `passed`
only when every metric gate passes and is otherwise `failed`.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_geovlm_prompt_summary.py -v`

Expected: PASS.

```bash
git add geoadapter/bench/geovlm_prompt_summary.py tests/test_geovlm_prompt_summary.py
git commit -m "feat: add GeoVLM evaluation gates"
```

## Task 6: Real Experiment Config, Runner, Checkpoints, and Previews

**Files:**
- Create: `geoadapter/bench/configs/geovlm_prompt_segmentation.yaml`
- Create: `geoadapter/bench/run_geovlm_prompt_segmentation.py`
- Create: `tests/test_geovlm_prompt_runner.py`

- [ ] **Step 1: Write a failing config-contract test**

```python
from pathlib import Path
import yaml


def test_geovlm_prompt_config_is_real_data_only():
    cfg = yaml.safe_load(Path(
        "geoadapter/bench/configs/geovlm_prompt_segmentation.yaml"
    ).read_text(encoding="utf-8"))
    assert cfg["experiment"]["dataset"] == "landcoverai"
    assert cfg["experiment"]["source_num_classes"] == 5
    assert cfg["experiment"]["target_classes"] == ["building", "road", "water"]
    assert cfg["experiment"]["seeds"] == [42, 123, 456]
    assert cfg["experiment"]["allow_synthetic_fallback"] is False
    assert cfg["text_encoder"]["model_id"] == "google/siglip-base-patch16-224"
    assert cfg["prithvi"]["use_checkpoint_position_embeddings"] is True
    assert cfg["methods"] == [
        "siglip_film_dense_similarity_houlsby",
        "no_text_three_binary_heads_houlsby",
    ]
```

- [ ] **Step 2: Create the exact real experiment YAML**

```yaml
experiment:
  name: paper12_geovlm_prompt_segmentation
  dataset: landcoverai
  dataset_root: ./data/landcoverai
  source_num_classes: 5
  target_classes: [building, road, water]
  epochs: 50
  batch_size: 8
  seeds: [42, 123, 456]
  empty_target_cap: 0.25
  allow_synthetic_fallback: false
  prompt_config: geoadapter/bench/configs/geovlm_prompts.yaml
methods:
  - siglip_film_dense_similarity_houlsby
  - no_text_three_binary_heads_houlsby
prithvi:
  checkpoint: data/weights/prithvi/Prithvi_100M.pt
  use_checkpoint_position_embeddings: true
  input_channels: 6
  patch_size: 16
peft:
  type: houlsby
  bottleneck_dim: 64
text_encoder:
  model_id: google/siglip-base-patch16-224
  revision: null
  local_files_only: false
model:
  condition_dim: 256
  decoder_dim: 128
training:
  lr: 0.001
  lr_peft: 0.0001
  bce_weight: 1.0
  dice_weight: 1.0
  positive_weight_clip: [1.0, 20.0]
evaluation:
  threshold: 0.5
  bootstrap_iterations: 1000
  preview_count: 12
```

- [ ] **Step 3: Write failing pure-helper tests before the runner**

Test these pure functions without TorchGeo, SigLIP, or the real checkpoint:

```python
from geoadapter.bench.run_geovlm_prompt_segmentation import (
    checkpoint_metadata,
    completed_keys,
    estimate_positive_weights,
    sha256_file,
)


def test_completed_keys_are_method_seed_pairs():
    rows = [{"method": "prompt", "seed": 42}, {"method": "baseline", "seed": 42}]
    assert completed_keys(rows) == {("prompt", 42), ("baseline", 42)}


def test_positive_weights_are_class_specific_and_clipped():
    masks = [torch.tensor([[1, 0], [3, 4]]), torch.tensor([[0, 0], [0, 0]])]
    weights = estimate_positive_weights(masks, clip=(1.0, 20.0))
    assert set(weights) == {"building", "road", "water"}
    assert all(1.0 <= value <= 20.0 for value in weights.values())
```

- [ ] **Step 4: Implement runner assembly with explicit dependency failures**

The runner must expose `build_model(config, method, device)` and:

1. require the Prithvi checkpoint path;
2. instantiate `PrithviBackbone(..., use_checkpoint_position_embeddings=True)`;
3. inject Houlsby into every block;
4. wrap input with `ZeroPadAdapter(3, 6)` in a small `InputAdaptedModel` module;
5. instantiate frozen `SiglipTextEncoder` only for the prompt method;
6. build `PromptSegmentationModel` or `ThreeHeadSegmentationBaseline`;
7. verify frozen/trainable parameter groups and print both counts.

Do not call or modify `run_benchmark.py`; this task has different batch and
text contracts.

Wrap the TorchGeo dataset with a prompt-specific view that applies
`normalize_landcoverai_image` and validates mask values before any sampler or
model sees a batch. The runner must assert normalized tensors are finite and
within `[0,1]`. Do not change `_SegmentationDataset` globally because historical
Paper12 runs consumed the previous scale and must remain reproducible.

- [ ] **Step 5: Implement append-safe training and checkpoint metadata**

The CLI is:

```bash
python -m geoadapter.bench.run_geovlm_prompt_segmentation \
  --config geoadapter/bench/configs/geovlm_prompt_segmentation.yaml \
  --output paper12_results/geovlm_prompt_segmentation.json \
  --summary-output paper12_results/geovlm_prompt_segmentation_summary.json \
  --checkpoint-dir /path/to/checkpoints \
  --preview-dir /path/to/previews \
  --stage seed42
```

Supported stages are `seed42` and `full`. `seed42` runs seed 42 for the prompt
method first, validates finite/decreasing loss, nonconstant predictions for all
three classes, prompt-dependent probability maps, and checkpoint reload. `full`
runs both methods for all three seeds and skips only completed `(method, seed)`
pairs already present in the raw JSON.

Save checkpoints atomically through a sibling `.tmp` file and `Path.replace`.
Metadata must include:

```python
{
    "schema": "paper12.geovlm_prompt_checkpoint.v1",
    "method": method,
    "seed": seed,
    "prithvi_sha256": sha256_file(prithvi_path),
    "position_policy": "mean_temporal_3x14x14_then_bilinear",
    "image_normalization": "rgb_float32_divide_255",
    "siglip_model_id": config["text_encoder"]["model_id"],
    "siglip_revision": config["text_encoder"].get("revision"),
    "prompt_config_sha256": sha256_file(prompt_path),
    "class_mapping": PROMPT_TARGET_CLASS_IDS,
    "condition_dim": config["model"]["condition_dim"],
    "decoder_dim": config["model"]["decoder_dim"],
    "threshold": config["evaluation"]["threshold"],
    "dependency_versions": dependency_versions(),
}
```

On resume or inference, validate schema, class mapping, dimensions, Prithvi
hash, prompt hash, model id, and image-normalization policy before loading
trainable state.

- [ ] **Step 6: Implement deterministic evaluation rows and previews**

For every `(method, seed, class)` row, store seen/held-out IoU and Dice,
foreground share, empty-mask rate, inference latency, and trainable/frozen
parameter counts. For the prompt method, also store aligned arrays for correct
IoU, mean wrong-prompt IoU, and probability-map changes. Use every seen and
held-out prompt, average each image/class before appending the per-sample arrays,
and preserve validation sample order.

Write at most `preview_count` PNG panels per seed containing RGB, target,
probability, and thresholded prediction. Preview filenames include seed, class,
prompt split, and stable sample index. Store only preview paths in JSON; do not
embed image bytes.

After every completed seed, atomically rewrite raw JSON and call
`build_summary`. The summary remains `mvp_status: incomplete` until all required
three-seed rows exist; it becomes `failed` or `passed` only after the complete
matrix is evaluated.

- [ ] **Step 7: Add runner tests with injected tiny builders**

Monkeypatch dataset/model builders so a two-epoch CPU test exercises:

- no network and no real data;
- both methods;
- checkpoint save/reload;
- append/skip behavior;
- prompt fields present only on the prompt method;
- `synthetic_fallback` absent or false;
- incomplete summary when only seed 42 exists.

The runner’s real path must never catch dataset/model exceptions and replace
them with synthetic data.

- [ ] **Step 8: Run tests and commit**

Run: `python -m pytest tests/test_geovlm_prompt_runner.py tests/test_geovlm_prompt_summary.py tests/test_prompt_segmentation_engine.py -v`

Expected: PASS.

```bash
git add geoadapter/bench/configs/geovlm_prompt_segmentation.yaml geoadapter/bench/run_geovlm_prompt_segmentation.py tests/test_geovlm_prompt_runner.py
git commit -m "feat: add GeoVLM prompt experiment runner"
```

## Task 7: Offline Checkpoint Inference and Geospatial Outputs

**Files:**
- Create: `geoadapter/inference/__init__.py`
- Create: `geoadapter/inference/prompt_segmentation.py`
- Create: `scripts/run_geovlm_prompt_segmentation.py`
- Create: `tests/test_geovlm_prompt_inference.py`

- [ ] **Step 1: Write failing non-geospatial and GeoTIFF output tests**

Use a tiny checkpoint and injected fake model builder. For PNG input, assert the
service writes:

- `<stem>_mask.png`;
- `<stem>_probability.npy`;
- `<stem>_preview.png`;
- `<stem>_metadata.json`.

For a projected GeoTIFF, assert it writes a one-band uint8 GeoTIFF mask with the
same CRS, transform, width, and height. Assert metadata contains the exact
prompt, `validated_semantic_scope = ["building", "road", "water"]`, threshold,
foreground share, checkpoint schema, and source hashes.

- [ ] **Step 2: Implement image loading without invented georeferencing**

```python
@dataclass(frozen=True)
class PromptImage:
    tensor: torch.Tensor
    rgb: np.ndarray
    crs: object | None
    transform: object | None
    source_path: Path


def load_prompt_image(path: str | Path) -> PromptImage:
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(path) as src:
            if src.count < 3:
                raise ValueError("GeoTIFF must contain at least three RGB bands")
            rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)
            crs, transform = src.crs, src.transform
    elif path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        crs, transform = None, None
    else:
        raise ValueError(f"unsupported image extension: {path.suffix}")
    if not np.isfinite(rgb).all():
        raise ValueError("input image contains non-finite values")
    scaled = rgb.astype(np.float32)
    if scaled.min() < 0 or scaled.max() > 255:
        raise ValueError("MVP RGB values must be in 0..255")
    scaled /= 255.0
    tensor = torch.from_numpy(np.moveaxis(scaled, -1, 0)).contiguous()
    return PromptImage(tensor, rgb, crs, transform, path)
```

- [ ] **Step 3: Implement checkpoint validation, model reconstruction, and inference**

`load_prompt_checkpoint` calls the same metadata validator and model builder as
the runner. It loads SigLIP with `local_files_only` when requested, rejects the
no-text baseline checkpoint, and never parses prompt keywords.

```python
@torch.no_grad()
def run_prompt_inference(model, image, prompt, *, threshold=0.5, device="cpu"):
    if not prompt.strip():
        raise ValueError("prompt must be non-empty English text")
    logits = model(image.tensor.unsqueeze(0).to(device), [prompt])
    probability = logits.sigmoid()[0].cpu().numpy().astype(np.float32)
    mask = (probability >= threshold).astype(np.uint8)
    return probability, mask
```

Write GeoTIFF only when `crs` and `transform` are present. Otherwise write PNG
and NumPy outputs. Create a deterministic preview with Pillow; do not add
matplotlib to core dependencies.

- [ ] **Step 4: Add and test the thin CLI**

```python
# scripts/run_geovlm_prompt_segmentation.py
from geoadapter.inference.prompt_segmentation import main

if __name__ == "__main__":
    main()
```

CLI:

```bash
python scripts/run_geovlm_prompt_segmentation.py \
  --image sample.tif \
  --prompt "segment all water bodies" \
  --checkpoint /path/to/seed42.pt \
  --output-dir results/geovlm_prompt_inference \
  --threshold 0.5 \
  --local-files-only
```

Test non-empty prompt validation, threshold bounds `[0,1]`, unsupported files,
checkpoint mismatch, and output paths.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_geovlm_prompt_inference.py tests/test_geovlm_prompt_runner.py -v`

Expected: PASS.

```bash
git add geoadapter/inference scripts/run_geovlm_prompt_segmentation.py tests/test_geovlm_prompt_inference.py
git commit -m "feat: add offline GeoVLM prompt inference"
```

## Task 8: Colab Notebook Generator and Contract

**Files:**
- Modify: `scripts/make_paper12_colab_notebooks.py`
- Create: `colab/paper12_geovlm_prompt_segmentation_colab.ipynb`
- Modify: `tests/test_paper12_colab_notebooks.py`

- [ ] **Step 1: Write the failing notebook/config contract test**

```python
def test_paper12_geovlm_prompt_segmentation_colab_contract():
    path = COLAB_DIR / "paper12_geovlm_prompt_segmentation_colab.ipynb"
    text = read_notebook_text(path)
    assert "blob/master/colab/paper12_geovlm_prompt_segmentation_colab.ipynb" in text
    assert "pip install -q -e '.[geovlm]' torchgeo" in text
    assert "google/siglip-base-patch16-224" in text
    assert "Prithvi_100M.pt" in text
    assert "geovlm_prompt_segmentation.yaml" in text
    assert "--stage seed42" in text
    assert "--stage full" in text
    assert "geovlm_prompt_segmentation.json" in text
    assert "geovlm_prompt_segmentation_summary.json" in text
    assert "mvp_status" in text
    assert "expected method/seed pairs = 6" in text
    assert "/content/drive/MyDrive/paper12_checkpoints/geovlm_prompt_segmentation" in text
```

- [ ] **Step 2: Add the notebook generator function**

Add `GEOVLM_PROMPT_OUT` and `geovlm_prompt_notebook()` using existing notebook
helpers. Required cells, in order:

1. Markdown scope warning: this is a three-concept English prompt MVP, not a
   complete or open-vocabulary ArcGIS GeoVLM.
2. Mount Drive; create result, checkpoint, preview, Hugging Face cache paths.
3. Print GPU, Python, disk, and CUDA details.
4. Clone/pull `master` into `/content/AlphaEarth-System`; record commit hash.
5. Install `-e '.[geovlm]' torchgeo` and print exact package versions.
6. Stage `Prithvi_100M.pt` from Drive or Hugging Face; calculate SHA-256.
7. Pre-cache `google/siglip-base-patch16-224`; record resolved model revision.
8. Download/verify LandCoverAI and assert mask values are within `{0,1,2,3,4}`.
9. Copy the checked-in YAML to a Colab-local config with absolute data/checkpoint
   paths and `allow_synthetic_fallback: false`.
10. Run focused offline tests.
11. Run `--stage seed42`; reload checkpoint; print smoke checks and failed gates.
12. Require an explicit `RUN_FULL_MATRIX = False` toggle. When true, run
   `--stage full` for six method/seed pairs.
13. Validate raw pair count, run summary CLI with 1,000 bootstrap iterations,
   print every gate and `mvp_status`, list previews/checkpoints, and copy outputs
   to Drive.

No notebook cell may change the configured model to another text tower or use
synthetic fallback.

- [ ] **Step 3: Generate and verify the notebook**

Run: `python scripts/make_paper12_colab_notebooks.py`

Expected: `[ok] wrote ...paper12_geovlm_prompt_segmentation_colab.ipynb` and
existing notebooks remain unchanged unless their deterministic rendering
already differs from checked-in content.

Run: `python -m pytest tests/test_paper12_colab_notebooks.py -v`

Expected: PASS.

- [ ] **Step 4: Commit generator and generated artifact**

```bash
git add scripts/make_paper12_colab_notebooks.py colab/paper12_geovlm_prompt_segmentation_colab.ipynb tests/test_paper12_colab_notebooks.py
git commit -m "feat: add Paper12 GeoVLM Colab workflow"
```

## Task 9: Documentation, Static Audit, and Local Verification

**Files:**
- Modify: `paper12/README.md`
- Create: `docs/geovlm_prompt_segmentation_mvp.md`
- Test: all focused tests and full suite

- [ ] **Step 1: Add bounded status and exact commands**

Document:

- validated scope: English prompts for building, road, water on LandCoverAI;
- architecture: Prithvi + checkpoint positions + Houlsby + frozen SigLIP +
  FiLM/similarity decoder;
- official five-class source mapping;
- local test command;
- Colab notebook path and Drive output locations;
- seed-42 and full runner commands;
- summary/gate command;
- offline inference command;
- explicit statement that no real result or MVP completion claim exists until
  committed three-seed JSON passes every gate.

Do not edit `paper12/sections/*.tex`, submission PDFs, or result claims in this
task.

- [ ] **Step 2: Run the complete focused test set**

Run:

```bash
python -m pytest \
  tests/test_prompt_segmentation_data.py \
  tests/test_prithvi_position_embeddings.py \
  tests/test_prompt_segmentation_model.py \
  tests/test_prompt_segmentation_engine.py \
  tests/test_geovlm_prompt_summary.py \
  tests/test_geovlm_prompt_runner.py \
  tests/test_geovlm_prompt_inference.py \
  tests/test_paper12_colab_notebooks.py -v
```

Expected: all tests PASS; no network access or model download occurs.

- [ ] **Step 3: Run legacy regression tests around touched modules**

Run:

```bash
python -m pytest \
  tests/test_prithvi.py \
  tests/test_backbone_factory.py \
  tests/test_heads.py \
  tests/test_datasets.py \
  tests/test_trainer.py \
  tests/test_benchmark_runner.py -v
```

Expected: all tests PASS, proving historical paths retain default behavior.

- [ ] **Step 4: Run the full maintained suite**

Run: `python -m pytest tests -v`

Expected: PASS with only already-known skips/warnings. If the count differs from
the historical 244 passed/6 skipped, report the new exact count and investigate
all failures; do not assume the historical count remains current.

- [ ] **Step 5: Run source and artifact checks**

Run:

```bash
python -m compileall -q geoadapter scripts/run_geovlm_prompt_segmentation.py
python scripts/make_paper12_colab_notebooks.py
git diff --check
git status --short
```

Expected: compile exit 0, deterministic notebook generation, clean whitespace,
and only intended implementation/documentation changes before the final commit.

- [ ] **Step 6: Commit documentation and local verification checkpoint**

```bash
git add paper12/README.md docs/geovlm_prompt_segmentation_mvp.md
git commit -m "docs: document GeoVLM prompt MVP workflow"
```

## Task 10: Real Colab Evidence Intake and Status Decision

**Files:**
- Add only after real execution: `paper12_results/geovlm_prompt_segmentation.json`
- Add only after real execution: `paper12_results/geovlm_prompt_segmentation_summary.json`
- Optionally modify after author approval: `paper12/README.md`

- [ ] **Step 1: Run the generated notebook through seed 42**

Required evidence before continuing:

- real LandCoverAI cache verified;
- real Prithvi checkpoint hash recorded;
- real SigLIP model id/revision recorded;
- loss decreases and remains finite;
- each target class produces a nonconstant mask;
- prompt probability-map change is nonzero;
- checkpoint reload reproduces logits.

If Stage 1 fails, preserve diagnostics in Drive and stop. Do not run the full
matrix or alter the acceptance thresholds.

- [ ] **Step 2: Run the full six method/seed pairs**

Set `RUN_FULL_MATRIX = True`, resume from seed 42, and let the append-safe runner
complete all prompt/baseline pairs for seeds 42, 123, and 456.

- [ ] **Step 3: Verify the result files before copying into Git**

Run locally after downloading from Drive:

```bash
python -m geoadapter.bench.geovlm_prompt_summary \
  --input paper12_results/geovlm_prompt_segmentation.json \
  --output paper12_results/geovlm_prompt_segmentation_summary.json \
  --bootstrap-iterations 1000
python -m pytest tests/test_geovlm_prompt_summary.py -v
git diff --check
```

Check that raw JSON has all six `(method, seed)` groups, prompt rows contain all
three classes, no row indicates synthetic fallback, and the summary contains
every gate.

- [ ] **Step 4: Apply the status decision without overclaiming**

If `mvp_status == "passed"`, update only status documentation to say that the
bounded three-concept LandCoverAI MVP passed. Do not claim open-vocabulary,
cross-dataset, Chinese-language, captioning, VQA, or ArcGIS parity.

If `mvp_status == "failed"`, keep the raw and summary evidence only if useful,
state the exact failed gates, and follow the approved tuning order:

1. BCE/Dice coefficients;
2. empty-target cap;
3. conditioning or decoder dimension.

Each tuning run gets a distinct config/result identifier; never overwrite a
failed run.

- [ ] **Step 5: Commit evidence only after author review**

```bash
git add paper12_results/geovlm_prompt_segmentation.json paper12_results/geovlm_prompt_segmentation_summary.json paper12/README.md
git commit -m "results: record Paper12 GeoVLM prompt segmentation evidence"
```

Do not push, edit the manuscript, or integrate backend/frontend behavior without
a separate explicit request.

## Plan Self-Review

- Spec coverage: architecture, data/prompt contract, position embeddings,
  frozen SigLIP, FiLM/similarity decoder, no-text baseline, BCE+Dice, empty
  sampling, three seeds, held-out prompts, counterfactual bootstrap, checkpoint
  metadata, inference artifacts, Colab, and bounded claims are each assigned to
  a task.
- Scope: implementation remains one offline model evidence chain. Product UI,
  API, ArcGIS packaging, manuscript integration, and additional languages/tasks
  remain separate projects.
- Type consistency: prompt models return `[B,H,W]` logits; targets use
  `[B,H,W]` float tensors; official class ids are `{1,3,4}`; runner, checkpoint,
  summary, and inference all use the same schema names and target mapping.
- Preservation: historical configs/results and the default Prithvi forward path
  remain unchanged.
- Verification: local tests are offline; only the Colab phase may download
  TorchGeo data or Hugging Face weights; no completion claim occurs before real
  three-seed gates pass.
