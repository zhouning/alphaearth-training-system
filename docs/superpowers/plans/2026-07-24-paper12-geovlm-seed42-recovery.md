# Paper12 GeoVLM Seed-42 Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the real LandCoverAI GeoVLM seed-42 smoke run by making the SigLIP cache explicit, enforcing target-present prompt training, selecting a reproducible best checkpoint with a fixed probe, and rejecting artifacts from the failed training contract.

**Architecture:** Add a small pure training-data module that scans supported targets, reserves deterministic probes, and creates balanced per-epoch prompt assignments with a hard per-batch empty-target cap. Keep model loading, probe evaluation, atomic best/last checkpointing, result validation, and experiment gating in the existing runner so the public CLI remains unchanged. Version the complete behavior in config, checkpoints, and raw rows, and make the generated Colab require a manual failed-run archive before the recovery starts.

**Tech Stack:** Python 3.11+, PyTorch, Transformers, PyYAML, pytest, nbformat-compatible JSON notebooks, Google Colab, Google Drive.

---

## File Map

- Create `geoadapter/bench/geovlm_training.py`: pure target-present indexing, deterministic probe selection, balanced prompt assignments, and the assigned dataset view.
- Create `tests/test_geovlm_training.py`: focused unit tests for pool scanning, probe selection, class balance, determinism, and total empty-target caps.
- Modify `geoadapter/data/prompt_segmentation.py`: materialize a prompt batch from runner-supplied class names without changing the legacy random sampler.
- Modify `geoadapter/models/text_encoder.py`: accept and forward an optional Hugging Face `cache_dir`.
- Modify `geoadapter/bench/run_geovlm_prompt_segmentation.py`: training-contract validation, explicit text-cache construction, probe diagnostics, best/last checkpoints, result diagnostics, and best-checkpoint evaluation/reproduction.
- Modify `geoadapter/bench/configs/geovlm_prompt_segmentation.yaml`: checked-in recovery contract, probe count, and nullable local cache path.
- Modify `tests/test_prompt_segmentation_model.py`: verify both Transformers loaders receive the same cache arguments.
- Modify `tests/test_prompt_segmentation_data.py`: verify scheduled class names create exactly the requested binary targets.
- Modify `tests/test_geovlm_prompt_runner.py`: verify the recovery contract, degradation-resistant selection, resume behavior, artifact rejection, and recorded diagnostics.
- Modify `tests/test_paper12_colab_notebooks.py`: verify explicit Drive cache use, manual failed-run archiving, and `.best.pt` inspection.
- Modify `scripts/make_paper12_colab_notebooks.py`: generate the recovery-safe Colab workflow.
- Regenerate `colab/paper12_geovlm_prompt_segmentation_colab.ipynb`: checked-in notebook derived from the generator.
- Modify `docs/geovlm_prompt_segmentation_mvp.md`: document the failed run, recovery contract, archive/run sequence, diagnostics, and decision boundary.

### Task 1: Version The Recovery Contract And Resolve SigLIP From Its Explicit Cache

**Files:**
- Modify: `tests/test_prompt_segmentation_model.py`
- Modify: `tests/test_geovlm_prompt_runner.py`
- Modify: `geoadapter/models/text_encoder.py`
- Modify: `geoadapter/bench/run_geovlm_prompt_segmentation.py`
- Modify: `geoadapter/bench/configs/geovlm_prompt_segmentation.yaml`

- [ ] **Step 1: Write failing tests for cache forwarding and the checked-in contract**

Replace the fake Transformers loaders in `test_siglip_text_encoder_freezes_tower_and_keeps_it_in_eval` with call-recording loaders, construct the encoder with a cache directory and revision, and add the exact assertions below:

```python
    calls = {}

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["tokenizer"] = (model_id, kwargs)
            return cls()

        def __call__(self, prompts, **kwargs):
            return {"input_ids": torch.ones(len(prompts), 3, dtype=torch.long)}

    class FakeSiglipTextModel(nn.Module):
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["model"] = (model_id, kwargs)
            return cls()

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4))
            self.config = types.SimpleNamespace(hidden_size=4)

        def forward(self, input_ids):
            pooled = self.weight.unsqueeze(0).expand(input_ids.shape[0], -1)
            return types.SimpleNamespace(pooler_output=pooled)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=FakeTokenizer,
            SiglipTextModel=FakeSiglipTextModel,
        ),
    )
    text_encoder_module = importlib.import_module("geoadapter.models.text_encoder")
    encoder = text_encoder_module.SiglipTextEncoder(
        "google/siglip-base-patch16-224",
        revision="resolved-revision",
        cache_dir="/content/drive/MyDrive/huggingface_cache/paper12_geovlm",
        local_files_only=True,
    )

    expected_loader_kwargs = {
        "revision": "resolved-revision",
        "cache_dir": "/content/drive/MyDrive/huggingface_cache/paper12_geovlm",
        "local_files_only": True,
    }
    assert calls == {
        "tokenizer": ("google/siglip-base-patch16-224", expected_loader_kwargs),
        "model": ("google/siglip-base-patch16-224", expected_loader_kwargs),
    }
```

Extend `test_geovlm_prompt_config_is_real_data_only` with:

```python
    assert config["experiment"]["training_contract"] == (
        "paper12.geovlm_prompt_training.v2"
    )
    assert config["experiment"]["probe_positives_per_class"] == 2
    assert config["text_encoder"]["cache_dir"] is None
```

Import `build_text_encoder` in `tests/test_geovlm_prompt_runner.py` and add a
runner-level regression so adding the encoder argument without wiring the
config cannot pass:

```python
def test_runner_build_text_encoder_forwards_explicit_cache(monkeypatch):
    import geoadapter.models.text_encoder as text_encoder_module

    captured = {}

    class FakeTextEncoder:
        def __init__(self, model_id, **kwargs):
            captured["model_id"] = model_id
            captured.update(kwargs)

    monkeypatch.setattr(
        text_encoder_module, "SiglipTextEncoder", FakeTextEncoder
    )
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["text_encoder"].update(
        {
            "revision": "resolved-revision",
            "cache_dir": "/content/drive/MyDrive/huggingface_cache/paper12_geovlm",
            "local_files_only": True,
        }
    )

    build_text_encoder(config)

    assert captured == {
        "model_id": "google/siglip-base-patch16-224",
        "revision": "resolved-revision",
        "cache_dir": "/content/drive/MyDrive/huggingface_cache/paper12_geovlm",
        "local_files_only": True,
    }
```

Replace its existing v1 schema assertion, then extend
`test_checkpoint_metadata_hashes_external_contracts` with:

```python
    assert metadata["schema"] == "paper12.geovlm_prompt_checkpoint.v2"
    assert metadata["training_contract"] == "paper12.geovlm_prompt_training.v2"
    assert metadata["target_pool_policy"] == "supported_target_present_only"
    assert metadata["empty_target_cap"] == 0.25
    assert metadata["probe_positives_per_class"] == 2
    assert metadata["best_checkpoint_policy"] == (
        "finite_nonconstant_prompt_change_loss_v1"
    )
    assert "siglip_cache_dir" not in metadata
```

- [ ] **Step 2: Run the cache and contract tests to verify RED**

Run:

```bash
python -m pytest tests/test_prompt_segmentation_model.py::test_siglip_text_encoder_freezes_tower_and_keeps_it_in_eval tests/test_geovlm_prompt_runner.py::test_runner_build_text_encoder_forwards_explicit_cache tests/test_geovlm_prompt_runner.py::test_geovlm_prompt_config_is_real_data_only tests/test_geovlm_prompt_runner.py::test_checkpoint_metadata_hashes_external_contracts -v
```

Expected: FAIL because `SiglipTextEncoder.__init__` rejects `cache_dir`, the config lacks the recovery keys, and checkpoint metadata still uses schema v1.

- [ ] **Step 3: Add the minimal cache and contract implementation**

Add `cache_dir` to `SiglipTextEncoder` and pass it unchanged to both loaders:

```python
class SiglipTextEncoder(nn.Module):
    def __init__(
        self,
        model_id: str = "google/siglip-base-patch16-224",
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
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
        loader_kwargs = {
            "revision": revision,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **loader_kwargs)
        self.model = SiglipTextModel.from_pretrained(model_id, **loader_kwargs)
        self.output_dim = int(self.model.config.hidden_size)
        self.model.requires_grad_(False)
        self.model.eval()
```

Import `Path` in `geoadapter/models/text_encoder.py`. Add this helper to the runner and call it from the prompt branch of `build_model`:

```python
def build_text_encoder(config: dict[str, Any]):
    from geoadapter.models.text_encoder import SiglipTextEncoder

    return SiglipTextEncoder(
        config["text_encoder"]["model_id"],
        revision=config["text_encoder"].get("revision"),
        cache_dir=config["text_encoder"].get("cache_dir"),
        local_files_only=bool(config["text_encoder"].get("local_files_only", False)),
    )
```

Replace the inline `SiglipTextEncoder` construction in `build_model` with `text_encoder = build_text_encoder(config)`.

Add these config fields without changing the optimizer, loss, model, threshold, or gate values:

```yaml
experiment:
  training_contract: paper12.geovlm_prompt_training.v2
  probe_positives_per_class: 2
text_encoder:
  cache_dir: null
```

Change checkpoint metadata schema to `paper12.geovlm_prompt_checkpoint.v2` and add:

```python
        "training_contract": config["experiment"]["training_contract"],
        "target_pool_policy": "supported_target_present_only",
        "empty_target_cap": float(config["experiment"]["empty_target_cap"]),
        "probe_positives_per_class": int(
            config["experiment"]["probe_positives_per_class"]
        ),
        "best_checkpoint_policy": "finite_nonconstant_prompt_change_loss_v1",
```

Add the same six fields (`schema` plus the five recovery fields) to `validate_checkpoint_metadata`. Keep `cache_dir` out of metadata so Drive paths do not change the scientific identity.

- [ ] **Step 4: Run the focused tests to verify GREEN**

Run the Step 2 command again.

Expected: `4 passed`.

- [ ] **Step 5: Commit the cache and contract change**

```bash
git add geoadapter/models/text_encoder.py geoadapter/bench/run_geovlm_prompt_segmentation.py geoadapter/bench/configs/geovlm_prompt_segmentation.yaml tests/test_prompt_segmentation_model.py tests/test_geovlm_prompt_runner.py
git commit -m "fix: version GeoVLM recovery cache contract"
```

### Task 2: Build A Deterministic Target-Present Pool And Training Probe

**Files:**
- Create: `geoadapter/bench/geovlm_training.py`
- Create: `tests/test_geovlm_training.py`

- [ ] **Step 1: Write failing pool and probe tests**

Create `tests/test_geovlm_training.py` with the dataset fixture and these tests:

```python
import torch
from torch.utils.data import Dataset

from geoadapter.bench.geovlm_training import (
    reserve_training_probe,
    scan_target_present_pool,
)


class MaskDataset(Dataset):
    def __init__(self, masks):
        self.masks = [torch.tensor(mask, dtype=torch.long) for mask in masks]

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):
        return torch.zeros(3, 2, 2), self.masks[index]


def test_target_present_pool_excludes_no_target_tiles_and_records_classes():
    dataset = MaskDataset(
        [
            [[0, 0], [2, 2]],
            [[1, 0], [0, 0]],
            [[3, 4], [0, 0]],
            [[4, 4], [2, 0]],
        ]
    )

    pool = scan_target_present_pool(dataset)

    assert pool.source_size == 4
    assert pool.excluded_no_target_count == 1
    assert pool.excluded_no_target_share == 0.25
    assert [sample.source_index for sample in pool.samples] == [1, 2, 3]
    assert pool.samples[1].present_classes == ("water", "road")


def test_target_present_pool_fails_for_empty_pool_and_missing_class():
    empty = MaskDataset([[[0, 2], [2, 0]]])
    with pytest.raises(ValueError, match="target-present training pool is empty"):
        scan_target_present_pool(empty)

    missing_water = MaskDataset(
        [
            [[1, 0], [0, 0]],
            [[4, 0], [0, 0]],
        ]
    )
    with pytest.raises(ValueError, match="no samples for: water"):
        scan_target_present_pool(missing_water)


def test_probe_is_deterministic_bounded_and_removed_once_from_training():
    dataset = MaskDataset(
        [
            [[1, 3], [0, 0]],
            [[1, 0], [0, 0]],
            [[1, 4], [0, 0]],
            [[3, 0], [0, 0]],
            [[3, 4], [0, 0]],
            [[4, 0], [0, 0]],
            [[1, 0], [0, 0]],
            [[3, 0], [0, 0]],
            [[4, 0], [0, 0]],
        ]
    )
    pool = scan_target_present_pool(dataset)

    first = reserve_training_probe(pool, seed=42, positives_per_class=2)
    second = reserve_training_probe(pool, seed=42, positives_per_class=2)

    assert first == second
    assert all(len(indices) == 2 for indices in first.probe_indices_by_class.values())
    assert len(first.probe_indices) <= 6
    assert len(first.probe_sha256) == 64
    assert not set(first.probe_indices) & {
        sample.source_index for sample in first.training_samples
    }
    assert len(first.training_samples) + len(first.probe_indices) == len(pool.samples)
```

Add `import pytest` at the top of the file.

- [ ] **Step 2: Run the new tests to verify RED**

Run:

```bash
python -m pytest tests/test_geovlm_training.py -v
```

Expected: collection ERROR with `ModuleNotFoundError: geoadapter.bench.geovlm_training`.

- [ ] **Step 3: Implement the pool and probe API**

Create `geoadapter/bench/geovlm_training.py` with these public records and functions:

```python
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import torch

from geoadapter.data.prompt_segmentation import (
    PROMPT_TARGET_CLASS_IDS,
    validate_landcoverai_mask,
)


CLASS_NAMES = tuple(PROMPT_TARGET_CLASS_IDS)


@dataclass(frozen=True)
class TargetPresentSample:
    source_index: int
    present_classes: tuple[str, ...]


@dataclass(frozen=True)
class TargetPresentPool:
    source_size: int
    samples: tuple[TargetPresentSample, ...]
    excluded_no_target_count: int

    @property
    def excluded_no_target_share(self) -> float:
        return self.excluded_no_target_count / self.source_size


@dataclass(frozen=True)
class TrainingProbeSplit:
    training_samples: tuple[TargetPresentSample, ...]
    probe_indices: tuple[int, ...]
    probe_indices_by_class: dict[str, tuple[int, ...]]
    probe_sha256: str


def _missing_classes(samples: tuple[TargetPresentSample, ...]) -> list[str]:
    return [
        name
        for name in CLASS_NAMES
        if not any(name in sample.present_classes for sample in samples)
    ]


def scan_target_present_pool(dataset) -> TargetPresentPool:
    samples = []
    for source_index in range(len(dataset)):
        _, mask = dataset[source_index]
        validate_landcoverai_mask(mask)
        present = tuple(
            name
            for name, class_id in PROMPT_TARGET_CLASS_IDS.items()
            if bool(mask.eq(class_id).any())
        )
        if present:
            samples.append(TargetPresentSample(source_index, present))
    if not samples:
        raise ValueError("target-present training pool is empty")
    sample_tuple = tuple(samples)
    missing = _missing_classes(sample_tuple)
    if missing:
        raise ValueError(
            "target-present training pool has no samples for: " + ", ".join(missing)
        )
    return TargetPresentPool(
        source_size=len(dataset),
        samples=sample_tuple,
        excluded_no_target_count=len(dataset) - len(sample_tuple),
    )


def reserve_training_probe(
    pool: TargetPresentPool,
    *,
    seed: int,
    positives_per_class: int = 2,
) -> TrainingProbeSplit:
    if positives_per_class <= 0:
        raise ValueError("positives_per_class must be positive")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(pool.samples), generator=generator).tolist()
    shuffled = tuple(pool.samples[index] for index in order)
    by_class = {
        name: tuple(
            sample.source_index
            for sample in shuffled
            if name in sample.present_classes
        )[:positives_per_class]
        for name in CLASS_NAMES
    }
    short = [name for name, indices in by_class.items() if len(indices) < positives_per_class]
    if short:
        raise ValueError(
            "insufficient probe positives for: " + ", ".join(short)
        )
    probe_set = {index for indices in by_class.values() for index in indices}
    probe_indices = tuple(
        sample.source_index for sample in shuffled if sample.source_index in probe_set
    )
    training_samples = tuple(
        sample for sample in pool.samples if sample.source_index not in probe_set
    )
    if not training_samples:
        raise ValueError("probe reservation leaves an empty training pool")
    missing = _missing_classes(training_samples)
    if missing:
        raise ValueError(
            "probe reservation leaves no training samples for: " + ", ".join(missing)
        )
    stable_payload = json.dumps(
        {name: list(by_class[name]) for name in CLASS_NAMES},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return TrainingProbeSplit(
        training_samples=training_samples,
        probe_indices=probe_indices,
        probe_indices_by_class=by_class,
        probe_sha256=hashlib.sha256(stable_payload).hexdigest(),
    )
```

- [ ] **Step 4: Run the pool and probe tests to verify GREEN**

Run the Step 2 command again.

Expected: `3 passed`.

- [ ] **Step 5: Commit the target-present pool and probe selection**

```bash
git add geoadapter/bench/geovlm_training.py tests/test_geovlm_training.py
git commit -m "feat: add GeoVLM target-present training split"
```

### Task 3: Enforce Balanced Prompt Assignments And A Total Empty-Target Cap

**Files:**
- Modify: `geoadapter/bench/geovlm_training.py`
- Modify: `geoadapter/data/prompt_segmentation.py`
- Modify: `tests/test_geovlm_training.py`
- Modify: `tests/test_prompt_segmentation_data.py`

- [ ] **Step 1: Write failing assignment and prompt-materialization tests**

Append this import and test to `tests/test_prompt_segmentation_data.py`:

```python
from geoadapter.data.prompt_segmentation import prompt_batch_from_class_names


def test_prompt_batch_from_class_names_preserves_the_schedule():
    masks = torch.stack(
        [
            torch.tensor([[1, 0], [0, 0]]),
            torch.tensor([[3, 3], [0, 0]]),
            torch.tensor([[4, 0], [4, 0]]),
        ]
    )
    config = load_prompt_config(CONFIG)

    batch = prompt_batch_from_class_names(
        masks,
        ("water", "water", "road"),
        config,
        generator=torch.Generator().manual_seed(7),
    )

    assert batch.class_names == ("water", "water", "road")
    assert batch.class_ids.tolist() == [3, 3, 4]
    assert batch.empty_count == 1
    assert batch.targets.flatten(1).sum(dim=1).tolist() == [0.0, 2.0, 2.0]
```

Append the new imports and assignment test to `tests/test_geovlm_training.py`:

```python
from geoadapter.bench.geovlm_training import (
    AssignedPromptDataset,
    build_epoch_assignments,
)


def test_epoch_assignments_are_deterministic_balanced_and_cap_total_empties():
    masks = []
    for class_id in (1, 3, 4):
        masks.extend([[[class_id, 0], [0, 0]]] * 8)
    dataset = MaskDataset(masks)
    pool = scan_target_present_pool(dataset)
    split = reserve_training_probe(pool, seed=42, positives_per_class=2)

    first = build_epoch_assignments(
        split,
        batch_size=4,
        empty_target_cap=0.25,
        seed=42,
    )
    second = build_epoch_assignments(
        split,
        batch_size=4,
        empty_target_cap=0.25,
        seed=42,
    )

    assert first == second
    present_by_index = {
        sample.source_index: sample.present_classes for sample in pool.samples
    }
    for start in range(0, len(first), 4):
        batch = first[start : start + 4]
        assert sum(item.empty_target for item in batch) <= 1
        assert all(
            item.empty_target
            == (item.class_name not in present_by_index[item.source_index])
            for item in batch
        )
    positive_counts = {
        name: sum(
            item.class_name == name and not item.empty_target for item in first
        )
        for name in ("building", "water", "road")
    }
    assert max(positive_counts.values()) - min(positive_counts.values()) <= 1
    assigned = AssignedPromptDataset(dataset, first)
    _, _, class_name = assigned[0]
    assert class_name == first[0].class_name
```

- [ ] **Step 2: Run assignment tests to verify RED**

Run:

```bash
python -m pytest tests/test_geovlm_training.py tests/test_prompt_segmentation_data.py -v
```

Expected: collection ERROR because `AssignedPromptDataset`, `build_epoch_assignments`, and `prompt_batch_from_class_names` do not exist.

- [ ] **Step 3: Implement exact scheduled prompt materialization**

Add this function beside `sample_prompt_batch` in `geoadapter/data/prompt_segmentation.py`:

```python
def prompt_batch_from_class_names(
    masks: torch.Tensor,
    class_names: tuple[str, ...] | list[str],
    config: PromptConfig,
    *,
    generator: torch.Generator | None = None,
) -> PromptBatch:
    if masks.ndim != 3:
        raise ValueError("masks must have shape [B,H,W]")
    names = tuple(class_names)
    if len(names) != int(masks.shape[0]):
        raise ValueError("class_names must have one entry per mask")
    unsupported = sorted(set(names) - set(PROMPT_TARGET_CLASS_IDS))
    if unsupported:
        raise ValueError("unsupported scheduled classes: " + ", ".join(unsupported))
    prompts = []
    targets = []
    for mask, name in zip(masks, names):
        validate_landcoverai_mask(mask)
        values = config.classes[name].training
        prompt_index = int(torch.randint(len(values), (), generator=generator))
        prompts.append(values[prompt_index])
        targets.append(multiclass_to_binary(mask, PROMPT_TARGET_CLASS_IDS[name]))
    target_tensor = torch.stack(targets)
    empty_count = int(target_tensor.flatten(1).sum(dim=1).eq(0).sum())
    return PromptBatch(
        class_ids=torch.tensor(
            [PROMPT_TARGET_CLASS_IDS[name] for name in names], dtype=torch.long
        ),
        class_names=names,
        prompts=tuple(prompts),
        targets=target_tensor,
        empty_count=empty_count,
        voluntary_empty_count=empty_count,
    )
```

- [ ] **Step 4: Implement deterministic capped epoch assignments**

Add the following definitions to `geoadapter/bench/geovlm_training.py`:

```python
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PromptAssignment:
    source_index: int
    class_name: str
    empty_target: bool


class AssignedPromptDataset(Dataset):
    def __init__(self, dataset, assignments: tuple[PromptAssignment, ...]):
        self.dataset = dataset
        self.assignments = assignments

    def __len__(self):
        return len(self.assignments)

    def __getitem__(self, index):
        assignment = self.assignments[index]
        image, mask = self.dataset[assignment.source_index]
        return image, mask, assignment.class_name


def _balanced_names(
    count: int,
    names: tuple[str, ...],
    generator: torch.Generator,
) -> list[str]:
    if count == 0:
        return []
    order = torch.randperm(len(names), generator=generator).tolist()
    cycle = tuple(names[index] for index in order)
    return [cycle[index % len(cycle)] for index in range(count)]


def _draw_source(
    candidates: tuple[int, ...], generator: torch.Generator
) -> int:
    return candidates[int(torch.randint(len(candidates), (), generator=generator))]


def build_epoch_assignments(
    split: TrainingProbeSplit,
    *,
    batch_size: int,
    empty_target_cap: float,
    seed: int,
) -> tuple[PromptAssignment, ...]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= empty_target_cap <= 1.0:
        raise ValueError("empty_target_cap must be between 0 and 1")
    present_by_index = {
        sample.source_index: sample.present_classes for sample in split.training_samples
    }
    positive_candidates = {
        name: tuple(
            index for index, present in present_by_index.items() if name in present
        )
        for name in CLASS_NAMES
    }
    missing = [name for name, indices in positive_candidates.items() if not indices]
    if missing:
        raise ValueError("training split has no positive samples for: " + ", ".join(missing))
    negative_candidates = {
        name: tuple(
            index for index, present in present_by_index.items() if name not in present
        )
        for name in CLASS_NAMES
    }
    negative_names = tuple(
        name for name in CLASS_NAMES if negative_candidates[name]
    )
    epoch_size = len(split.training_samples)
    batch_sizes = tuple(
        min(batch_size, epoch_size - start)
        for start in range(0, epoch_size, batch_size)
    )
    empty_counts = tuple(
        int(size * empty_target_cap) if negative_names else 0 for size in batch_sizes
    )
    positive_total = epoch_size - sum(empty_counts)
    generator = torch.Generator().manual_seed(seed)
    positive_names = _balanced_names(positive_total, CLASS_NAMES, generator)
    empty_names = _balanced_names(sum(empty_counts), negative_names, generator)
    positive_offset = 0
    empty_offset = 0
    assignments = []
    for size, empty_count in zip(batch_sizes, empty_counts):
        batch = []
        for name in positive_names[positive_offset : positive_offset + size - empty_count]:
            batch.append(
                PromptAssignment(
                    _draw_source(positive_candidates[name], generator), name, False
                )
            )
        positive_offset += size - empty_count
        for name in empty_names[empty_offset : empty_offset + empty_count]:
            batch.append(
                PromptAssignment(
                    _draw_source(negative_candidates[name], generator), name, True
                )
            )
        empty_offset += empty_count
        order = torch.randperm(len(batch), generator=generator).tolist()
        assignments.extend(batch[index] for index in order)
    return tuple(assignments)
```

- [ ] **Step 5: Run assignment tests to verify GREEN**

Run the Step 2 command again.

Expected: all tests in both files PASS.

- [ ] **Step 6: Commit deterministic prompt assignment**

```bash
git add geoadapter/bench/geovlm_training.py geoadapter/data/prompt_segmentation.py tests/test_geovlm_training.py tests/test_prompt_segmentation_data.py
git commit -m "fix: cap GeoVLM total empty targets"
```

### Task 4: Record A Fixed Per-Epoch Probe And Deterministic Best Rank

**Files:**
- Modify: `geoadapter/bench/run_geovlm_prompt_segmentation.py`
- Modify: `tests/test_geovlm_prompt_runner.py`

- [ ] **Step 1: Write failing tests for probe diagnostics and ranking**

Import `_evaluate_probe` and `probe_rank` in `tests/test_geovlm_prompt_runner.py`. Add:

```python
def test_probe_records_finite_class_variation_prompt_change_and_rank():
    dataset = _TinyPromptDataset()
    probe_indices_by_class = {
        "building": (0,),
        "water": (1,),
        "road": (2,),
    }
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    prompt_config = load_prompt_config(
        Path("geoadapter/bench/configs/geovlm_prompts.yaml")
    )
    trainer = build_trainer(_TinyConditionalModel(), config, "cpu")
    with torch.no_grad():
        trainer.model.condition_bias.copy_(torch.tensor([0.0, 1.0, 2.0]))

    probe = _evaluate_probe(
        trainer,
        dataset,
        probe_indices_by_class,
        prompt_config,
        {"building": 1.0, "water": 1.0, "road": 1.0},
        PROMPT_METHOD,
    )

    assert probe["finite"] is True
    assert probe["mean_loss"] > 0.0
    assert probe["nonconstant_class_count"] == 3
    assert probe["prompt_map_changed_class_count"] == 3
    assert probe["mean_prompt_probability_change"] > 0.0
    assert set(probe["classes"]) == {"building", "water", "road"}
    rank = probe_rank(probe)
    assert rank[:3] == (1, 3, 3)
    assert rank[3] > 0.0
    assert rank[4] == -probe["mean_loss"]


def test_probe_rank_prefers_map_change_then_lower_loss():
    base = {
        "finite": True,
        "nonconstant_class_count": 3,
        "prompt_map_changed_class_count": 3,
        "mean_prompt_probability_change": 0.2,
        "mean_loss": 1.0,
    }
    larger_change = {**base, "mean_prompt_probability_change": 0.3, "mean_loss": 2.0}
    lower_loss = {**base, "mean_loss": 0.5}

    assert probe_rank(larger_change) > probe_rank(base)
    assert probe_rank(lower_loss) > probe_rank(base)
```

Add `PROMPT_METHOD` to the existing runner imports.

- [ ] **Step 2: Run probe tests to verify RED**

Run:

```bash
python -m pytest tests/test_geovlm_prompt_runner.py::test_probe_records_finite_class_variation_prompt_change_and_rank tests/test_geovlm_prompt_runner.py::test_probe_rank_prefers_map_change_then_lower_loss -v
```

Expected: collection ERROR because `_evaluate_probe` and `probe_rank` are undefined.

- [ ] **Step 3: Implement bounded probe evaluation**

Add these helpers to the runner:

```python
def _probe_condition(method, class_name, prompt_config):
    if method == PROMPT_METHOD:
        return [prompt_config.classes[class_name].training[0]]
    return torch.tensor([PROMPT_TARGET_CLASS_IDS[class_name]], dtype=torch.long)


def _evaluate_probe(
    trainer,
    train,
    probe_indices_by_class,
    prompt_config,
    weights,
    method,
):
    class_names = tuple(PROMPT_TARGET_CLASS_IDS)
    classes = {}
    all_losses = []
    all_changes = []
    finite = True
    for class_index, class_name in enumerate(class_names):
        wrong_name = class_names[(class_index + 1) % len(class_names)]
        minimum = float("inf")
        maximum = -float("inf")
        changes = []
        losses = []
        for source_index in probe_indices_by_class[class_name]:
            image, mask = train[source_index]
            target = multiclass_to_binary(
                mask, PROMPT_TARGET_CLASS_IDS[class_name]
            ).unsqueeze(0)
            correct = trainer.predict(
                image.unsqueeze(0),
                _probe_condition(method, class_name, prompt_config),
            )
            wrong = trainer.predict(
                image.unsqueeze(0),
                _probe_condition(method, wrong_name, prompt_config),
            )
            positive_weight = torch.tensor(
                [weights[class_name]], dtype=torch.float32, device=trainer.device
            )
            loss = trainer.criterion(
                correct,
                target.to(trainer.device),
                positive_weight,
            )
            correct_probability = correct.sigmoid().detach().cpu()
            wrong_probability = wrong.sigmoid().detach().cpu()
            loss_value = float(loss.detach().cpu())
            change = float((correct_probability - wrong_probability).abs().mean())
            minimum = min(minimum, float(correct_probability.min()))
            maximum = max(maximum, float(correct_probability.max()))
            losses.append(loss_value)
            changes.append(change)
            finite = bool(
                finite
                and torch.isfinite(correct).all()
                and torch.isfinite(wrong).all()
                and np.isfinite(loss_value)
                and np.isfinite(change)
            )
        prediction_range = maximum - minimum
        mean_change = float(sum(changes) / len(changes))
        classes[class_name] = {
            "prediction_range": prediction_range,
            "prediction_nonconstant": bool(prediction_range > 0.0),
            "mean_prompt_probability_change": mean_change,
            "prompt_map_changed": bool(mean_change > 0.0),
        }
        all_losses.extend(losses)
        all_changes.extend(changes)
    return {
        "finite": finite,
        "mean_loss": float(sum(all_losses) / len(all_losses)),
        "nonconstant_class_count": sum(
            int(value["prediction_nonconstant"]) for value in classes.values()
        ),
        "prompt_map_changed_class_count": sum(
            int(value["prompt_map_changed"]) for value in classes.values()
        ),
        "mean_prompt_probability_change": float(
            sum(all_changes) / len(all_changes)
        ),
        "classes": classes,
    }


def probe_rank(probe: dict[str, Any]) -> tuple[int, int, int, float, float]:
    finite = bool(probe["finite"])
    return (
        int(finite),
        int(probe["nonconstant_class_count"]) if finite else 0,
        int(probe["prompt_map_changed_class_count"]) if finite else 0,
        float(probe["mean_prompt_probability_change"])
        if finite
        else -float("inf"),
        -float(probe["mean_loss"]) if finite else -float("inf"),
    )
```

This probe uses only the reserved training-split examples and the first training prompt per class. It never reads held-out prompts or official validation samples.

- [ ] **Step 4: Run probe tests to verify GREEN**

Run the Step 2 command again.

Expected: `2 passed`.

- [ ] **Step 5: Commit probe diagnostics**

```bash
git add geoadapter/bench/run_geovlm_prompt_segmentation.py tests/test_geovlm_prompt_runner.py
git commit -m "feat: add GeoVLM checkpoint selection probe"
```

### Task 5: Separate Resumable Last State From Selected Best State

**Files:**
- Modify: `geoadapter/bench/run_geovlm_prompt_segmentation.py`
- Modify: `tests/test_geovlm_prompt_runner.py`

- [ ] **Step 1: Replace the old checkpoint test with a failing degradation test**

First expand `_TinyPromptDataset` to three deterministic examples per class so
the default two-positive probe reservation still leaves one training example
per class. Keep the first three indices as building, water, and road so the
focused probe test remains explicit:

```python
class _TinyPromptDataset(Dataset):
    def __init__(self):
        self.images = []
        self.masks = []
        for _repeat in range(3):
            for class_id, location in ((1, (0, 0)), (3, (0, 4)), (4, (4, 0))):
                mask = torch.zeros(8, 8, dtype=torch.long)
                row, col = location
                mask[row : row + 4, col : col + 4] = class_id
                image = torch.zeros(3, 8, 8)
                image[0] = mask.eq(1)
                image[1] = mask.eq(3)
                image[2] = mask.eq(4)
                self.images.append(image)
                self.masks.append(mask)
        self.images.append(torch.zeros(3, 8, 8))
        self.masks.append(torch.full((8, 8), 2, dtype=torch.long))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.masks[index]
```

Initialize `_TinyConditionalModel.condition_bias` with distinct deterministic
values so the end-to-end injected prompt model has real condition-dependent
probability maps before optimization:

```python
        self.condition_bias = nn.Parameter(torch.tensor([0.0, 0.25, 0.5]))
```

Give `test_runner_appends_skips_and_reloads_with_injected_builders` a
`monkeypatch` argument and wrap the real epoch/probe helpers so its two-epoch
fixture deterministically selects a decreasing epoch-2 prefix while still
exercising real train steps and probe calculations:

```python
    import geoadapter.bench.run_geovlm_prompt_segmentation as runner

    real_train_one_epoch = runner._train_one_epoch
    real_evaluate_probe = runner._evaluate_probe
    epoch_calls = 0
    probe_calls = 0

    def decreasing_train_one_epoch(*args, **kwargs):
        nonlocal epoch_calls
        result = real_train_one_epoch(*args, **kwargs)
        result["loss"] = 2.0 - epoch_calls
        epoch_calls += 1
        return result

    def later_probe_ranks_higher(*args, **kwargs):
        nonlocal probe_calls
        result = real_evaluate_probe(*args, **kwargs)
        probe_calls += 1
        result["prompt_map_changed_class_count"] = 3
        result["mean_prompt_probability_change"] = float(probe_calls)
        return result

    monkeypatch.setattr(runner, "_train_one_epoch", decreasing_train_one_epoch)
    monkeypatch.setattr(runner, "_evaluate_probe", later_probe_ranks_higher)
```

Update `test_runner_appends_skips_and_reloads_with_injected_builders` to expect four checkpoint files for two method/seed pairs and names ending in `.last.pt` or `.best.pt`:

```python
    checkpoint_names = sorted(path.name for path in checkpoint_dir.glob("*.pt"))
    assert len(checkpoint_names) == 4
    assert sum(name.endswith(".last.pt") for name in checkpoint_names) == 2
    assert sum(name.endswith(".best.pt") for name in checkpoint_names) == 2
```

In the same test, assert the JSON-native diagnostics and total cap recorded on
every row:

```python
    assert all(row["training_contract"] == "paper12.geovlm_prompt_training.v2" for row in rows)
    assert all(row["source_training_size"] == 10 for row in rows)
    assert all(row["target_present_pool_size"] == 9 for row in rows)
    assert all(row["excluded_no_target_count"] == 1 for row in rows)
    assert all(row["excluded_no_target_share"] == 0.1 for row in rows)
    assert all(len(row["probe_sha256"]) == 64 for row in rows)
    assert all(
        set(row["probe_indices_by_class"]) == {"building", "water", "road"}
        for row in rows
    )
    assert all(
        row["observed_empty_target_share"]
        <= config["experiment"]["empty_target_cap"]
        for row in rows
    )
    assert all(len(row["full_loss_history"]) == 2 for row in rows)
    assert all(len(row["loss_history"]) == row["best_epoch"] for row in rows)
```

Replace `test_runner_resumes_an_incomplete_epoch_checkpoint` with this selection test. The fake epoch mutates a trainable tensor so saved best and last states are observably different, and the fake probe makes epoch 1 rank above epoch 2:

```python
def test_runner_keeps_degraded_last_state_but_evaluates_and_reproduces_best(
    monkeypatch, tmp_path
):
    import geoadapter.bench.run_geovlm_prompt_segmentation as runner

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    prithvi = tmp_path / "prithvi.pt"
    prithvi.write_bytes(b"tiny-prithvi")
    config["prithvi"]["checkpoint"] = str(prithvi)
    config["experiment"]["prompt_config"] = str(
        Path("geoadapter/bench/configs/geovlm_prompts.yaml").resolve()
    )
    config["experiment"]["epochs"] = 2
    config["experiment"]["batch_size"] = 2
    config["experiment"]["probe_positives_per_class"] = 1
    config["evaluation"]["preview_count"] = 0
    epoch_values = iter((1.0, 2.0))
    evaluated_values = []

    def model_builder(_config, _method, device):
        return _TinyConditionalModel().to(device)

    def fake_train_one_epoch(trainer, *_args, **_kwargs):
        value = next(epoch_values)
        with torch.no_grad():
            trainer.model.condition_bias.fill_(value)
        return {
            "loss": value,
            "sample_count": 2,
            "empty_target_count": 0,
            "prompt_counts": {"building": 1, "water": 1, "road": 0},
            "nonempty_prompt_counts": {"building": 1, "water": 1, "road": 0},
        }

    def fake_probe(trainer, *_args, **_kwargs):
        value = float(trainer.model.condition_bias[0])
        return {
            "finite": True,
            "mean_loss": value,
            "nonconstant_class_count": 3,
            "prompt_map_changed_class_count": 3,
            "mean_prompt_probability_change": 1.0 / value,
            "classes": {},
        }

    def fake_evaluate(trainer, *_args, **_kwargs):
        evaluated_values.append(float(trainer.model.condition_bias[0]))
        return [
            {
                "method": BASELINE_METHOD,
                "seed": 123,
                "class_name": name,
                "prediction_nonconstant": True,
            }
            for name in ("building", "water", "road")
        ]

    monkeypatch.setattr(runner, "_train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(runner, "_evaluate_probe", fake_probe)
    monkeypatch.setattr(runner, "_evaluate_method", fake_evaluate)
    dataset = _TinyPromptDataset()

    rows = runner._run_pair(
        config,
        BASELINE_METHOD,
        123,
        dataset,
        dataset,
        tmp_path / "checkpoints",
        tmp_path / "previews",
        "cpu",
        model_builder,
    )

    last_path = tmp_path / "checkpoints" / f"{BASELINE_METHOD}__seed123.last.pt"
    best_path = tmp_path / "checkpoints" / f"{BASELINE_METHOD}__seed123.best.pt"
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    assert last["epoch"] == 2
    assert best["epoch"] == 1
    assert float(last["trainable_model"]["condition_bias"][0]) == 2.0
    assert float(best["trainable_model"]["condition_bias"][0]) == 1.0
    assert evaluated_values == [1.0]
    assert all(row["checkpoint_reproduced"] is True for row in rows)
    assert all(row["best_epoch"] == 1 for row in rows)
    assert all(row["loss_history"] == [1.0] for row in rows)
    assert all(row["full_loss_history"] == [1.0, 2.0] for row in rows)
```

- [ ] **Step 2: Run runner checkpoint tests to verify RED**

Run:

```bash
python -m pytest tests/test_geovlm_prompt_runner.py::test_runner_appends_skips_and_reloads_with_injected_builders tests/test_geovlm_prompt_runner.py::test_runner_keeps_degraded_last_state_but_evaluates_and_reproduces_best -v
```

Expected: FAIL because the runner still writes one `.pt`, never ranks a probe, and evaluates the last epoch.

- [ ] **Step 3: Make `_train_one_epoch` consume assigned class names and return diagnostics**

Import `prompt_batch_from_class_names` and the four training helpers from `geoadapter.bench.geovlm_training`. Replace `_train_one_epoch` with:

```python
def _train_one_epoch(
    trainer,
    loader,
    prompt_config: PromptConfig,
    weights: dict[str, float],
    seed: int,
    method: str,
):
    generator = torch.Generator().manual_seed(seed)
    losses = []
    prompt_counts = {name: 0 for name in PROMPT_TARGET_CLASS_IDS}
    nonempty_prompt_counts = {name: 0 for name in PROMPT_TARGET_CLASS_IDS}
    empty_target_count = 0
    sample_count = 0
    for images, masks, class_names in loader:
        batch = prompt_batch_from_class_names(
            masks,
            tuple(class_names),
            prompt_config,
            generator=generator,
        )
        conditions = batch.prompts if method == PROMPT_METHOD else batch.class_ids
        positive_weights = _positive_weights_for_batch(
            batch.class_names, weights, trainer.device
        )
        losses.append(
            trainer.train_step(images, conditions, batch.targets, positive_weights)
        )
        empty_flags = batch.targets.flatten(1).sum(dim=1).eq(0).tolist()
        for name, is_empty in zip(batch.class_names, empty_flags):
            prompt_counts[name] += 1
            nonempty_prompt_counts[name] += int(not is_empty)
        empty_target_count += batch.empty_count
        sample_count += len(batch.class_names)
    trainer.scheduler.step()
    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "sample_count": sample_count,
        "empty_target_count": empty_target_count,
        "prompt_counts": prompt_counts,
        "nonempty_prompt_counts": nonempty_prompt_counts,
    }
```

Add `_merge_training_stats` to sum epoch counts across resume boundaries:

```python
def _merge_training_stats(total, epoch):
    total["sample_count"] += int(epoch["sample_count"])
    total["empty_target_count"] += int(epoch["empty_target_count"])
    for field in ("prompt_counts", "nonempty_prompt_counts"):
        for name in PROMPT_TARGET_CLASS_IDS:
            total[field][name] += int(epoch[field][name])
```

- [ ] **Step 4: Replace `_run_pair` with target-present, probe-ranked best/last orchestration**

Use these exact checkpoint names:

```python
    checkpoint_base = Path(checkpoint_dir) / f"{method}__seed{seed}"
    last_path = checkpoint_base.with_suffix(".last.pt")
    best_path = checkpoint_base.with_suffix(".best.pt")
    legacy_path = checkpoint_base.with_suffix(".pt")
```

At the start, reject `legacy_path` with an archive instruction; scan and split the training data before constructing the loader:

```python
    if legacy_path.exists():
        raise ValueError(
            f"incompatible legacy checkpoint {legacy_path}; archive it before recovery"
        )
    pool = scan_target_present_pool(train)
    split = reserve_training_probe(
        pool,
        seed=seed,
        positives_per_class=int(
            config["experiment"]["probe_positives_per_class"]
        ),
    )
```

Initialize `losses`, `probe_history`, aggregate counts, `best_rank`, and `best_epoch`. If `.last.pt` exists, validate its metadata, restore trainer/optimizer/scheduler, and restore every history/count/rank field. Reject a lone `.best.pt` without `.last.pt`. For every remaining epoch, run exactly this sequence:

```python
        assignments = build_epoch_assignments(
            split,
            batch_size=int(config["experiment"]["batch_size"]),
            empty_target_cap=float(config["experiment"]["empty_target_cap"]),
            seed=seed + epoch,
        )
        loader = DataLoader(
            AssignedPromptDataset(train, assignments),
            batch_size=int(config["experiment"]["batch_size"]),
            shuffle=False,
        )
        epoch_result = _train_one_epoch(
            trainer,
            loader,
            prompt_config,
            weights,
            seed + epoch,
            method,
        )
        losses.append(float(epoch_result["loss"]))
        _merge_training_stats(training_stats, epoch_result)
        probe = _evaluate_probe(
            trainer,
            train,
            split.probe_indices_by_class,
            prompt_config,
            weights,
            method,
        )
        probe["epoch"] = epoch + 1
        probe["training_loss"] = float(epoch_result["loss"])
        probe_history.append(probe)
        current_rank = probe_rank(probe)
        state = trainer.state_dict(epoch=epoch + 1, metadata=metadata)
        state.update(
            {
                "loss_history": list(losses),
                "probe_history": list(probe_history),
                "training_stats": training_stats,
                "probe_indices_by_class": {
                    name: list(indices)
                    for name, indices in split.probe_indices_by_class.items()
                },
                "probe_sha256": split.probe_sha256,
            }
        )
        if best_rank is None or current_rank > best_rank:
            best_rank = current_rank
            best_epoch = epoch + 1
            best_state = dict(state)
            best_state["best_epoch"] = best_epoch
            best_state["best_probe_rank"] = list(best_rank)
            _save_checkpoint(best_path, best_state)
        state["best_epoch"] = best_epoch
        state["best_probe_rank"] = list(best_rank)
        _save_checkpoint(last_path, state)
```

After training, load and validate `.best.pt`, load it into the active trainer, and use only that trainer and path for final evaluation and `_checkpoint_reproduces`. Set `selected_losses = losses[:best_epoch]`. Add these exact fields to every raw row:

```python
        row.update(
            {
                "training_contract": metadata["training_contract"],
                "checkpoint_reproduced": reproduced,
                "source_training_size": pool.source_size,
                "target_present_pool_size": len(pool.samples),
                "excluded_no_target_count": pool.excluded_no_target_count,
                "excluded_no_target_share": pool.excluded_no_target_share,
                "probe_indices_by_class": {
                    name: list(indices)
                    for name, indices in split.probe_indices_by_class.items()
                },
                "probe_sha256": split.probe_sha256,
                "per_class_prompt_counts": training_stats["prompt_counts"],
                "per_class_nonempty_prompt_counts": training_stats[
                    "nonempty_prompt_counts"
                ],
                "observed_empty_target_count": training_stats[
                    "empty_target_count"
                ],
                "observed_training_sample_count": training_stats["sample_count"],
                "observed_empty_target_share": (
                    training_stats["empty_target_count"]
                    / training_stats["sample_count"]
                ),
                "best_epoch": best_epoch,
                "best_probe_rank": list(best_rank),
                "best_probe": probe_history[best_epoch - 1],
                "full_loss_history": list(losses),
                "loss_history": selected_losses,
                "loss_first": selected_losses[0],
                "loss_last": selected_losses[-1],
            }
        )
```

Keep final validation complete and unchanged. Do not pass `split.training_samples` or probe samples into `_evaluate_method`.

- [ ] **Step 5: Add an exact `.last.pt` resume regression test**

Add this test, which saves a complete epoch-1 last/best pair, resumes exactly
one epoch, and confirms that the worse resumed epoch does not replace best:

```python
def test_runner_resumes_last_checkpoint_without_replacing_better_best(
    monkeypatch, tmp_path
):
    import geoadapter.bench.run_geovlm_prompt_segmentation as runner

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    prithvi = tmp_path / "prithvi.pt"
    prithvi.write_bytes(b"tiny-prithvi")
    config["prithvi"]["checkpoint"] = str(prithvi)
    config["experiment"]["prompt_config"] = str(
        Path("geoadapter/bench/configs/geovlm_prompts.yaml").resolve()
    )
    config["experiment"]["epochs"] = 2
    config["experiment"]["batch_size"] = 2
    config["experiment"]["probe_positives_per_class"] = 1
    config["evaluation"]["preview_count"] = 0
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    dataset = _TinyPromptDataset()
    pool = runner.scan_target_present_pool(dataset)
    split = runner.reserve_training_probe(
        pool, seed=123, positives_per_class=1
    )

    def model_builder(_config, _method, device):
        return _TinyConditionalModel().to(device)

    trainer = build_trainer(
        model_builder(config, BASELINE_METHOD, "cpu"), config, "cpu"
    )
    state = trainer.state_dict(
        epoch=1,
        metadata=checkpoint_metadata(config, BASELINE_METHOD, 123),
    )
    state.update(
        {
            "loss_history": [2.0],
            "probe_history": [
                {
                    "epoch": 1,
                    "training_loss": 2.0,
                    "finite": True,
                    "mean_loss": 2.0,
                    "nonconstant_class_count": 3,
                    "prompt_map_changed_class_count": 3,
                    "mean_prompt_probability_change": 0.5,
                    "classes": {},
                }
            ],
            "training_stats": {
                "sample_count": 2,
                "empty_target_count": 0,
                "prompt_counts": {"building": 1, "water": 1, "road": 0},
                "nonempty_prompt_counts": {
                    "building": 1,
                    "water": 1,
                    "road": 0,
                },
            },
            "probe_indices_by_class": split.probe_indices_by_class,
            "probe_sha256": split.probe_sha256,
            "best_epoch": 1,
            "best_probe_rank": [1, 3, 3, 0.5, -2.0],
        }
    )
    last_path = checkpoint_dir / f"{BASELINE_METHOD}__seed123.last.pt"
    best_path = checkpoint_dir / f"{BASELINE_METHOD}__seed123.best.pt"
    torch.save(state, last_path)
    torch.save(state, best_path)
    train_calls = []

    def fake_train_one_epoch(trainer, *_args, **_kwargs):
        train_calls.append(1)
        with torch.no_grad():
            trainer.model.condition_bias.fill_(2.0)
        return {
            "loss": 3.0,
            "sample_count": 2,
            "empty_target_count": 0,
            "prompt_counts": {"building": 1, "water": 1, "road": 0},
            "nonempty_prompt_counts": {"building": 1, "water": 1, "road": 0},
        }

    def worse_probe(*_args, **_kwargs):
        return {
            "finite": True,
            "mean_loss": 3.0,
            "nonconstant_class_count": 3,
            "prompt_map_changed_class_count": 3,
            "mean_prompt_probability_change": 0.1,
            "classes": {},
        }

    monkeypatch.setattr(runner, "_train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(runner, "_evaluate_probe", worse_probe)
    monkeypatch.setattr(
        runner,
        "_evaluate_method",
        lambda *_args, **_kwargs: [
            {
                "method": BASELINE_METHOD,
                "seed": 123,
                "class_name": name,
                "prediction_nonconstant": True,
            }
            for name in ("building", "water", "road")
        ],
    )

    rows = runner._run_pair(
        config,
        BASELINE_METHOD,
        123,
        dataset,
        dataset,
        checkpoint_dir,
        tmp_path / "previews",
        "cpu",
        model_builder,
    )

    resumed = torch.load(last_path, map_location="cpu", weights_only=False)
    selected = torch.load(best_path, map_location="cpu", weights_only=False)
    assert train_calls == [1]
    assert resumed["epoch"] == 2
    assert resumed["loss_history"] == [2.0, 3.0]
    assert selected["epoch"] == 1
    assert all(row["best_epoch"] == 1 for row in rows)
    assert all(row["full_loss_history"] == [2.0, 3.0] for row in rows)
    assert all(row["loss_history"] == [2.0] for row in rows)
```

- [ ] **Step 6: Run runner tests to verify GREEN**

Run:

```bash
python -m pytest tests/test_geovlm_prompt_runner.py -v
```

Expected: every runner test PASS, including best/last selection, resume, reproduction, and selected loss-prefix assertions.

- [ ] **Step 7: Commit best/last checkpoint orchestration**

```bash
git add geoadapter/bench/run_geovlm_prompt_segmentation.py tests/test_geovlm_prompt_runner.py
git commit -m "fix: select GeoVLM best probe checkpoint"
```

### Task 6: Reject Failed-Contract Raw Results And Checkpoints Without Mutation

**Files:**
- Modify: `geoadapter/bench/run_geovlm_prompt_segmentation.py`
- Modify: `tests/test_geovlm_prompt_runner.py`

- [ ] **Step 1: Write failing immutability tests for old raw and checkpoint artifacts**

Add:

```python
def test_runner_rejects_old_raw_results_without_modifying_them(tmp_path):
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    output = tmp_path / "failed.json"
    original = b'{"schema":"paper12.geovlm_prompt_results.v1","rows":[]}'
    output.write_bytes(original)
    dataset_called = False

    def dataset_builder(_config):
        nonlocal dataset_called
        dataset_called = True
        return _TinyPromptDataset(), _TinyPromptDataset()

    with pytest.raises(ValueError, match="archive.*before recovery"):
        run_experiment(
            config,
            output_path=output,
            summary_output_path=tmp_path / "summary.json",
            checkpoint_dir=tmp_path / "checkpoints",
            preview_dir=tmp_path / "previews",
            stage="seed42",
            dataset_builder=dataset_builder,
        )

    assert output.read_bytes() == original
    assert dataset_called is False
    assert not (tmp_path / "summary.json").exists()


def test_pair_rejects_legacy_checkpoint_without_modifying_it(tmp_path):
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    prithvi = tmp_path / "prithvi.pt"
    prithvi.write_bytes(b"tiny-prithvi")
    config["prithvi"]["checkpoint"] = str(prithvi)
    config["experiment"]["prompt_config"] = str(
        Path("geoadapter/bench/configs/geovlm_prompts.yaml").resolve()
    )
    config["experiment"]["probe_positives_per_class"] = 1
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    legacy = checkpoint_dir / f"{BASELINE_METHOD}__seed123.pt"
    legacy.write_bytes(b"failed-checkpoint")

    with pytest.raises(ValueError, match="archive.*before recovery"):
        _run_pair(
            config,
            BASELINE_METHOD,
            123,
            _TinyPromptDataset(),
            _TinyPromptDataset(),
            checkpoint_dir,
            tmp_path / "previews",
            "cpu",
            lambda _config, _method, _device: _TinyConditionalModel(),
        )

    assert legacy.read_bytes() == b"failed-checkpoint"
    assert not list(checkpoint_dir.glob("*.last.pt"))
    assert not list(checkpoint_dir.glob("*.best.pt"))
```

Add `import pytest` and `_run_pair` to the test imports if they are not already present.

- [ ] **Step 2: Run rejection tests to verify RED**

Run:

```bash
python -m pytest tests/test_geovlm_prompt_runner.py::test_runner_rejects_old_raw_results_without_modifying_them tests/test_geovlm_prompt_runner.py::test_pair_rejects_legacy_checkpoint_without_modifying_it -v
```

Expected: FAIL because old raw payloads are currently accepted and the archive error contract is absent.

- [ ] **Step 3: Validate the configured and persisted training contract before any write**

Add the runner constant and validation helpers:

```python
TRAINING_CONTRACT = "paper12.geovlm_prompt_training.v2"


def _require_training_contract(config):
    actual = config["experiment"].get("training_contract")
    if actual != TRAINING_CONTRACT:
        raise ValueError(
            f"unsupported GeoVLM training contract: {actual!r}; "
            f"expected {TRAINING_CONTRACT!r}"
        )
    return actual


def _rows_from_compatible_payload(payload, path, training_contract):
    if (
        not isinstance(payload, dict)
        or payload.get("training_contract") != training_contract
        or not isinstance(payload.get("rows"), list)
        or any(
            row.get("training_contract") != training_contract
            for row in payload["rows"]
        )
    ):
        raise ValueError(
            f"incompatible GeoVLM result artifact {path}; "
            "archive it before recovery"
        )
    return payload["rows"]
```

Call `_require_training_contract(config)` at the top of `run_experiment`. When `output_path` exists, parse it and call `_rows_from_compatible_payload` before building datasets or writing summary output. Change `_raw_results_payload` to accept the contract and emit:

```python
    payload = {
        "schema": "paper12.geovlm_prompt_results.v2",
        "training_contract": training_contract,
        "rows": rows,
    }
```

Update the raw schema assertion in the end-to-end runner test to v2. Ensure each row already receives `training_contract` from Task 5. Preserve the legacy checkpoint rejection at the very start of `_run_pair`, before model construction or checkpoint writes.

- [ ] **Step 4: Run rejection and complete runner tests to verify GREEN**

Run:

```bash
python -m pytest tests/test_geovlm_prompt_runner.py -v
```

Expected: all runner tests PASS and both byte-for-byte immutability assertions hold.

- [ ] **Step 5: Commit failed-artifact isolation**

```bash
git add geoadapter/bench/run_geovlm_prompt_segmentation.py tests/test_geovlm_prompt_runner.py
git commit -m "fix: isolate failed GeoVLM training artifacts"
```

### Task 7: Generate A Recovery-Safe Colab And Document The Real Run Sequence

**Files:**
- Modify: `tests/test_paper12_colab_notebooks.py`
- Modify: `scripts/make_paper12_colab_notebooks.py`
- Modify: `colab/paper12_geovlm_prompt_segmentation_colab.ipynb`
- Modify: `docs/geovlm_prompt_segmentation_mvp.md`

- [ ] **Step 1: Write failing notebook contract assertions**

Extend `test_paper12_geovlm_prompt_segmentation_colab_contract` with:

```python
    assert 'config["text_encoder"]["cache_dir"] = str(HF_CACHE_DIR)' in text
    assert "paper12.geovlm_prompt_training.v2" in text
    assert "ARCHIVE_FAILED_RUN = False" in text
    assert "failed_seed42_20260724" in text
    assert "archive it before recovery" in text
    assert "siglip_film_dense_similarity_houlsby__seed42.best.pt" in text
    assert "siglip_film_dense_similarity_houlsby__seed42.pt" in text
```

- [ ] **Step 2: Run the notebook contract test to verify RED**

Run:

```bash
python -m pytest tests/test_paper12_colab_notebooks.py::test_paper12_geovlm_prompt_segmentation_colab_contract -v
```

Expected: FAIL because the generated config omits `cache_dir`, there is no manual archive gate, and the notebook inspects the legacy checkpoint name.

- [ ] **Step 3: Add an explicit manual archive gate to the notebook generator**

Remove the existing Drive-to-local JSON copy loop from the path setup cell so
an incompatible failed JSON is never copied into `RAW_JSON` before archival.
Immediately after that setup cell, add a generated code cell with this logic:

```python
# Archive the failed seed-42 artifacts once before running the v2 recovery.
ARCHIVE_FAILED_RUN = False
FAILED_ARCHIVE_DIR = DRIVE_RESULTS_DIR / "failed_seed42_20260724"
FAILED_RAW_JSON = DRIVE_RAW_JSON
FAILED_SUMMARY_JSON = DRIVE_SUMMARY_JSON
FAILED_CHECKPOINT = (
    CHECKPOINT_DIR / "siglip_film_dense_similarity_houlsby__seed42.pt"
)
FAILED_PREVIEWS = sorted(PREVIEW_DIR.glob("seed42__*.png"))
failed_artifacts = [
    path
    for path in (
        FAILED_RAW_JSON,
        FAILED_SUMMARY_JSON,
        FAILED_CHECKPOINT,
    )
    if path.exists()
] + FAILED_PREVIEWS
if failed_artifacts and not ARCHIVE_FAILED_RUN:
    raise RuntimeError(
        "Failed seed-42 artifacts still exist; set ARCHIVE_FAILED_RUN = True "
        "once to archive them before recovery: "
        + ", ".join(str(path) for path in failed_artifacts)
    )
if failed_artifacts:
    FAILED_ARCHIVE_DIR.mkdir(parents=True, exist_ok=False)
    for source in failed_artifacts:
        destination = FAILED_ARCHIVE_DIR / source.name
        shutil.move(str(source), str(destination))
        print("Archived", source, "to", destination)
for source, destination in (
    (DRIVE_RAW_JSON, RAW_JSON),
    (DRIVE_SUMMARY_JSON, SUMMARY_JSON),
):
    if source.exists():
        shutil.copy2(source, destination)
```

This is an explicit user-controlled move. The default `False` stops rather
than overwrites the supplied failed JSON, checkpoint, or seed-42 previews.
Globbing preview files instead of testing the already-created preview
directory prevents an empty directory from triggering the archive gate on
every clean rerun. Moving the Drive-to-local copy after the gate prevents the
old v1 JSON from poisoning the fresh local output path.

- [ ] **Step 4: Write the explicit cache and new best-checkpoint path into generated cells**

In the Colab config cell add:

```python
config["experiment"]["training_contract"] = "paper12.geovlm_prompt_training.v2"
config["text_encoder"]["cache_dir"] = str(HF_CACHE_DIR)
```

In the seed-42 verification cell replace the legacy reload path with:

```python
checkpoint_path = (
    CHECKPOINT_DIR
    / "siglip_film_dense_similarity_houlsby__seed42.best.pt"
)
checkpoint_payload = torch.load(
    checkpoint_path, map_location="cpu", weights_only=False
)
assert checkpoint_payload["metadata"]["training_contract"] == (
    "paper12.geovlm_prompt_training.v2"
)
print(
    "selected epoch:",
    checkpoint_payload["best_epoch"],
    "rank:",
    checkpoint_payload["best_probe_rank"],
)
```

- [ ] **Step 5: Regenerate the notebook and verify valid Python cells**

Run:

```bash
python scripts/make_paper12_colab_notebooks.py
python -m pytest tests/test_paper12_colab_notebooks.py::test_paper12_geovlm_prompt_segmentation_colab_contract tests/test_paper12_colab_notebooks.py::test_paper12_geovlm_prompt_segmentation_code_cells_are_valid_python -v
```

Expected: generator reports the GeoVLM notebook written or unchanged as appropriate, then `2 passed`.

- [ ] **Step 6: Update the recovery documentation**

In `docs/geovlm_prompt_segmentation_mvp.md`, change Status to state that the first real seed-42 run failed three smoke checks and that no performance claim is authorized. Add a `Seed-42 recovery` section that states:

```text
1. Set ARCHIVE_FAILED_RUN = True once and run the archive cell.
2. Confirm the failed JSON, summary, legacy checkpoint, and previews are under failed_seed42_20260724.
3. Reset ARCHIVE_FAILED_RUN = False.
4. Run the notebook through the seed-42 cell under paper12.geovlm_prompt_training.v2.
5. Inspect source/pool/excluded counts, probe history, observed empty-target share, best epoch, full loss history, and selected loss prefix.
6. Keep RUN_FULL_MATRIX = False unless all four unchanged smoke checks pass.
```

Document that `.last.pt` is the only resume source, `.best.pt` is the only evaluation/inference source, official validation remains complete, and a second failure requires a separately approved optimizer-stabilization design. Do not add any result number to the Paper12 manuscript.

- [ ] **Step 7: Run notebook and documentation checks**

Run:

```bash
python -m pytest tests/test_paper12_colab_notebooks.py -v
python scripts/make_paper12_colab_notebooks.py
$notebookHash = (Get-FileHash colab/paper12_geovlm_prompt_segmentation_colab.ipynb -Algorithm SHA256).Hash
python scripts/make_paper12_colab_notebooks.py
$regeneratedHash = (Get-FileHash colab/paper12_geovlm_prompt_segmentation_colab.ipynb -Algorithm SHA256).Hash
if ($notebookHash -ne $regeneratedHash) { throw "GeoVLM notebook generation is not deterministic" }
```

Expected: all notebook tests PASS and the two SHA-256 values are identical.

- [ ] **Step 8: Commit the Colab and recovery runbook**

```bash
git add tests/test_paper12_colab_notebooks.py scripts/make_paper12_colab_notebooks.py colab/paper12_geovlm_prompt_segmentation_colab.ipynb docs/geovlm_prompt_segmentation_mvp.md
git commit -m "docs: add GeoVLM seed42 recovery workflow"
```

### Task 8: Verify The Complete Offline Recovery And Guard The Real-Run Boundary

**Files:**
- Verify: all files changed in Tasks 1-7

- [ ] **Step 1: Run the complete focused GeoVLM suite**

Run:

```bash
python -m pytest tests/test_prompt_segmentation_data.py tests/test_prithvi_position_embeddings.py tests/test_prompt_segmentation_model.py tests/test_prompt_segmentation_engine.py tests/test_geovlm_training.py tests/test_geovlm_prompt_summary.py tests/test_geovlm_prompt_runner.py tests/test_geovlm_prompt_inference.py tests/test_paper12_colab_notebooks.py -v
```

Expected: all focused tests PASS with only previously known dependency warnings.

- [ ] **Step 2: Run the complete maintained test suite**

Run:

```bash
python -m pytest -q
```

Expected: all maintained tests PASS; skipped tests and the two existing rasterio warnings are allowed, but no new failure or warning is allowed.

- [ ] **Step 3: Verify syntax, notebook determinism, and whitespace**

Run:

```bash
python -m compileall -q geoadapter scripts tests
python scripts/make_paper12_colab_notebooks.py
git diff --exit-code -- colab/paper12_geovlm_prompt_segmentation_colab.ipynb
git diff --check
```

Expected: every command exits 0; notebook generation produces no content change; `git diff --check` prints nothing.

- [ ] **Step 4: Audit the staged/tracked artifact boundary**

Run:

```bash
git status --short
git diff --name-only master...HEAD
git diff --stat master...HEAD
```

Expected: only source, tests, config, generated notebook, design, plan, and documentation are listed. No JSON result, `.pt` checkpoint, Hugging Face cache file, LandCoverAI data, PNG preview, or `missing/` file is present.

- [ ] **Step 5: Commit any verification-only correction, then stop before the real run**

If verification required a source correction, rerun the smallest failing test first, repeat Steps 1-4, and commit only that correction:

```bash
git add geoadapter tests scripts colab docs
git commit -m "test: finalize GeoVLM seed42 recovery"
```

Do not run seeds 123/456 or the baseline matrix locally. The next state-changing action is the user-controlled A100 Colab seed-42 run after manual archival. A fresh result may authorize the full matrix only when `finite_decreasing_loss`, `nonconstant_predictions`, `prompt_dependent_probability_maps`, and `checkpoint_reproduced` are all true.
