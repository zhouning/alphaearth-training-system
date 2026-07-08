import pytest
import torch
import torch.nn as nn

from geoadapter.bench.run_benchmark import build_adapter
from geoadapter.bench.run_benchmark import run_single_experiment
from geoadapter.adapters.geo_adapter import GeoAdapter
from geoadapter.adapters.learned_channel_bridge import LearnedChannelBridgeAdapter
from geoadapter.adapters.zero_pad import ZeroPadAdapter


@pytest.mark.parametrize(
    ("kind", "expected_type"),
    [
        ("zero_pad", ZeroPadAdapter),
        ("geo_adapter", GeoAdapter),
        ("learned_channel_bridge", LearnedChannelBridgeAdapter),
    ],
)
def test_build_adapter_selects_configured_adapter(kind, expected_type):
    adapter = build_adapter(kind, in_channels=10, out_channels=6)
    assert isinstance(adapter, expected_type)


def test_build_adapter_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown adapter"):
        build_adapter("not_an_adapter", in_channels=10, out_channels=6)


def test_run_single_experiment_can_require_real_dataset(monkeypatch, tmp_path):
    class TinyBackbone(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.blocks = nn.ModuleList()
            self.weight = nn.Parameter(torch.zeros(1), requires_grad=False)

    import geoadapter.models.prithvi as prithvi_module

    monkeypatch.setattr(prithvi_module, "PrithviBackbone", TinyBackbone)
    cfg = {
        "experiment": {
            "dataset": "linhe_lulc",
            "dataset_root": str(tmp_path),
            "year": 2022,
            "task": "segmentation",
            "num_classes": 6,
            "epochs": 0,
            "batch_size": 1,
            "allow_synthetic_fallback": False,
        },
        "training": {"lr": 1e-3},
        "prithvi": {"pretrained": False, "checkpoint": None},
    }

    with pytest.raises(RuntimeError, match="synthetic fallback disabled"):
        run_single_experiment(
            {"name": "linear_probe", "adapter": "zero_pad", "peft": None},
            {"preset": "rgb_3band"},
            cfg,
            seed=42,
        )


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
