import pytest
import torch
import torch.nn as nn

from geoadapter.bench.run_benchmark import build_adapter
from geoadapter.bench.run_benchmark import run_single_experiment
from geoadapter.adapters.geo_adapter import GeoAdapter
from geoadapter.adapters.learned_channel_bridge import LearnedChannelBridgeAdapter
from geoadapter.adapters.zero_pad import ZeroPadAdapter


def _raise_missing_eurosat(**_kwargs):
    raise FileNotFoundError("test dataset")


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


def test_segmentation_decoder_kwargs_defaults_to_linear():
    from geoadapter.bench.run_benchmark import _segmentation_decoder_kwargs

    assert _segmentation_decoder_kwargs({}, {}) == {
        "decoder_type": "linear",
        "hidden_dim": None,
    }


def test_segmentation_decoder_kwargs_uses_global_config():
    from geoadapter.bench.run_benchmark import _segmentation_decoder_kwargs

    assert _segmentation_decoder_kwargs(
        {},
        {"segmentation": {"decoder_type": "conv_lite", "hidden_dim": 64}},
    ) == {
        "decoder_type": "conv_lite",
        "hidden_dim": 64,
    }


def test_segmentation_decoder_kwargs_method_config_overrides_global_config():
    from geoadapter.bench.run_benchmark import _segmentation_decoder_kwargs

    assert _segmentation_decoder_kwargs(
        {"segmentation": {"decoder_type": "conv_lite", "hidden_dim": 128}},
        {"segmentation": {"decoder_type": "linear", "hidden_dim": None}},
    ) == {
        "decoder_type": "conv_lite",
        "hidden_dim": 128,
    }


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


def test_run_single_experiment_uses_backbone_metadata(monkeypatch, tmp_path):
    from dataclasses import dataclass

    import geoadapter.bench.run_benchmark as runner
    import geoadapter.data.datasets as datasets_module

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
    monkeypatch.setattr(
        datasets_module,
        "load_eurosat",
        _raise_missing_eurosat,
    )

    cfg = {
        "experiment": {
            "dataset": "eurosat",
            "dataset_root": str(tmp_path / "missing_dataset"),
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


def test_run_single_experiment_passes_segmentation_decoder_config(
    monkeypatch, tmp_path
):
    from dataclasses import dataclass

    import geoadapter.bench.run_benchmark as runner
    import geoadapter.data.datasets as datasets_module
    import geoadapter.models.heads as heads_module

    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList()
            self.weight = nn.Parameter(torch.zeros(1), requires_grad=False)

        def forward(self, x, return_spatial=False):
            features = torch.zeros(x.shape[0], 1, 32, device=x.device)
            if return_spatial:
                return features, (1, 1)
            return features.squeeze(1)

    @dataclass(frozen=True)
    class TinySpec:
        name: str
        model: nn.Module
        feature_dim: int
        input_channels: int
        blocks: nn.ModuleList

    seen = {}

    class CapturingSegmentationHead(nn.Module):
        def __init__(
            self, *, in_dim, num_classes, patch_size, decoder_type="linear", hidden_dim=None
        ):
            super().__init__()
            seen.update(
                {
                    "in_dim": in_dim,
                    "num_classes": num_classes,
                    "patch_size": patch_size,
                    "decoder_type": decoder_type,
                    "hidden_dim": hidden_dim,
                }
            )
            self.weight = nn.Parameter(torch.zeros(1))

        def forward(self, x, spatial_dims=None):
            return torch.zeros(
                x.shape[0], seen["num_classes"], seen["patch_size"], seen["patch_size"]
            )

    monkeypatch.setattr(heads_module, "SegmentationHead", CapturingSegmentationHead)
    monkeypatch.setattr(
        runner,
        "build_backbone",
        lambda cfg: TinySpec(
            name="tiny_backbone",
            model=TinyBackbone(),
            feature_dim=32,
            input_channels=3,
            blocks=nn.ModuleList(),
        ),
    )
    monkeypatch.setattr(
        datasets_module,
        "load_eurosat",
        _raise_missing_eurosat,
    )

    cfg = {
        "experiment": {
            "dataset": "eurosat",
            "dataset_root": str(tmp_path / "missing_dataset"),
            "task": "segmentation",
            "num_classes": 5,
            "epochs": 0,
            "batch_size": 8,
            "allow_synthetic_fallback": True,
        },
        "training": {"lr": 1e-3},
        "backbone": {
            "name": "tiny_backbone",
            "family": "satmae",
            "patch_size": 8,
            "pretrained": False,
        },
        "segmentation": {"decoder_type": "conv_lite", "hidden_dim": 16},
    }

    runner.run_single_experiment(
        {"name": "linear_probe", "adapter": "zero_pad", "peft": None},
        {"preset": "rgb_3band"},
        cfg,
        seed=42,
    )

    assert seen == {
        "in_dim": 32,
        "num_classes": 5,
        "patch_size": 8,
        "decoder_type": "conv_lite",
        "hidden_dim": 16,
    }
