import os
import sys

import numpy as np
import torch
from fastapi.testclient import TestClient


# Ensure we can import the FastAPI backend (ae_backend/app)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AE_BACKEND_DIR = os.path.join(PROJECT_ROOT, "ae_backend")
if AE_BACKEND_DIR not in sys.path:
    sys.path.insert(0, AE_BACKEND_DIR)


def test_datasets_endpoint_exposes_linhe_patches_when_index_exists(tmp_path, monkeypatch):
    from app.main import app
    from app.core.config import settings

    data_dir = tmp_path / "data"
    raw_data_dir = tmp_path / "raw_data"
    linhe_dir = data_dir / "linhe_patches"
    linhe_dir.mkdir(parents=True)

    index_path = linhe_dir / "_index.parquet"
    index_path.write_bytes(b"")

    monkeypatch.setattr(settings, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(settings, "RAW_DATA_DIR", str(raw_data_dir))

    client = TestClient(app)
    resp = client.get("/api/ae/pipeline/datasets")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "success"

    ids = [d["id"] for d in payload["data"]]
    assert "linhe_patches" in ids


def test_real_patch_dataset_loads_npz_rgb_and_pads_to_five_channels(tmp_path):
    from app.services.trainer import RealPatchDataset

    linhe_root = tmp_path / "linhe_patches"
    scene_dir = linhe_root / "scene_a"
    scene_dir.mkdir(parents=True)

    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    rgb[..., 0] = 0
    rgb[..., 1] = 128
    rgb[..., 2] = 255

    np.savez(scene_dir / "p_00000_00000.npz", rgb=rgb)

    dataset = RealPatchDataset(str(linhe_root), patch_size=128)
    sample = dataset[0]

    assert sample.shape == (5, 128, 128)
    assert torch.allclose(sample[0], torch.zeros((128, 128), dtype=torch.float32))
    assert torch.allclose(sample[1], torch.full((128, 128), 128.0 / 255.0, dtype=torch.float32), atol=1e-6)
    assert torch.allclose(sample[2], torch.ones((128, 128), dtype=torch.float32))
    assert torch.allclose(sample[3], torch.zeros((128, 128), dtype=torch.float32))
    assert torch.allclose(sample[4], torch.zeros((128, 128), dtype=torch.float32))


def test_alphaearth_trainer_uses_linhe_dataset_path(tmp_path):
    from unittest.mock import MagicMock, patch

    captured_paths = []

    class FakeDataset:
        def __init__(self, path, patch_size=128):
            captured_paths.append(path)

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return torch.zeros(5, 128, 128)

    fake_model = MagicMock()
    fake_model.parameters.return_value = []
    fake_model.to.return_value = fake_model

    fake_optimizer = MagicMock()
    fake_dataloader = MagicMock()

    with patch("app.services.trainer.settings") as mock_settings, \
         patch("app.services.trainer.RealPatchDataset", FakeDataset), \
         patch("app.services.trainer.PrithviAlphaEarthEncoder", return_value=fake_model), \
         patch("app.services.trainer.optim.Adam", return_value=fake_optimizer), \
         patch("app.services.trainer.DataLoader", return_value=fake_dataloader):
        mock_settings.DATA_DIR = str(tmp_path / "data")
        mock_settings.RAW_DATA_DIR = str(tmp_path / "data" / "weights" / "raw_data")

        from app.services.trainer import AlphaEarthTrainer

        AlphaEarthTrainer(
            job_id="test-job-1",
            dataset_id="linhe_patches",
            ws_manager=None,
            epochs=1,
            peft_method="linear_probe",
        )

    assert len(captured_paths) == 1
    assert captured_paths[0].endswith(os.path.join("data", "linhe_patches"))


def test_resolve_dataset_path_variants():
    from app.core.config import settings
    from app.services.trainer import resolve_dataset_path

    assert resolve_dataset_path("linhe_patches") == os.path.join(settings.DATA_DIR, "linhe_patches")
    assert resolve_dataset_path("memory_abc") == "memory_abc"
    assert resolve_dataset_path("job123") == os.path.join(settings.RAW_DATA_DIR, "dataset_job123")
