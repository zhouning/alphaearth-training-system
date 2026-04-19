import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AE_BACKEND_DIR = os.path.join(PROJECT_ROOT, "ae_backend")
if AE_BACKEND_DIR not in sys.path:
    sys.path.insert(0, AE_BACKEND_DIR)


def test_load_patch_index_and_labels(tmp_path):
    from app.api.labels import load_patch_index, load_labels

    dataset_dir = tmp_path / "linhe_patches"
    dataset_dir.mkdir(parents=True)

    pd.DataFrame([
        {"patch_path": "linhe_patches/scene_a/p_00000_00000.npz", "dataset_id": "linhe_patches"}
    ]).to_parquet(dataset_dir / "_index.parquet")

    pd.DataFrame([
        {"dataset_id": "linhe_patches", "patch_path": "linhe_patches/scene_a/p_00000_00000.npz",
         "label": "耕地", "reviewer": "tester", "labeled_at": "2026-04-18T12:00:00"}
    ]).to_parquet(dataset_dir / "_labels.parquet")

    index_df = load_patch_index(dataset_dir)
    labels_df = load_labels(dataset_dir)

    assert len(index_df) == 1
    assert len(labels_df) == 1
    assert labels_df.iloc[0]["label"] == "耕地"


def test_load_labels_returns_empty_when_missing(tmp_path):
    from app.api.labels import load_labels

    dataset_dir = tmp_path / "linhe_patches"
    dataset_dir.mkdir(parents=True)

    labels_df = load_labels(dataset_dir)
    assert len(labels_df) == 0
    assert list(labels_df.columns) == ["dataset_id", "patch_path", "label", "reviewer", "labeled_at"]


def test_labels_next_returns_first_unlabeled_patch(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.core.config import settings

    dataset_dir = tmp_path / "linhe_patches"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame([
        {"patch_path": "linhe_patches/scene_a/p_00000_00000.npz"},
        {"patch_path": "linhe_patches/scene_a/p_00000_00128.npz"},
    ]).to_parquet(dataset_dir / "_index.parquet")
    pd.DataFrame([
        {"dataset_id": "linhe_patches", "patch_path": "linhe_patches/scene_a/p_00000_00000.npz",
         "label": "耕地", "reviewer": "tester", "labeled_at": "2026-04-18T12:00:00"}
    ]).to_parquet(dataset_dir / "_labels.parquet")

    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))

    client = TestClient(app)
    resp = client.get("/api/ae/labels/next?dataset_id=linhe_patches")
    data = resp.json()["data"]

    assert resp.status_code == 200
    assert data["patch_path"] == "linhe_patches/scene_a/p_00000_00128.npz"


def test_labels_next_returns_first_labeled_patch_in_review_mode(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.core.config import settings

    dataset_dir = tmp_path / "linhe_patches"
    dataset_dir.mkdir(parents=True)

    pd.DataFrame([
        {"patch_path": "linhe_patches/scene_a/p_00000_00000.npz"},
        {"patch_path": "linhe_patches/scene_a/p_00000_00128.npz"},
    ]).to_parquet(dataset_dir / "_index.parquet")

    # Note: order matters; review mode should return the first row in _labels.parquet.
    pd.DataFrame([
        {"dataset_id": "linhe_patches", "patch_path": "linhe_patches/scene_a/p_00000_00128.npz",
         "label": "道路", "reviewer": "tester", "labeled_at": "2026-04-18T12:00:00"},
        {"dataset_id": "linhe_patches", "patch_path": "linhe_patches/scene_a/p_00000_00000.npz",
         "label": "耕地", "reviewer": "tester", "labeled_at": "2026-04-18T12:00:01"},
    ]).to_parquet(dataset_dir / "_labels.parquet")

    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))

    client = TestClient(app)
    resp = client.get("/api/ae/labels/next", params={"dataset_id": "linhe_patches", "mode": "labeled"})
    data = resp.json()["data"]

    assert resp.status_code == 200
    assert data["patch_path"] == "linhe_patches/scene_a/p_00000_00128.npz"
    assert data["current_label"] == "道路"


def test_labels_next_filters_by_existing_label(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.core.config import settings

    dataset_dir = tmp_path / "linhe_patches"
    dataset_dir.mkdir(parents=True)

    pd.DataFrame([
        {"patch_path": "linhe_patches/scene_a/a.npz"},
        {"patch_path": "linhe_patches/scene_a/b.npz"},
    ]).to_parquet(dataset_dir / "_index.parquet")

    pd.DataFrame([
        {"dataset_id": "linhe_patches", "patch_path": "linhe_patches/scene_a/a.npz",
         "label": "耕地", "reviewer": "tester", "labeled_at": "2026-04-18T12:00:00"},
        {"dataset_id": "linhe_patches", "patch_path": "linhe_patches/scene_a/b.npz",
         "label": "林地", "reviewer": "tester", "labeled_at": "2026-04-18T12:00:01"},
    ]).to_parquet(dataset_dir / "_labels.parquet")

    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))

    client = TestClient(app)
    resp = client.get("/api/ae/labels/next", params={
        "dataset_id": "linhe_patches",
        "mode": "labeled",
        "label": "林地",
    })
    data = resp.json()["data"]

    assert resp.status_code == 200
    assert data["patch_path"] == "linhe_patches/scene_a/b.npz"
    assert data["current_label"] == "林地"


def test_labels_stats_returns_counts(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.core.config import settings

    dataset_dir = tmp_path / "linhe_patches"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame([
        {"patch_path": "linhe_patches/scene_a/a.npz"},
        {"patch_path": "linhe_patches/scene_a/b.npz"},
    ]).to_parquet(dataset_dir / "_index.parquet")
    pd.DataFrame([
        {"dataset_id": "linhe_patches", "patch_path": "linhe_patches/scene_a/a.npz",
         "label": "林地", "reviewer": "tester", "labeled_at": "2026-04-18T12:00:00"}
    ]).to_parquet(dataset_dir / "_labels.parquet")

    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))

    client = TestClient(app)
    resp = client.get("/api/ae/labels/stats?dataset_id=linhe_patches")
    data = resp.json()["data"]

    assert data["total"] == 2
    assert data["labeled"] == 1
    assert data["unlabeled"] == 1


def test_labels_preview_returns_base64_png(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.core.config import settings

    dataset_dir = tmp_path / "linhe_patches"
    scene_dir = dataset_dir / "scene_a"
    scene_dir.mkdir(parents=True)
    np.savez_compressed(scene_dir / "p_00000_00000.npz", rgb=np.ones((3, 32, 32), dtype=np.uint8) * 255)

    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))

    client = TestClient(app)
    resp = client.get("/api/ae/labels/preview", params={
        "dataset_id": "linhe_patches",
        "patch_path": "linhe_patches/scene_a/p_00000_00000.npz",
    })
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["image_base64"].startswith("data:image/png;base64,")


def test_labels_save_persists_row(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.core.config import settings

    dataset_dir = tmp_path / "linhe_patches"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame([{"patch_path": "linhe_patches/scene_a/p.npz"}]).to_parquet(dataset_dir / "_index.parquet")

    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))

    client = TestClient(app)
    resp = client.post("/api/ae/labels/save", json={
        "dataset_id": "linhe_patches",
        "patch_path": "linhe_patches/scene_a/p.npz",
        "label": "耕地",
        "reviewer": "tester",
    })

    saved = pd.read_parquet(dataset_dir / "_labels.parquet")
    assert resp.status_code == 200
    assert len(saved) == 1
    assert saved.iloc[0]["label"] == "耕地"
