import importlib.util
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import pytest


def require_parquet_engine() -> None:
    if importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet"):
        return
    pytest.skip("pyarrow or fastparquet is required for parquet label API tests")


def test_load_patch_index_and_labels(tmp_path: Path):
    # Ensure `import app.*` works when running tests from repo root.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "ae_backend"))
    require_parquet_engine()

    dataset_dir = tmp_path / "linhe_patches"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    index_df = pd.DataFrame(
        [
            {
                "patch_path": "patches/000001.tif",
                "dataset_id": "linhe_demo",
            }
        ]
    )
    index_df.to_parquet(dataset_dir / "_index.parquet", index=False)

    labels_df_in = pd.DataFrame(
        [
            {
                "dataset_id": "linhe_demo",
                "patch_path": "patches/000001.tif",
                "label": "cropland",
                "reviewer": "tester",
                "labeled_at": datetime(2026, 4, 18, tzinfo=timezone.utc),
            }
        ]
    )
    labels_df_in.to_parquet(dataset_dir / "_labels.parquet", index=False)

    from app.api.labels import load_patch_index, load_labels

    loaded_index = load_patch_index(dataset_dir)
    loaded_labels = load_labels(dataset_dir)

    assert len(loaded_index) == 1
    assert len(loaded_labels) == 1
    assert loaded_labels.iloc[0]["label"] == "cropland"
