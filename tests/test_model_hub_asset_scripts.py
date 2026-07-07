import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]


def test_fetch_public_sample_dry_run_lists_crop_source():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/model_hub/fetch_public_sample.py",
            "--asset",
            "prithvi_crop",
            "--dry-run",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "ibm-nasa-geospatial" in result.stdout
    assert "dry-run" in result.stdout.lower()


def test_verify_assets_outputs_json():
    result = subprocess.run(
        [sys.executable, "scripts/model_hub/verify_assets.py", "--json"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "prithvi_crop_classification_arcgis_style" in result.stdout
