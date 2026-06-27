import sys
from pathlib import Path

import pytest


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "ae_backend"))


def test_summarize_cached_crop_demo_returns_crop_result_and_artifacts(tmp_path: Path):
    from app.services.model_hub_crop import summarize_cached_crop_demo

    crop_dir = tmp_path / "prithvi_crop_demo"
    crop_dir.mkdir()
    (crop_dir / "crop_preview.png").write_bytes(b"png")
    (crop_dir / "crop_polygons.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )
    (crop_dir / "crop_summary.csv").write_text(
        "class,pixels,fraction\nmaize,6400,0.64\n",
        encoding="utf-8",
    )

    result = summarize_cached_crop_demo(options={"crop_dir": str(crop_dir)})

    assert result["result"]["task"] == "crop_classification"
    assert result["result"]["model_id"] == "prithvi_crop_classification_arcgis_style"
    assert result["result"]["summary"]["dominant_class"] == "maize"
    assert result["result"]["summary"]["class_pixel_counts"]["maize"] == 6400
    assert result["result"]["summary"]["class_area_fraction"]["maize"] == 0.64
    assert (
        result["result"]["model_package"]["package_type"]
        == "arcgis_style_pretrained_imagery_model"
    )
    assert {artifact["kind"] for artifact in result["artifacts"]} == {
        "png",
        "geojson",
        "csv",
    }
    assert all(
        Path(artifact["path"]).name
        in {"crop_preview.png", "crop_polygons.geojson", "crop_summary.csv"}
        for artifact in result["artifacts"]
    )


def test_summarize_cached_crop_demo_returns_planned_artifact_paths_without_files(
    tmp_path: Path,
):
    from app.services.model_hub_crop import summarize_cached_crop_demo

    crop_dir = tmp_path / "empty_crop_demo"

    result = summarize_cached_crop_demo(options={"crop_dir": str(crop_dir)})

    assert result["result"]["summary"]["dominant_class"] == "maize"
    assert {artifact["kind"] for artifact in result["artifacts"]} == {
        "png",
        "geojson",
        "csv",
    }
    assert any("planned artifact paths" in log for log in result["logs"])


def test_summarize_cached_crop_demo_falls_back_when_summary_csv_is_malformed(
    tmp_path: Path,
):
    from app.services.model_hub_crop import summarize_cached_crop_demo

    crop_dir = tmp_path / "malformed_crop_demo"
    crop_dir.mkdir()
    (crop_dir / "crop_preview.png").write_bytes(b"png")
    (crop_dir / "crop_polygons.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )
    (crop_dir / "crop_summary.csv").write_text(
        "class,pixels,fraction\nmaize,not-a-number,\n",
        encoding="utf-8",
    )

    result = summarize_cached_crop_demo(options={"crop_dir": str(crop_dir)})

    assert result["result"]["summary"]["dominant_class"] == "maize"
    assert result["result"]["summary"]["class_pixel_counts"]["maize"] == 6400
    assert result["result"]["summary"]["class_area_fraction"]["maize"] == 0.457143
    assert any("invalid crop_summary.csv" in log for log in result["logs"])


@pytest.mark.parametrize(
    "csv_row",
    [
        "maize,6400,nan\n",
        "maize,6400,inf\n",
        "maize,6400,1.2\n",
        "maize,6400,-0.1\n",
        "maize,-1,0.64\n",
        "maize,not-a-number,0.64\n",
    ],
)
def test_summarize_cached_crop_demo_falls_back_for_invalid_summary_values(
    tmp_path: Path,
    csv_row: str,
):
    from app.services.model_hub_crop import summarize_cached_crop_demo

    crop_dir = tmp_path / "invalid_values_crop_demo"
    crop_dir.mkdir()
    (crop_dir / "crop_preview.png").write_bytes(b"png")
    (crop_dir / "crop_polygons.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )
    (crop_dir / "crop_summary.csv").write_text(
        f"class,pixels,fraction\n{csv_row}",
        encoding="utf-8",
    )

    result = summarize_cached_crop_demo(options={"crop_dir": str(crop_dir)})

    assert result["result"]["summary"]["dominant_class"] == "maize"
    assert result["result"]["summary"]["class_pixel_counts"]["maize"] == 6400
    assert result["result"]["summary"]["class_area_fraction"]["maize"] == 0.457143
    assert any("invalid crop_summary.csv" in log for log in result["logs"])
