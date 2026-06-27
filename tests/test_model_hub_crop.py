import sys
from pathlib import Path


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
