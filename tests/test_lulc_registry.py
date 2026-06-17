import json
import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "ae_backend"))


def test_build_lulc_capability_registry_reports_public_products_and_local_models(tmp_path: Path):
    from app.services.lulc_registry import build_lulc_capability_registry

    weights_dir = tmp_path / "weights"
    model_dir = weights_dir / "linhe_lulc"
    model_dir.mkdir(parents=True)
    (model_dir / "geoadapter__rgb_3band__seed123.pt").write_bytes(b"geo")
    (model_dir / "houlsby__rgb_3band__seed123.pt").write_bytes(b"houlsby")

    results_path = tmp_path / "linhe_lulc_seg.json"
    results_path.write_text(
        json.dumps(
            [
                {
                    "method": "geoadapter",
                    "modality": "rgb_3band",
                    "seed": 123,
                    "trainable_params": 4756,
                    "mIoU": 0.2750912667,
                },
                {
                    "method": "houlsby",
                    "modality": "rgb_3band",
                    "seed": 123,
                    "trainable_params": 1194246,
                    "mIoU": 0.2970963179,
                },
            ]
        ),
        encoding="utf-8",
    )

    public_cache_dir = tmp_path / "esri_lulc"
    public_cache_dir.mkdir()
    (public_cache_dir / "2022.tif").write_bytes(b"stub")

    registry = build_lulc_capability_registry(
        weights_dir=weights_dir,
        results_path=results_path,
        public_cache_dir=public_cache_dir,
    )

    assert registry["task"] == "lulc_segmentation"
    assert {mode["id"] for mode in registry["modes"]} == {"public_product", "local_model"}
    assert registry["default_local_model_id"] == "linhe-lulc-geoadapter-seed123"

    local_models = {model["id"]: model for model in registry["local_models"]}
    assert local_models["linhe-lulc-geoadapter-seed123"]["ready"] is True
    assert local_models["linhe-lulc-geoadapter-seed123"]["validation"]["mIoU"] == 0.2750912667
    assert local_models["linhe-lulc-houlsby-seed123"]["validation"]["mIoU"] == 0.2970963179

    public_products = {product["id"]: product for product in registry["public_products"]}
    assert public_products["esri_lulc_cache"]["status"] == "ready"
    assert public_products["esri_lulc_cache"]["available_years"] == [2022]
    assert public_products["dynamic_world_gee"]["status"] == "requires_earth_engine_auth"
    assert public_products["esa_worldcover_static"]["status"] == "not_configured"
