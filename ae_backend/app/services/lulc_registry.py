from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

from app.core.config import PROJECT_ROOT, settings
from app.services.inference import LULC_CLASS_NAMES
from app.services.lulc_public import PUBLIC_LULC_CLASS_NAMES


DEFAULT_LOCAL_MODEL_ID = "linhe-lulc-geoadapter-seed123"

LOCAL_MODEL_SPECS = [
    {
        "id": "linhe-lulc-geoadapter-seed123",
        "method": "geoadapter",
        "seed": 123,
        "checkpoint": "linhe_lulc/geoadapter__rgb_3band__seed123.pt",
        "adapter": "GeoAdapter input-stage adapter",
        "default": True,
        "note": "Small, stable default model for RGB patch inference.",
    },
    {
        "id": "linhe-lulc-houlsby-seed123",
        "method": "houlsby",
        "seed": 123,
        "checkpoint": "linhe_lulc/houlsby__rgb_3band__seed123.pt",
        "adapter": "Houlsby transformer adapters",
        "default": False,
        "note": "Higher formal validation mIoU, heavier checkpoint.",
    },
]


def _load_metrics(results_path: Path) -> dict[tuple[str, int], dict]:
    if not results_path.exists():
        return {}
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    metrics: dict[tuple[str, int], dict] = {}
    by_method: dict[str, list[dict]] = {}
    for row in rows:
        method = row.get("method")
        seed = row.get("seed")
        if method is None or seed is None:
            continue
        metrics[(str(method), int(seed))] = {
            "mIoU": row.get("mIoU"),
            "trainable_params": row.get("trainable_params"),
            "modality": row.get("modality"),
            "seed": seed,
        }
        by_method.setdefault(str(method), []).append(row)

    for method, method_rows in by_method.items():
        values = [float(r["mIoU"]) for r in method_rows if r.get("mIoU") is not None]
        if not values:
            continue
        aggregate = {
            "mIoU_mean": mean(values),
            "mIoU_std": stdev(values) if len(values) > 1 else 0.0,
            "n_seeds": len(values),
        }
        for row in method_rows:
            seed = row.get("seed")
            if seed is not None and (method, int(seed)) in metrics:
                metrics[(method, int(seed))]["method_summary"] = aggregate
    return metrics


def _available_public_years(public_cache_dir: Path) -> list[int]:
    if not public_cache_dir.exists():
        return []
    years: list[int] = []
    for path in public_cache_dir.glob("*.tif"):
        try:
            years.append(int(path.stem))
        except ValueError:
            continue
    return sorted(set(years))


def build_lulc_capability_registry(
    *,
    weights_dir: str | Path | None = None,
    results_path: str | Path | None = None,
    public_cache_dir: str | Path | None = None,
) -> dict:
    weights = Path(weights_dir or settings.WEIGHTS_DIR)
    metrics_path = Path(results_path or (Path(PROJECT_ROOT) / "results" / "linhe_lulc_seg_from_drive.json"))
    cache_dir = Path(public_cache_dir or (Path(PROJECT_ROOT) / "results" / "linhe" / "esri_lulc"))

    metrics = _load_metrics(metrics_path)
    local_models = []
    for spec in LOCAL_MODEL_SPECS:
        checkpoint = weights / spec["checkpoint"]
        validation = metrics.get((spec["method"], spec["seed"]))
        local_models.append(
            {
                **spec,
                "mode": "local_model",
                "ready": checkpoint.exists(),
                "checkpoint_path": str(checkpoint),
                "classes": LULC_CLASS_NAMES,
                "class_schema": "alphaearth_linhe_rgb_6class",
                "input": {
                    "type": "uploaded_rgb_image",
                    "shape": "H x W x 3",
                    "normalization": "uint8 / 255",
                },
                "validation": validation,
                "limitations": [
                    "Validated on Linhe RGB patches with Esri-derived LULC labels.",
                    "Single-scene or out-of-region predictions should be checked with /lulc/evaluate.",
                ],
            }
        )

    years = _available_public_years(cache_dir)
    public_products = [
        {
            "id": "esri_lulc_cache",
            "mode": "public_product",
            "status": "ready" if years else "not_configured",
            "product": "Impact Observatory / Esri / Microsoft 10m annual LULC",
            "available_years": years,
            "query": {
                "endpoint": "/api/ae/inference/lulc/public",
                "parameters": ["provider_id", "year", "minx", "miny", "maxx", "maxy", "bbox_crs"],
            },
            "classes": PUBLIC_LULC_CLASS_NAMES,
            "class_schema": "linhe_esri_6class_public_cache",
            "cache_dir": str(cache_dir),
            "limitations": [
                "Local cache only covers rasters already downloaded into results/linhe/esri_lulc.",
                "This product is 10 m annual LULC, not a trained local AlphaEarth checkpoint.",
            ],
        },
        {
            "id": "dynamic_world_gee",
            "mode": "public_product",
            "status": "requires_earth_engine_auth",
            "product": "Google Dynamic World V1",
            "classes": [
                "water",
                "trees",
                "grass",
                "flooded_vegetation",
                "crops",
                "shrub_and_scrub",
                "built",
                "bare",
                "snow_and_ice",
            ],
            "native_resolution_m": 10,
            "source_url": "https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1",
            "limitations": [
                "Requires Earth Engine credentials and online query support.",
                "Near-real-time Sentinel-2 product with 9-class probabilities.",
            ],
        },
        {
            "id": "esa_worldcover_static",
            "mode": "public_product",
            "status": "not_configured",
            "product": "ESA WorldCover 10m 2020/2021",
            "source_url": "https://esa-worldcover.org/en",
            "limitations": [
                "Useful as a static global baseline once local tiles are configured.",
            ],
        },
    ]

    return {
        "task": "lulc_segmentation",
        "modes": [
            {
                "id": "public_product",
                "name": "Public ready-made LULC product",
                "purpose": "Use existing global or regional land-cover products by area and date.",
            },
            {
                "id": "local_model",
                "name": "Local AlphaEarth checkpoint inference",
                "purpose": "Run uploaded RGB patches through local Linhe-trained segmentation checkpoints.",
            },
        ],
        "default_local_model_id": DEFAULT_LOCAL_MODEL_ID,
        "local_models": local_models,
        "public_products": public_products,
    }
