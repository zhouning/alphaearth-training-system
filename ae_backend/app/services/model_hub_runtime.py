from __future__ import annotations

import numpy as np


class ModelHubRuntimeError(ValueError):
    """Raised when a model-hub job cannot be executed."""


def _run_lulc_demo_patch(model_id: str, options: dict) -> dict:
    from app.api.inference import get_lulc_service

    service = get_lulc_service(model_id=model_id)
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    prediction = service.predict_image(image)
    return {
        "result": {
            "task": "lulc_segmentation",
            "model_id": prediction.get("model_id", model_id),
            "summary": {
                "class_pixel_counts": prediction.get("class_pixel_counts", {}),
                "class_area_fraction": prediction.get("class_area_fraction", {}),
                "mask_shape": prediction.get("mask_shape"),
            },
        },
        "artifacts": [{"kind": "json", "path": "inline"}],
        "logs": ["ran LULC demo patch runtime"],
    }


def run_model_hub_job(*, model_id: str, input_mode: str, options: dict) -> dict:
    if model_id == "lulc_6class_prithvi_houlsby" and input_mode == "demo_patch":
        return _run_lulc_demo_patch(model_id, options)
    if model_id == "semantic_change_prithvi" and input_mode == "cached_demo":
        from app.services.model_hub_change import summarize_cached_linhe_change

        return summarize_cached_linhe_change(options=options)
    if model_id == "prithvi_crop_classification_arcgis_style" and input_mode == "cached_demo":
        from app.services.model_hub_crop import summarize_cached_crop_demo

        return summarize_cached_crop_demo(options=options)
    raise ModelHubRuntimeError(
        f"Unsupported model-hub job: model_id={model_id}, input_mode={input_mode}"
    )
