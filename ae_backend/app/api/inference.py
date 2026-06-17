from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from app.core.config import PROJECT_ROOT
from app.core.config import settings
from app.services.inference import (
    CheckpointCompatibilityError,
    LULCInferenceService,
    compute_segmentation_metrics,
)
from app.services.lulc_public import PublicLULCNotAvailableError
from app.services.lulc_public import query_public_lulc as _query_public_lulc
from app.services.lulc_registry import build_lulc_capability_registry


router = APIRouter()
DEFAULT_LULC_CHECKPOINT = "linhe_lulc/geoadapter__rgb_3band__seed123.pt"
DEFAULT_PUBLIC_LULC_CACHE = Path(PROJECT_ROOT) / "results" / "linhe" / "esri_lulc"


def _resolve_checkpoint_path(checkpoint_path: str | None) -> Path:
    if not checkpoint_path:
        checkpoint_path = DEFAULT_LULC_CHECKPOINT

    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = Path(settings.WEIGHTS_DIR) / path
    return path


@lru_cache(maxsize=4)
def _cached_lulc_service(
    checkpoint_path: str,
    model_id: str | None,
) -> LULCInferenceService:
    resolved_checkpoint = Path(checkpoint_path)
    prithvi_checkpoint = Path(settings.WEIGHTS_DIR) / "prithvi" / "Prithvi_100M.pt"
    return LULCInferenceService.from_checkpoint(
        checkpoint_path=resolved_checkpoint,
        prithvi_checkpoint_path=prithvi_checkpoint if prithvi_checkpoint.exists() else None,
        model_id=model_id,
    )


def get_lulc_service(
    *,
    checkpoint_path: str | None = None,
    model_id: str | None = None,
) -> LULCInferenceService:
    resolved_checkpoint = _resolve_checkpoint_path(checkpoint_path)
    return _cached_lulc_service(str(resolved_checkpoint), model_id)


async def _read_rgb_image(file: UploadFile) -> np.ndarray:
    content = await file.read()
    try:
        with Image.open(io.BytesIO(content)) as image:
            return np.asarray(image.convert("RGB"))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.") from exc


async def _read_label_mask(file: UploadFile) -> np.ndarray:
    content = await file.read()
    try:
        with Image.open(io.BytesIO(content)) as image:
            return np.asarray(image)
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded label file is not a readable image.") from exc


def query_public_lulc(
    *,
    provider_id: str,
    year: int,
    bbox: tuple[float, float, float, float],
    bbox_crs: str,
) -> dict:
    return _query_public_lulc(
        provider_id=provider_id,
        year=year,
        bbox=bbox,
        bbox_crs=bbox_crs,
        cache_dir=DEFAULT_PUBLIC_LULC_CACHE,
    )


@router.get("/lulc/modes")
def get_lulc_modes():
    return build_lulc_capability_registry()


@router.get("/lulc/public")
def infer_public_lulc(
    provider_id: str = Query(default="esri_lulc_cache"),
    year: int = Query(..., ge=1900, le=2100),
    minx: float = Query(...),
    miny: float = Query(...),
    maxx: float = Query(...),
    maxy: float = Query(...),
    bbox_crs: str = Query(default="EPSG:4326"),
):
    if minx >= maxx or miny >= maxy:
        raise HTTPException(status_code=400, detail="Invalid bbox: min values must be below max values.")
    try:
        return query_public_lulc(
            provider_id=provider_id,
            year=year,
            bbox=(minx, miny, maxx, maxy),
            bbox_crs=bbox_crs,
        )
    except PublicLULCNotAvailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/lulc")
async def infer_lulc(
    file: UploadFile = File(...),
    checkpoint_path: str | None = Form(default=None),
    model_id: str | None = Form(default=None),
):
    image = await _read_rgb_image(file)
    try:
        service = get_lulc_service(checkpoint_path=checkpoint_path, model_id=model_id)
        return service.predict_image(image)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CheckpointCompatibilityError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/lulc/evaluate")
async def evaluate_lulc(
    file: UploadFile = File(...),
    label_file: UploadFile = File(...),
    checkpoint_path: str | None = Form(default=None),
    model_id: str | None = Form(default=None),
    ignore_index: int | None = Form(default=None),
):
    image = await _read_rgb_image(file)
    label = await _read_label_mask(label_file)
    if label.ndim == 3:
        label = label[:, :, 0]
    try:
        service = get_lulc_service(checkpoint_path=checkpoint_path, model_id=model_id)
        prediction = service.predict_image(image)
        predicted_mask = np.asarray(prediction["mask"], dtype=np.int64)
        metrics = compute_segmentation_metrics(
            predicted_mask,
            label.astype(np.int64),
            class_names=prediction.get("classes"),
            ignore_index=ignore_index,
        )
        return {
            "task": "lulc_segmentation_evaluation",
            "prediction": prediction,
            "evaluation": metrics,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CheckpointCompatibilityError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
