from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.core.config import PROJECT_ROOT
from app.services.model_hub_registry import ModelHubRegistry, load_model_registry


router = APIRouter()

DEFAULT_REGISTRY_PATH = (
    Path(PROJECT_ROOT) / "ae_backend" / "app" / "data" / "model_hub_models.json"
)


@lru_cache(maxsize=1)
def get_model_registry() -> ModelHubRegistry:
    return load_model_registry(DEFAULT_REGISTRY_PATH)


@router.get("/models")
def list_models():
    return get_model_registry().to_public_dict()


@router.get("/models/{model_id}")
def get_model(model_id: str):
    try:
        return get_model_registry().get_model(model_id).to_dict()
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model_id: {model_id}",
        ) from exc
