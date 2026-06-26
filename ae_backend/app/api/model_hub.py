from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import PROJECT_ROOT
from app.services.model_hub_jobs import ModelHubJobStore
from app.services.model_hub_registry import ModelHubRegistry, load_model_registry


router = APIRouter()

DEFAULT_REGISTRY_PATH = (
    Path(PROJECT_ROOT) / "ae_backend" / "app" / "data" / "model_hub_models.json"
)
JOB_STORE = ModelHubJobStore()


class ModelHubJobRequest(BaseModel):
    model_id: str
    input_mode: str = Field(default="cached_demo")
    options: dict = Field(default_factory=dict)


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


@router.post("/jobs")
def create_job(request: ModelHubJobRequest):
    try:
        get_model_registry().get_model(request.model_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model_id: {request.model_id}",
        ) from exc
    return JOB_STORE.create_job(
        model_id=request.model_id,
        input_mode=request.input_mode,
        options=request.options,
    )


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return JOB_STORE.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown job_id: {job_id}",
        ) from exc
