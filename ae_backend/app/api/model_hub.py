from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import PROJECT_ROOT
from app.services.model_hub_jobs import ModelHubJobStore
from app.services.model_hub_registry import ModelHubRegistry, load_model_registry
from app.services.model_hub_runtime import ModelHubRuntimeError, run_model_hub_job


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


def _runtime_modes_for_model(model: dict) -> set[str]:
    runtime_modes = set(model.get("package_profile", {}).get("runtime_modes", []))
    default_mode = model.get("input_spec", {}).get("default_demo_input_mode")
    if default_mode:
        runtime_modes.add(default_mode)
    if model.get("status") == "demo_only":
        runtime_modes.add("cached_demo")
    if model.get("model_id") == "lulc_6class_prithvi_houlsby":
        runtime_modes.add("demo_patch")
    return runtime_modes


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
        model_entry = get_model_registry().get_model(request.model_id).to_dict()
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model_id: {request.model_id}",
        ) from exc

    job = JOB_STORE.create_job(
        model_id=request.model_id,
        input_mode=request.input_mode,
        options=request.options,
    )
    should_execute_now = request.input_mode in _runtime_modes_for_model(model_entry)
    if not should_execute_now:
        return JOB_STORE.get_job(job["job_id"])

    try:
        JOB_STORE.mark_running(job["job_id"], log="job accepted")
        runtime_result = run_model_hub_job(
            model_id=request.model_id,
            input_mode=request.input_mode,
            options=request.options,
        )
        JOB_STORE.mark_succeeded(
            job["job_id"],
            result=runtime_result["result"],
            artifacts=runtime_result["artifacts"],
            logs=runtime_result.get("logs", []),
        )
    except ModelHubRuntimeError as exc:
        JOB_STORE.mark_failed(job["job_id"], error=str(exc))
    except Exception as exc:
        JOB_STORE.mark_failed(job["job_id"], error=str(exc))
    return JOB_STORE.get_job(job["job_id"])


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return JOB_STORE.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown job_id: {job_id}",
        ) from exc
