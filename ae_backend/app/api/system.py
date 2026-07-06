from __future__ import annotations

from fastapi import APIRouter

from app.api.model_hub import get_model_registry
from app.services.system_capabilities import build_system_capabilities


router = APIRouter()


@router.get("/capabilities")
def get_system_capabilities():
    return build_system_capabilities(get_model_registry())
