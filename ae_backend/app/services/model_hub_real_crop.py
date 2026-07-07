from __future__ import annotations

from pathlib import Path

from app.core.config import PROJECT_ROOT
from app.services.model_hub_crop_raster import validate_prithvi_crop_raster
from app.services.model_hub_runtime import ModelHubRuntimeError


CROP_MODEL_ID = "prithvi_crop_classification_arcgis_style"


def _default_weights_dir() -> Path:
    return Path(PROJECT_ROOT) / "data" / "weights" / "prithvi_crop"


def _has_weight_files(weights_dir: Path) -> bool:
    if not weights_dir.exists():
        return False
    if weights_dir.is_file():
        return weights_dir.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"}
    return any(
        child.is_file() and child.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"}
        for child in weights_dir.rglob("*")
    )


def _require_crop_runtime(weights_dir: Path) -> None:
    if not _has_weight_files(weights_dir):
        raise ModelHubRuntimeError(
            f"Prithvi crop weights are missing at {weights_dir}. "
            "Use scripts/model_hub/fetch_public_sample.py --asset prithvi_crop --dry-run "
            "to inspect the approved public source before downloading weights."
        )
    try:
        import terratorch  # noqa: F401
    except ImportError as exc:
        raise ModelHubRuntimeError(
            "Prithvi crop real inference requires TerraTorch-compatible runtime dependencies."
        ) from exc


def run_real_crop_inference(*, options: dict) -> dict:
    raster_path = options.get("raster_path")
    if not raster_path:
        raise ModelHubRuntimeError("raster_path is required for real crop inference")

    validation = validate_prithvi_crop_raster(raster_path)
    weights_dir = Path(options.get("weights_dir") or _default_weights_dir())
    _require_crop_runtime(weights_dir)
    raise ModelHubRuntimeError(
        "Prithvi crop neural runtime passed input and asset guards, but the "
        "TerraTorch inference adapter has not completed local verification. "
        f"validation={validation['band_count']} bands, model_id={CROP_MODEL_ID}."
    )
