"""Export Paper12 checkpoint predictions for an ArcGIS validation packet.

This script writes checkpoint-backed model masks into `paper12_masks/`. It does
not create manual labels, alter ArcGIS masks, or evaluate replacement status.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np


PREDICTION_EXPORT_SCHEMA = "paper12.arcgis_replacement_packet_prediction_export.v1"
ANNOTATION_MANIFEST_NAME = "arcgis_replacement_annotation_manifest.csv"
SUMMARY_NAME = "paper12_prediction_export_summary.json"


class LULCPredictor(Protocol):
    model_id: str
    device: str

    def predict_image(self, image: np.ndarray) -> dict[str, Any]:
        ...


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _resolve(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else base_dir / path


def _first_nonempty(*values: str | None) -> str:
    for value in values:
        if value and value.strip():
            return value.strip()
    return ""


def _relative_to_packet(path: Path, packet_dir: Path) -> str:
    try:
        return path.relative_to(packet_dir).as_posix()
    except ValueError:
        return str(path)


def _load_rgb(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    elif path.suffix.lower() == ".npz":
        payload = np.load(path)
        key = "rgb" if "rgb" in payload.files else sorted(payload.files)[0]
        array = payload[key]
    else:
        raise ValueError(f"Unsupported RGB source format: {path}")

    rgb = np.asarray(array)
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB array in CHW or HWC layout, got {rgb.shape}")
    return rgb.astype(np.uint8)


def _load_mask_shape(path: Path) -> tuple[int, ...]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return tuple(np.load(path).shape)
    if suffix == ".npz":
        payload = np.load(path)
        key = "mask" if "mask" in payload.files else sorted(payload.files)[0]
        return tuple(payload[key].shape)
    raise ValueError(f"Unsupported mask format for shape check: {path}")


def _default_service_factory(
    *,
    checkpoint_path: str | Path,
    prithvi_checkpoint_path: str | Path | None,
    model_id: str | None,
    device: str | None,
) -> Callable[[], LULCPredictor]:
    repo_root = Path(__file__).resolve().parents[1]
    backend_root = repo_root / "ae_backend"
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

    from app.services.inference import LULCInferenceService

    def factory() -> LULCPredictor:
        return LULCInferenceService.from_checkpoint(
            checkpoint_path=checkpoint_path,
            prithvi_checkpoint_path=prithvi_checkpoint_path,
            model_id=model_id,
            device=device,
        )

    return factory


def export_packet_predictions(
    *,
    packet_dir: str | Path,
    checkpoint_path: str | Path,
    prithvi_checkpoint_path: str | Path | None = None,
    model_id: str | None = None,
    device: str | None = None,
    overwrite: bool = False,
    service_factory: Callable[[], LULCPredictor] | None = None,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run a Paper12 LULC predictor on packet RGB chips and save masks."""
    packet_dir = Path(packet_dir)
    annotation_manifest_path = packet_dir / ANNOTATION_MANIFEST_NAME
    rows = _read_manifest(annotation_manifest_path)
    output_dir = packet_dir / "paper12_masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(summary_path) if summary_path is not None else packet_dir / SUMMARY_NAME

    factory = service_factory or _default_service_factory(
        checkpoint_path=checkpoint_path,
        prithvi_checkpoint_path=prithvi_checkpoint_path,
        model_id=model_id,
        device=device,
    )
    service: LULCPredictor | None = None
    outputs: list[dict[str, Any]] = []
    failed_samples: list[dict[str, Any]] = []
    skipped_existing_count = 0

    for row in rows:
        sample_id = row.get("sample_id", "").strip()
        output_value = _first_nonempty(
            row.get("paper12_mask_path"),
            row.get("paper12_mask_target_path"),
            f"paper12_masks/{sample_id}.npy",
        )
        output_path = _resolve(packet_dir, output_value)
        if output_path.exists() and not overwrite:
            skipped_existing_count += 1
            continue

        rgb_value = _first_nonempty(row.get("source_rgb_path"), f"rgb/{sample_id}.npy")
        arcgis_value = _first_nonempty(row.get("arcgis_mask_path"), f"arcgis_masks/{sample_id}.npy")
        rgb_path = _resolve(packet_dir, rgb_value)
        arcgis_path = _resolve(packet_dir, arcgis_value)

        try:
            rgb = _load_rgb(rgb_path)
            expected_shape = _load_mask_shape(arcgis_path)
        except (OSError, ValueError) as exc:
            failed_samples.append(
                {"error": str(exc), "sample_id": sample_id}
            )
            continue

        if service is None:
            service = factory()
        prediction = service.predict_image(rgb)
        mask = np.asarray(prediction["mask"], dtype=np.uint8)
        if tuple(mask.shape) != expected_shape:
            failed_samples.append(
                {
                    "error": "prediction_shape_mismatch",
                    "expected_shape": list(expected_shape),
                    "predicted_shape": list(mask.shape),
                    "sample_id": sample_id,
                }
            )
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, mask)
        outputs.append(
            {
                "confidence_summary": prediction.get("confidence_summary", {}),
                "mask_path": _relative_to_packet(output_path, packet_dir),
                "sample_id": sample_id,
            }
        )

    summary: dict[str, Any] = {
        "schema": PREDICTION_EXPORT_SCHEMA,
        "packet_dir": str(packet_dir),
        "annotation_manifest_path": str(annotation_manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "prithvi_checkpoint_path": None if prithvi_checkpoint_path is None else str(prithvi_checkpoint_path),
        "model_id": model_id if model_id is not None else getattr(service, "model_id", None),
        "device": device if device is not None else getattr(service, "device", None),
        "output_dir": str(output_dir),
        "sample_count": len(rows),
        "exported_sample_count": len(outputs),
        "skipped_existing_count": skipped_existing_count,
        "failed_samples": failed_samples,
        "manual_ground_truth_created": False,
        "arcgis_reference_modified": False,
        "outputs": outputs,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export Paper12 checkpoint predictions for a validation packet."
    )
    parser.add_argument("--packet-dir", required=True, help="Validation packet directory.")
    parser.add_argument("--checkpoint", required=True, help="Paper12 segmentation checkpoint.")
    parser.add_argument("--prithvi-checkpoint", help="Optional Prithvi backbone checkpoint.")
    parser.add_argument("--model-id", help="Model identifier to record in summary.")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Runtime device.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing paper12 masks.")
    parser.add_argument("--output-summary", help="Optional summary JSON path.")
    args = parser.parse_args(argv)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 2
    prithvi_checkpoint = Path(args.prithvi_checkpoint) if args.prithvi_checkpoint else None
    if prithvi_checkpoint is not None and not prithvi_checkpoint.exists():
        print(f"Prithvi checkpoint not found: {prithvi_checkpoint}", file=sys.stderr)
        return 2

    try:
        summary = export_packet_predictions(
            packet_dir=args.packet_dir,
            checkpoint_path=checkpoint_path,
            prithvi_checkpoint_path=prithvi_checkpoint,
            model_id=args.model_id,
            device=args.device,
            overwrite=args.overwrite,
            summary_path=args.output_summary,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
