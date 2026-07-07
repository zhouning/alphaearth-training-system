from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ae_backend"))

from app.services.model_asset_registry import load_model_asset_registry  # noqa: E402


ASSET_ALIASES = {
    "prithvi_crop": "prithvi_crop_classification_arcgis_style",
    "prithvi_flood": "water_flood_prithvi",
    "lulc": "lulc_6class_prithvi_houlsby",
    "building": "building_extraction_prithvi",
    "road": "road_hardscape_prithvi",
    "change": "semantic_change_prithvi",
}


def _asset_record(asset: str) -> dict:
    model_id = ASSET_ALIASES.get(asset, asset)
    registry = load_model_asset_registry()
    for record in registry["models"]:
        if record["model_id"] == model_id:
            return record
    raise SystemExit(f"Unknown asset: {asset}")


def _target_roots(record: dict) -> dict:
    weights = record.get("weights", {})
    test_data = record.get("test_data", {})
    return {
        "weights": weights.get("local_paths", []),
        "test_data": test_data.get("local_paths", []),
    }


def _describe_plan(record: dict, *, dry_run: bool) -> dict:
    weights = record.get("weights", {})
    test_data = record.get("test_data", {})
    return {
        "mode": "dry-run" if dry_run else "execute",
        "model_id": record["model_id"],
        "runtime_kind": record["runtime_kind"],
        "weights": {
            "source": weights.get("source"),
            "repo_id": weights.get("repo_id"),
            "license": weights.get("license"),
            "local_paths": weights.get("local_paths", []),
        },
        "test_data": {
            "dataset_id": test_data.get("dataset_id"),
            "source_url": test_data.get("source_url"),
            "license": test_data.get("license"),
            "input_profile": test_data.get("input_profile"),
            "local_paths": test_data.get("local_paths", []),
        },
        "targets": _target_roots(record),
    }


def _execute_huggingface_snapshot(record: dict, *, include_weights: bool, include_test_data: bool) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for --execute downloads. "
            "Install it explicitly before running network downloads."
        ) from exc

    weights = record.get("weights", {})
    test_data = record.get("test_data", {})
    if include_weights and weights.get("source") == "huggingface" and weights.get("repo_id"):
        targets = weights.get("local_paths") or []
        if not targets:
            raise SystemExit("No local weight target configured.")
        snapshot_download(repo_id=weights["repo_id"], local_dir=REPO_ROOT / targets[0])
    if include_test_data and test_data.get("source_url", "").startswith("https://huggingface.co/datasets/"):
        dataset_repo = test_data["source_url"].split("/datasets/", 1)[1]
        targets = test_data.get("local_paths") or []
        if not targets:
            raise SystemExit("No local test-data target configured.")
        snapshot_download(repo_id=dataset_repo, repo_type="dataset", local_dir=REPO_ROOT / targets[0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch or inspect Model Hub public assets.")
    parser.add_argument("--asset", required=True, help="Asset alias or model_id.")
    parser.add_argument("--dry-run", action="store_true", help="Print sources and targets only.")
    parser.add_argument("--execute", action="store_true", help="Run supported downloads.")
    parser.add_argument("--include-weights", action="store_true", help="Download model weights when supported.")
    parser.add_argument("--include-test-data", action="store_true", help="Download test data when supported.")
    args = parser.parse_args(argv)

    if args.dry_run and args.execute:
        parser.error("--dry-run and --execute are mutually exclusive")
    if not args.dry_run and not args.execute:
        args.dry_run = True

    record = _asset_record(args.asset)
    plan = _describe_plan(record, dry_run=args.dry_run)
    print(json.dumps(plan, indent=2, ensure_ascii=False))

    if args.dry_run:
        print("dry-run: no network request was made.")
        return 0

    if not args.include_weights and not args.include_test_data:
        raise SystemExit("--execute requires --include-weights and/or --include-test-data")
    _execute_huggingface_snapshot(
        record,
        include_weights=args.include_weights,
        include_test_data=args.include_test_data,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
