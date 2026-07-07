from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT
from app.services import system_verification
from app.services.model_hub_registry import ModelHubRegistry


PROJECT_ROOT_PATH = Path(PROJECT_ROOT).resolve()
ALLOWED_LOCAL_ROOTS = ("paper12_results", "results", "linhe_results")
ALLOWED_REGISTRY_REFS = {
    "model_hub_registry": "Model Hub registry",
    "system_capabilities": "System capability service",
}
PREVIEW_LIMIT_BYTES = 256 * 1024
JSON_ITEM_LIMIT = 20
JSON_DEPTH_LIMIT = 4
TEXT_LINE_LIMIT = 16
CSV_ROW_LIMIT = 8

JSON_EXTENSIONS = {".json", ".geojson"}
CSV_EXTENSIONS = {".csv"}
TEXT_EXTENSIONS = {".txt", ".md", ".log"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
TENSOR_EXTENSIONS = {".npz", ".pt", ".pth"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _to_posix(ref: str) -> str:
    return ref.strip().replace("\\", "/")


def _kind_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in JSON_EXTENSIONS:
        return "json"
    if suffix in CSV_EXTENSIONS:
        return "csv"
    if suffix == ".md":
        return "markdown"
    if suffix in TEXT_EXTENSIONS:
        return "text"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix == ".pdf":
        return "pdf"
    if suffix in TENSOR_EXTENSIONS:
        return "tensor"
    return "binary"


def _safe_ref_for_output(ref: str) -> str:
    normalized = _to_posix(str(ref))
    if Path(normalized).is_absolute():
        return "<absolute path blocked>"
    return normalized


def _resolve_local_ref(ref: str) -> tuple[Path | None, str | None, str | None]:
    normalized = _to_posix(str(ref))
    if not normalized:
        return None, None, "Evidence ref is empty."
    if ".." in Path(normalized).parts:
        return None, None, "Evidence ref uses parent-directory traversal."

    root = PROJECT_ROOT_PATH.resolve()
    candidate = Path(normalized)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / candidate).resolve()

    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError:
        return None, None, "Evidence ref is outside the project root."

    if not any(relative == allowed or relative.startswith(f"{allowed}/") for allowed in ALLOWED_LOCAL_ROOTS):
        return None, None, "Evidence ref is outside allowed evidence roots."
    return resolved, relative, None


def _compact_json(value: Any, *, depth: int = 0) -> Any:
    if depth >= JSON_DEPTH_LIMIT:
        return "<truncated>"
    if isinstance(value, dict):
        items = list(value.items())
        compact = {
            str(key): _compact_json(item, depth=depth + 1)
            for key, item in items[:JSON_ITEM_LIMIT]
        }
        if len(items) > JSON_ITEM_LIMIT:
            compact["__truncated__"] = True
        return compact
    if isinstance(value, list):
        compact_list = [_compact_json(item, depth=depth + 1) for item in value[:JSON_ITEM_LIMIT]]
        if len(value) > JSON_ITEM_LIMIT:
            compact_list.append({"__truncated__": True})
        return compact_list
    if isinstance(value, str) and len(value) > 240:
        return f"{value[:240]}..."
    return value


def _json_preview(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "type": "json",
        "truncated": False,
        "content": _compact_json(payload),
    }


def _csv_preview(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    rows = list(csv.reader(text.splitlines()))
    header = rows[0] if rows else []
    data_rows = rows[1 : CSV_ROW_LIMIT + 1]
    return {
        "type": "csv",
        "truncated": len(rows) > CSV_ROW_LIMIT + 1,
        "header": header,
        "rows": data_rows,
    }


def _text_preview(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {
        "type": "text",
        "truncated": len(lines) > TEXT_LINE_LIMIT,
        "lines": lines[:TEXT_LINE_LIMIT],
    }


def _preview_for_path(path: Path, kind: str, size_bytes: int) -> tuple[bool, dict[str, Any] | None, str | None]:
    if size_bytes > PREVIEW_LIMIT_BYTES:
        return False, None, "Evidence artifact is available; inline preview skipped because the file is larger than 256 KB."
    try:
        if kind == "json":
            return True, _json_preview(path), None
        if kind == "csv":
            return True, _csv_preview(path), None
        if kind in {"text", "markdown"}:
            return True, _text_preview(path), None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, csv.Error) as exc:
        return False, None, f"Evidence artifact is available, but preview parsing failed: {exc.__class__.__name__}."
    return False, None, None


def _registry_artifact(ref: str) -> dict[str, Any]:
    return {
        "ref": ref,
        "kind": "registry",
        "status": "available",
        "source_id": ref,
        "label": ALLOWED_REGISTRY_REFS[ref],
        "previewable": False,
        "message": "Registry evidence source is recognized.",
    }


def _blocked_artifact(ref: str, message: str) -> dict[str, Any]:
    return {
        "ref": _safe_ref_for_output(ref),
        "kind": "blocked",
        "status": "blocked",
        "previewable": False,
        "message": message,
    }


def _local_artifact(ref: str) -> dict[str, Any]:
    path, safe_path, blocked_message = _resolve_local_ref(ref)
    if blocked_message or path is None or safe_path is None:
        return _blocked_artifact(ref, blocked_message or "Evidence ref cannot be resolved safely.")

    kind = _kind_for_path(path)
    if not path.exists():
        return {
            "ref": _to_posix(str(ref)),
            "kind": kind,
            "status": "missing",
            "safe_path": safe_path,
            "previewable": False,
            "message": "Referenced optional evidence artifact is missing.",
        }

    stat = path.stat()
    previewable, preview, preview_message = _preview_for_path(path, kind, stat.st_size)
    message = preview_message or "Evidence artifact is available."
    payload = {
        "ref": _to_posix(str(ref)),
        "kind": kind,
        "status": "available",
        "safe_path": safe_path,
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).replace(microsecond=0).isoformat(),
        "previewable": previewable,
        "message": message,
    }
    if preview is not None:
        payload["preview"] = preview
    return payload


def _missing_local_artifact(ref: str) -> dict[str, Any]:
    path, safe_path, blocked_message = _resolve_local_ref(ref)
    if blocked_message or path is None or safe_path is None:
        return _blocked_artifact(ref, blocked_message or "Evidence ref cannot be resolved safely.")
    return {
        "ref": _to_posix(str(ref)),
        "kind": _kind_for_path(path),
        "status": "missing",
        "safe_path": safe_path,
        "previewable": False,
        "message": "Referenced optional evidence artifact is missing.",
    }


def _check_reports_missing_optional_evidence(check: dict[str, Any]) -> bool:
    return (
        check.get("category") == "evidence_source"
        and check.get("status") == "warning"
        and "missing optional evidence file" in str(check.get("detail", "")).lower()
    )


def _artifact_for_ref(ref: str, check: dict[str, Any]) -> dict[str, Any]:
    normalized = _to_posix(str(ref))
    if normalized in ALLOWED_REGISTRY_REFS:
        return _registry_artifact(normalized)
    if normalized:
        if _check_reports_missing_optional_evidence(check):
            return _missing_local_artifact(normalized)
        return _local_artifact(normalized)
    return {
        "ref": "",
        "kind": "not_applicable",
        "status": "not_applicable",
        "previewable": False,
        "message": "Evidence ref is empty.",
    }


def _artifact_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    summary_keys = [
        "ref",
        "kind",
        "status",
        "safe_path",
        "source_id",
        "label",
        "size_bytes",
        "modified_at",
        "previewable",
        "message",
        "preview",
    ]
    return {key: artifact[key] for key in summary_keys if key in artifact}


def build_system_evidence(registry: ModelHubRegistry) -> dict[str, Any]:
    verification = system_verification.build_system_verification(registry)
    checks: list[dict[str, Any]] = []
    artifacts_by_ref: dict[str, dict[str, Any]] = {}

    for check in verification.get("checks", []):
        evidence = []
        for ref in check.get("evidence_refs", []):
            artifact = _artifact_for_ref(str(ref), check)
            evidence.append(artifact)
            artifacts_by_ref.setdefault(artifact["ref"], _artifact_summary(artifact))
        checks.append(
            {
                "check_id": check["id"],
                "capability_id": check.get("capability_id"),
                "check_status": check.get("status"),
                "check_title": check.get("title"),
                "remediation": check.get("remediation"),
                "evidence": evidence,
            }
        )

    artifacts = list(artifacts_by_ref.values())
    status_counts = Counter(artifact["status"] for artifact in artifacts)

    return {
        "system": "AlphaEarth System",
        "generated_at": _utc_now(),
        "summary": {
            "available": status_counts.get("available", 0),
            "missing": status_counts.get("missing", 0),
            "previewable": sum(1 for artifact in artifacts if artifact.get("previewable")),
            "blocked": status_counts.get("blocked", 0),
        },
        "checks": checks,
        "artifacts": artifacts,
        "notes": [
            "Evidence drill-down reads local metadata and small previews only.",
            "It does not execute model inference, train models, or regenerate results.",
            "ArcGIS replacement readiness remains conservative until checkpoint-backed validation evidence is attached.",
        ],
    }