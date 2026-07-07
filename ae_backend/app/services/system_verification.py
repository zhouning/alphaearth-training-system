from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from app.services import system_capabilities
from app.services.model_hub_registry import VALID_STATUSES, ModelHubRegistry


_STATUS_ORDER = {
    "fail": 3,
    "warning": 2,
    "pass": 1,
    "not_applicable": 0,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _check(
    *,
    capability_id: str,
    check_id: str,
    category: str,
    status: str,
    severity: str,
    title: str,
    detail: str,
    evidence_refs: list[str],
    remediation: str | None = None,
) -> dict[str, Any]:
    return {
        "id": f"{capability_id}:{check_id}",
        "capability_id": capability_id,
        "category": category,
        "status": status,
        "severity": severity,
        "title": title,
        "detail": detail,
        "evidence_refs": evidence_refs,
        "remediation": remediation,
    }


def _aggregate_status(statuses: list[str]) -> str:
    if not statuses:
        return "not_applicable"
    return max(statuses, key=lambda item: _STATUS_ORDER[item])


def _capability_status(capability: dict[str, Any], statuses: list[str]) -> str:
    if any(status == "fail" for status in statuses):
        return "fail"
    if any(status == "warning" for status in statuses):
        return "warning"
    if capability["readiness"] in {"planned", "not_configured"}:
        return "not_applicable"
    return "pass"


def _registry_status_check(capability: dict[str, Any]) -> dict[str, Any]:
    readiness = capability["readiness"]
    if readiness in VALID_STATUSES:
        return _check(
            capability_id=capability["id"],
            check_id="registry_status",
            category="registry_status",
            status="pass",
            severity="info",
            title="Registry status is valid",
            detail=f"Capability status {readiness!r} is registered and recognized.",
            evidence_refs=["model_hub_registry"],
        )
    return _check(
        capability_id=capability["id"],
        check_id="registry_status",
        category="registry_status",
        status="fail",
        severity="error",
        title="Registry status is unknown",
        detail=f"Capability status {readiness!r} is not recognized.",
        evidence_refs=["model_hub_registry"],
        remediation="Use one of ready, demo_only, planned, or not_configured.",
    )


def _runtime_mode_check(capability: dict[str, Any]) -> dict[str, Any]:
    if capability["readiness"] in {"planned", "not_configured"}:
        return _check(
            capability_id=capability["id"],
            check_id="runtime_mode_declared",
            category="runtime_mode",
            status="not_applicable",
            severity="info",
            title="Runtime mode is not required for planned capability",
            detail="Planned capabilities are tracked before executable runtime modes are configured.",
            evidence_refs=["system_capabilities"],
        )
    if capability["runtime_modes"]:
        return _check(
            capability_id=capability["id"],
            check_id="runtime_mode_declared",
            category="runtime_mode",
            status="pass",
            severity="info",
            title="Runtime mode is declared",
            detail=f"Declared runtime modes: {', '.join(capability['runtime_modes'])}.",
            evidence_refs=["model_hub_registry", "system_capabilities"],
        )
    return _check(
        capability_id=capability["id"],
        check_id="runtime_mode_declared",
        category="runtime_mode",
        status="fail",
        severity="error",
        title="Executable capability has no runtime mode",
        detail="Ready and demo capabilities must expose at least one runtime mode.",
        evidence_refs=["model_hub_registry", "system_capabilities"],
        remediation="Add a supported runtime mode or downgrade the capability readiness.",
    )


def _checkpoint_check(capability: dict[str, Any]) -> dict[str, Any]:
    if capability["readiness"] in {"planned", "not_configured"}:
        return _check(
            capability_id=capability["id"],
            check_id="checkpoint_configured",
            category="checkpoint_configuration",
            status="not_applicable",
            severity="info",
            title="Checkpoint is not required for planned capability",
            detail="Missing checkpoints do not fail planned capabilities.",
            evidence_refs=["system_capabilities"],
        )
    if capability["checkpoint"]["configured"]:
        return _check(
            capability_id=capability["id"],
            check_id="checkpoint_configured",
            category="checkpoint_configuration",
            status="pass",
            severity="info",
            title="Checkpoint is configured",
            detail=f"Checkpoint path is configured: {capability['checkpoint']['path']}.",
            evidence_refs=["model_hub_registry"],
        )
    if capability["workflow_level"] in {"demo", "contract_demo"}:
        return _check(
            capability_id=capability["id"],
            check_id="checkpoint_configured",
            category="checkpoint_configuration",
            status="pass",
            severity="info",
            title="Demo workflow does not require a checkpoint",
            detail="The capability is explicitly marked as a demo or contract demo, so missing weights do not block the demo contract.",
            evidence_refs=["model_hub_registry", "system_capabilities"],
        )
    return _check(
        capability_id=capability["id"],
        check_id="checkpoint_configured",
        category="checkpoint_configuration",
        status="fail",
        severity="error",
        title="Ready capability is missing a checkpoint",
        detail="Runnable ready capabilities must declare a checkpoint path.",
        evidence_refs=["model_hub_registry"],
        remediation="Configure a checkpoint path or downgrade the capability readiness.",
    )


def _replacement_boundary_check(capability: dict[str, Any]) -> dict[str, Any]:
    if capability["id"] != "prithvi_crop_classification_arcgis_style":
        return _check(
            capability_id=capability["id"],
            check_id="arcgis_replacement_guard",
            category="replacement_boundary",
            status="not_applicable",
            severity="info",
            title="ArcGIS replacement guard is not applicable",
            detail="This capability is not the ArcGIS-style Prithvi crop workflow.",
            evidence_refs=["system_capabilities"],
        )
    replacement = capability["arcgis_replacement"]
    if replacement["status"] == "not_ready":
        return _check(
            capability_id=capability["id"],
            check_id="arcgis_replacement_guard",
            category="replacement_boundary",
            status="pass",
            severity="info",
            title="ArcGIS replacement guard is conservative",
            detail="The crop capability is marked as demo-only and not a validated ArcGIS replacement.",
            evidence_refs=["model_hub_registry", "system_capabilities"],
            remediation="Attach a validated Prithvi crop checkpoint before claiming replacement readiness.",
        )
    return _check(
        capability_id=capability["id"],
        check_id="arcgis_replacement_guard",
        category="replacement_boundary",
        status="fail",
        severity="error",
        title="ArcGIS replacement guard is overclaiming",
        detail=f"Replacement status {replacement['status']!r} is not allowed without checkpoint-backed validation.",
        evidence_refs=["model_hub_registry", "system_capabilities"],
        remediation="Reset replacement status to not_ready until validation evidence exists.",
    )


def _production_evidence_check(capability: dict[str, Any]) -> dict[str, Any]:
    evidence = capability.get("production_evidence") or {}
    state = str(evidence.get("production_state") or "metadata_missing")
    capability_id = capability["id"]
    if state in {"production_candidate", "verification_required"}:
        return _check(
            capability_id=capability_id,
            check_id="production_evidence",
            category="production_evidence",
            status="pass",
            severity="info",
            title="Production evidence is locally usable",
            detail=f"Production evidence state is {state}.",
            evidence_refs=["system_capabilities"],
        )
    if state in {"download_required", "test_data_required", "training_required"}:
        return _check(
            capability_id=capability_id,
            check_id="production_evidence",
            category="production_evidence",
            status="warning",
            severity="warning",
            title="Production evidence is not complete",
            detail=f"Production evidence state is {state}.",
            evidence_refs=["system_capabilities"],
            remediation="Attach the required weights, test data, or training output before promoting this model.",
        )
    return _check(
        capability_id=capability_id,
        check_id="production_evidence",
        category="production_evidence",
        status="fail",
        severity="error",
        title="Production evidence metadata is missing",
        detail=f"Production evidence state is {state}.",
        evidence_refs=["system_capabilities"],
        remediation="Add this model to ae_backend/app/data/model_hub_assets.json.",
    )

def _evidence_source_checks(evidence_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    for index, source in enumerate(evidence_sources, start=1):
        label = str(source["label"])
        source_id = label.lower().replace(" ", "_").replace(".", "")
        if source.get("available"):
            checks.append(
                _check(
                    capability_id="system",
                    check_id=f"evidence_source_{index}_{source_id}",
                    category="evidence_source",
                    status="pass",
                    severity="info",
                    title=f"{label} evidence file is available",
                    detail=f"Optional evidence file exists at {source['path']}.",
                    evidence_refs=[source["path"]],
                )
            )
        else:
            checks.append(
                _check(
                    capability_id="system",
                    check_id=f"evidence_source_{index}_{source_id}",
                    category="evidence_source",
                    status="warning",
                    severity="warning",
                    title=f"{label} evidence file is missing",
                    detail=f"Missing optional evidence file: {source['path']}.",
                    evidence_refs=[source["path"]],
                    remediation="Regenerate or attach the optional Paper12 evidence artifact when available.",
                )
            )
    return checks


def _capability_verification(
    capability: dict[str, Any],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    statuses = [check["status"] for check in checks]
    overall_status = _capability_status(capability, statuses)
    blocking_issues = [check["detail"] for check in checks if check["status"] == "fail"]
    next_actions = [
        check["remediation"]
        for check in checks
        if check["status"] in {"warning", "fail"} and check.get("remediation")
    ]
    if capability["id"] == "prithvi_crop_classification_arcgis_style":
        action = "Attach a validated Prithvi crop checkpoint before claiming replacement readiness."
        if action not in next_actions:
            next_actions.append(action)
    for action in capability.get("next_steps", []):
        if action not in next_actions and overall_status in {"warning", "fail"}:
            next_actions.append(action)
    return {
        "id": capability["id"],
        "overall_status": overall_status,
        "checks": [check["id"] for check in checks],
        "blocking_issues": blocking_issues,
        "next_actions": next_actions,
    }


def build_system_verification(registry: ModelHubRegistry) -> dict[str, Any]:
    capability_payload = system_capabilities.build_system_capabilities(registry)
    all_checks: list[dict[str, Any]] = []
    capability_summaries = []

    for capability in capability_payload["capabilities"]:
        capability_checks = [
            _registry_status_check(capability),
            _runtime_mode_check(capability),
            _checkpoint_check(capability),
            _replacement_boundary_check(capability),
            _production_evidence_check(capability),
        ]
        all_checks.extend(capability_checks)
        capability_summaries.append(_capability_verification(capability, capability_checks))

    all_checks.extend(_evidence_source_checks(capability_payload.get("evidence_sources", [])))

    summary = dict(Counter(check["status"] for check in all_checks))
    for status in ["pass", "warning", "fail", "not_applicable"]:
        summary.setdefault(status, 0)

    return {
        "system": "AlphaEarth System",
        "generated_at": _utc_now(),
        "overall_status": _aggregate_status([check["status"] for check in all_checks]),
        "summary": {
            "pass": summary["pass"],
            "warning": summary["warning"],
            "fail": summary["fail"],
            "not_applicable": summary["not_applicable"],
        },
        "capabilities": capability_summaries,
        "checks": all_checks,
        "notes": [
            "Verification is deterministic and does not load model weights.",
            "A pass means the declared system contract is internally consistent, not that global production accuracy is proven.",
            "ArcGIS replacement readiness remains conservative until checkpoint-backed validation evidence is attached.",
        ],
    }

