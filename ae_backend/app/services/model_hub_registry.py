from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "model_id",
    "display_name",
    "task_type",
    "backbone",
    "adapter",
    "checkpoint_path",
    "input_spec",
    "output_spec",
    "class_schema",
    "metrics",
    "trained_region",
    "supported_sensors",
    "license",
    "status",
    "example_inputs",
}

VALID_STATUSES = {"ready", "demo_only", "planned", "not_configured"}


class RegistryValidationError(ValueError):
    """Raised when model hub registry data is malformed."""


def _require_string(record: dict[str, Any], field: str) -> str:
    value = record[field]
    if not isinstance(value, str):
        raise RegistryValidationError(f"{field} must be a string")
    return value


def _require_optional_string(record: dict[str, Any], field: str) -> str | None:
    value = record[field]
    if value is not None and not isinstance(value, str):
        raise RegistryValidationError(f"{field} must be a string or null")
    return value


def _require_dict(record: dict[str, Any], field: str) -> dict[str, Any]:
    value = record[field]
    if not isinstance(value, dict):
        raise RegistryValidationError(f"{field} must be an object")
    return dict(value)


def _require_string_list(record: dict[str, Any], field: str) -> list[str]:
    value = record[field]
    if not isinstance(value, list):
        raise RegistryValidationError(f"{field} must be a list")
    if not all(isinstance(item, str) for item in value):
        raise RegistryValidationError(f"{field} values must be strings")
    return list(value)


@dataclass(frozen=True)
class ModelHubEntry:
    model_id: str
    display_name: str
    task_type: str
    backbone: str
    adapter: str
    checkpoint_path: str | None
    input_spec: dict[str, Any]
    output_spec: dict[str, Any]
    class_schema: list[str]
    metrics: dict[str, Any]
    trained_region: str
    supported_sensors: list[str]
    license: str
    status: str
    example_inputs: list[str]
    extra_fields: dict[str, Any]

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "ModelHubEntry":
        if not isinstance(record, dict):
            raise RegistryValidationError("Model registry record must be a JSON object")

        missing_fields = sorted(REQUIRED_FIELDS - set(record))
        if missing_fields:
            raise RegistryValidationError(f"Missing required field(s): {', '.join(missing_fields)}")

        model_id = _require_string(record, "model_id")
        display_name = _require_string(record, "display_name")
        task_type = _require_string(record, "task_type")
        backbone = _require_string(record, "backbone")
        adapter = _require_string(record, "adapter")
        checkpoint_path = _require_optional_string(record, "checkpoint_path")
        input_spec = _require_dict(record, "input_spec")
        output_spec = _require_dict(record, "output_spec")
        class_schema = _require_string_list(record, "class_schema")
        metrics = _require_dict(record, "metrics")
        trained_region = _require_string(record, "trained_region")
        supported_sensors = _require_string_list(record, "supported_sensors")
        license = _require_string(record, "license")
        status = _require_string(record, "status")
        example_inputs = _require_string_list(record, "example_inputs")

        if status not in VALID_STATUSES:
            raise RegistryValidationError(f"Invalid status for model_id {model_id!r}: {status}")

        extra_fields = {
            key: deepcopy(value)
            for key, value in record.items()
            if key not in REQUIRED_FIELDS
        }

        return cls(
            model_id=model_id,
            display_name=display_name,
            task_type=task_type,
            backbone=backbone,
            adapter=adapter,
            checkpoint_path=checkpoint_path,
            input_spec=input_spec,
            output_spec=output_spec,
            class_schema=class_schema,
            metrics=metrics,
            trained_region=trained_region,
            supported_sensors=supported_sensors,
            license=license,
            status=status,
            example_inputs=example_inputs,
            extra_fields=extra_fields,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "task_type": self.task_type,
            "backbone": self.backbone,
            "adapter": self.adapter,
            "checkpoint_path": self.checkpoint_path,
            "input_spec": deepcopy(self.input_spec),
            "output_spec": deepcopy(self.output_spec),
            "class_schema": list(self.class_schema),
            "metrics": deepcopy(self.metrics),
            "trained_region": self.trained_region,
            "supported_sensors": list(self.supported_sensors),
            "license": self.license,
            "status": self.status,
            "example_inputs": list(self.example_inputs),
        }
        payload.update(deepcopy(self.extra_fields))
        return payload


class ModelHubRegistry:
    def __init__(self, models: list[ModelHubEntry]):
        self.models = list(models)
        self._by_id: dict[str, ModelHubEntry] = {}
        for model in self.models:
            if model.model_id in self._by_id:
                raise RegistryValidationError(f"Duplicate model_id: {model.model_id}")
            self._by_id[model.model_id] = model

    def get_model(self, model_id: str) -> ModelHubEntry:
        return self._by_id[model_id]

    def to_public_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {"models": [model.to_dict() for model in self.models]}


def load_model_registry(path: str | Path) -> ModelHubRegistry:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryValidationError(f"Invalid JSON in model registry: {exc.msg}") from exc

    if not isinstance(data, list):
        raise RegistryValidationError("Model registry must be a JSON list")

    entries = [ModelHubEntry.from_record(record) for record in data]
    return ModelHubRegistry(entries)
