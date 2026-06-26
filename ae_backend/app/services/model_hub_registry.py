from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "model_id",
    "display_name",
    "task_type",
    "backbone",
    "adapter",
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

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "ModelHubEntry":
        if not isinstance(record, dict):
            raise RegistryValidationError("Model registry record must be a JSON object")

        missing_fields = sorted(REQUIRED_FIELDS - set(record))
        if missing_fields:
            raise RegistryValidationError(f"Missing required field(s): {', '.join(missing_fields)}")

        status = str(record["status"])
        if status not in VALID_STATUSES:
            raise RegistryValidationError(f"Invalid status for model_id {record['model_id']!r}: {status}")

        try:
            input_spec = dict(record["input_spec"])
            output_spec = dict(record["output_spec"])
            class_schema = list(record["class_schema"])
            metrics = dict(record["metrics"])
            supported_sensors = list(record["supported_sensors"])
            example_inputs = list(record["example_inputs"])
        except (TypeError, ValueError) as exc:
            raise RegistryValidationError(f"Invalid container field for model_id {record['model_id']!r}") from exc

        checkpoint_path = record.get("checkpoint_path")
        return cls(
            model_id=str(record["model_id"]),
            display_name=str(record["display_name"]),
            task_type=str(record["task_type"]),
            backbone=str(record["backbone"]),
            adapter=str(record["adapter"]),
            checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
            input_spec=input_spec,
            output_spec=output_spec,
            class_schema=[str(value) for value in class_schema],
            metrics=metrics,
            trained_region=str(record["trained_region"]),
            supported_sensors=[str(value) for value in supported_sensors],
            license=str(record["license"]),
            status=status,
            example_inputs=[str(value) for value in example_inputs],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "task_type": self.task_type,
            "backbone": self.backbone,
            "adapter": self.adapter,
            "checkpoint_path": self.checkpoint_path,
            "input_spec": dict(self.input_spec),
            "output_spec": dict(self.output_spec),
            "class_schema": list(self.class_schema),
            "metrics": dict(self.metrics),
            "trained_region": self.trained_region,
            "supported_sensors": list(self.supported_sensors),
            "license": self.license,
            "status": self.status,
            "example_inputs": list(self.example_inputs),
        }


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
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RegistryValidationError("Model registry must be a JSON list")

    entries: list[ModelHubEntry] = []
    seen_model_ids: set[str] = set()
    for record in data:
        entry = ModelHubEntry.from_record(record)
        if entry.model_id in seen_model_ids:
            raise RegistryValidationError(f"Duplicate model_id: {entry.model_id}")
        seen_model_ids.add(entry.model_id)
        entries.append(entry)

    return ModelHubRegistry(entries)
