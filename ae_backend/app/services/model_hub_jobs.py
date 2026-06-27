from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelHubJobStore:
    def __init__(self):
        self._jobs: dict[str, dict] = {}

    def create_job(self, model_id: str, input_mode: str, options: dict) -> dict:
        now = _utc_now()
        job_id = uuid4().hex
        job = {
            "job_id": job_id,
            "model_id": model_id,
            "input_mode": input_mode,
            "options": deepcopy(options),
            "status": "pending",
            "created_at": now,
            "updated_at": now,
            "result": None,
            "artifacts": [],
            "logs": [],
            "error": None,
        }
        self._jobs[job_id] = job
        return deepcopy(job)

    def get_job(self, job_id: str) -> dict:
        if job_id not in self._jobs:
            raise KeyError(job_id)
        return deepcopy(self._jobs[job_id])

    def mark_running(self, job_id: str, log: str | None = None) -> None:
        job = self._jobs[job_id]
        job["status"] = "running"
        job["updated_at"] = _utc_now()
        if log:
            job["logs"].append(log)

    def mark_succeeded(
        self,
        job_id: str,
        result: dict,
        artifacts: list[dict],
        log: str | None = None,
        logs: list[str] | None = None,
    ) -> None:
        job = self._jobs[job_id]
        job["status"] = "succeeded"
        job["result"] = deepcopy(result)
        job["artifacts"] = deepcopy(artifacts)
        job["updated_at"] = _utc_now()
        if log:
            job["logs"].append(log)
        if logs:
            job["logs"].extend(str(item) for item in logs)

    def mark_failed(self, job_id: str, error: str) -> None:
        job = self._jobs[job_id]
        job["status"] = "failed"
        job["error"] = error
        job["updated_at"] = _utc_now()
        job["logs"].append(error)
