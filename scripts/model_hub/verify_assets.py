from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ae_backend"))

from app.api.model_hub import get_model_registry  # noqa: E402
from app.services.model_hub_evidence import build_model_hub_evidence  # noqa: E402


def _table(evidence: dict) -> str:
    rows = [
        "model_id | runtime_kind | production_state | weights | test_data",
        "--- | --- | --- | --- | ---",
    ]
    for model in evidence["models"]:
        rows.append(
            " | ".join(
                [
                    model["model_id"],
                    model["runtime_kind"],
                    model["production_state"],
                    "yes" if model["weights"]["presence"]["available"] else "no",
                    "yes" if model["test_data"]["presence"]["available"] else "no",
                ]
            )
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify local Model Hub assets.")
    parser.add_argument("--json", action="store_true", help="Print full evidence as JSON.")
    args = parser.parse_args(argv)

    evidence = build_model_hub_evidence(get_model_registry())
    if args.json:
        print(json.dumps(evidence, indent=2, ensure_ascii=False))
    else:
        print(_table(evidence))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
