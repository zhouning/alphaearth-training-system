"""Register the Linhe linear_probe checkpoint as a platform model asset.

Copies the best checkpoint to the weights directory and creates DB records
(AeDataset, AeTrainingJob, AeModel) so the model appears in the frontend
model asset library.

Usage:
  python scripts/linhe_register_model.py

If PostgreSQL is unavailable, the file copy still succeeds and the script
prints the SQL statements for manual execution.
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from shutil import copy2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ae_backend"))

SRC = ROOT / "results" / "linhe" / "linhe_linear_probe_rgb.pt"
DST_DIR = ROOT / "data" / "weights"
MODEL_ID = "linhe-linear-probe-rgb-v1"
JOB_ID = f"linhe-lp-{uuid.uuid4().hex[:8]}"
DATASET_ID = "linhe_patches"
MODEL_NAME = "LinHe-LinearProbe-RGB-v1"
EVAL_SCORE = 69.5
HYPERPARAMS = {"peft_method": "linear_probe", "epochs": 10, "lr": 0.001, "batch_size": 16}
METRICS = {"best_val_acc": 0.695, "weighted_f1": 0.6683, "macro_f1": 0.4287}


def copy_weights() -> Path:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    dst = DST_DIR / f"alphaearth_local_{MODEL_ID}.pt"
    if not SRC.exists():
        raise FileNotFoundError(f"Checkpoint not found: {SRC}")
    copy2(SRC, dst)
    print(f"[ok] copied {SRC.name} -> {dst}")
    return dst


def register_db(weights_key: str) -> None:
    try:
        from app.db.database import SessionLocal
        from app.models.domain import AeDataset, AeTrainingJob, AeModel, JobStatus
    except Exception as e:
        print(f"[warn] cannot import DB modules: {e}")
        print_fallback_sql(weights_key)
        return

    try:
        with SessionLocal() as db:
            if not db.query(AeDataset).filter(AeDataset.id == DATASET_ID).first():
                db.add(AeDataset(
                    id=DATASET_ID,
                    dataset_name="临河样例数据 (1293 patches, 5 classes)",
                    satellite_sources=["GF-1", "GF-6", "ZY-3", "JKF-01"],
                    patch_count=1293,
                ))
                db.flush()
                print(f"[ok] created AeDataset: {DATASET_ID}")
            else:
                print(f"[ok] AeDataset {DATASET_ID} already exists")

            job = AeTrainingJob(
                id=JOB_ID,
                dataset_id=DATASET_ID,
                status=JobStatus.COMPLETED,
                hyperparameters=HYPERPARAMS,
                metrics=METRICS,
                current_epoch=10,
            )
            db.add(job)
            db.flush()
            print(f"[ok] created AeTrainingJob: {JOB_ID}")

            db.query(AeModel).update({"is_active": False})

            model = AeModel(
                id=MODEL_ID,
                job_id=JOB_ID,
                model_name=MODEL_NAME,
                evaluation_score=EVAL_SCORE,
                weights_obs_key=weights_key,
                is_active=True,
            )
            db.add(model)
            db.commit()
            print(f"[ok] created AeModel: {MODEL_NAME} (active)")
    except Exception as e:
        print(f"[warn] DB registration failed: {e}")
        print_fallback_sql(weights_key)


def print_fallback_sql(weights_key: str) -> None:
    print("\n--- Manual SQL (run when DB is available) ---")
    print(f"""
INSERT INTO ae_datasets (id, dataset_name, satellite_sources, patch_count)
VALUES ('{DATASET_ID}', '临河样例数据', '{{"GF-1","GF-6","ZY-3","JKF-01"}}', 1293)
ON CONFLICT (id) DO NOTHING;

INSERT INTO ae_training_jobs (id, dataset_id, status, hyperparameters, metrics, current_epoch)
VALUES ('{JOB_ID}', '{DATASET_ID}', 'COMPLETED',
        '{HYPERPARAMS}', '{METRICS}', 10);

UPDATE ae_models SET is_active = false;

INSERT INTO ae_models (id, job_id, model_name, evaluation_score, weights_obs_key, is_active)
VALUES ('{MODEL_ID}', '{JOB_ID}', '{MODEL_NAME}', {EVAL_SCORE},
        '{weights_key}', true);
""")


def main() -> None:
    dst = copy_weights()
    weights_key = f"alphaearth/models/weights/{dst.name}"
    register_db(weights_key)
    print(f"\n[done] model registered: {MODEL_NAME}")
    print(f"  checkpoint: {dst}")
    print(f"  eval_score: {EVAL_SCORE}")


if __name__ == "__main__":
    main()
