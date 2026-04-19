from pathlib import Path
from datetime import datetime
from typing import Literal
import base64
from io import BytesIO

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from PIL import Image

from app.core.config import settings


LABEL_COLUMNS = ["dataset_id", "patch_path", "label", "reviewer", "labeled_at"]

router = APIRouter()


def load_patch_index(dataset_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(dataset_dir / "_index.parquet")


def load_labels(dataset_dir: Path) -> pd.DataFrame:
    labels_path = dataset_dir / "_labels.parquet"
    if not labels_path.exists():
        return pd.DataFrame(columns=LABEL_COLUMNS)
    return pd.read_parquet(labels_path)


def get_dataset_dir(dataset_id: str) -> Path:
    if dataset_id != "linhe_patches":
        raise HTTPException(status_code=400, detail="Only linhe_patches is supported in v1")
    return Path(settings.DATA_DIR) / dataset_id


@router.get("/next")
def get_next_patch(
    dataset_id: str = Query(...),
    mode: Literal["unlabeled", "labeled"] = Query("unlabeled"),
    label: str | None = Query(None),
):
    dataset_dir = get_dataset_dir(dataset_id)

    # Review mode: iterate already-labeled items, optionally filtered by label.
    if mode == "labeled":
        labels_df = load_labels(dataset_dir)
        if label:
            labels_df = labels_df[labels_df["label"] == label]
        if labels_df.empty:
            return {"status": "success", "data": None}

        row = labels_df.iloc[0]
        return {
            "status": "success",
            "data": {
                "dataset_id": dataset_id,
                "patch_path": row["patch_path"],
                "current_label": row.get("label"),
            },
        }

    # Default mode: serve first unlabeled patch.
    index_df = load_patch_index(dataset_dir)
    labels_df = load_labels(dataset_dir)
    labeled_paths = set(labels_df["patch_path"].tolist())
    unlabeled = index_df[~index_df["patch_path"].isin(labeled_paths)]
    if unlabeled.empty:
        return {"status": "success", "data": None}
    row = unlabeled.iloc[0]
    return {
        "status": "success",
        "data": {
            "dataset_id": dataset_id,
            "patch_path": row["patch_path"],
            "current_label": None,
        },
    }


@router.get("/stats")
def get_label_stats(dataset_id: str = Query(...)):
    dataset_dir = get_dataset_dir(dataset_id)
    index_df = load_patch_index(dataset_dir)
    labels_df = load_labels(dataset_dir)
    return {
        "status": "success",
        "data": {
            "dataset_id": dataset_id,
            "total": int(len(index_df)),
            "labeled": int(len(labels_df)),
            "unlabeled": int(len(index_df) - len(labels_df)),
        },
    }


@router.get("/preview")
def get_patch_preview(dataset_id: str = Query(...), patch_path: str = Query(...)):
    get_dataset_dir(dataset_id)

    # patch_path in parquet is stored project-relative, e.g. data\linhe_patches\scene\p_*.npz
    project_root = Path(settings.DATA_DIR).parent
    patch_file = project_root / patch_path
    if not patch_file.exists():
        raise HTTPException(status_code=404, detail=f"Patch file not found: {patch_file}")
    arr = np.load(patch_file)["rgb"]
    if arr.ndim == 3 and arr.shape[0] <= 10:
        arr = arr.transpose(1, 2, 0)
    image = Image.fromarray(arr.astype(np.uint8))
    buf = BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"status": "success", "data": {"patch_path": patch_path, "image_base64": f"data:image/png;base64,{encoded}"}}


class SaveLabelRequest(BaseModel):
    dataset_id: str
    patch_path: str
    label: str
    reviewer: str


@router.post("/save")
def save_label(payload: SaveLabelRequest):
    dataset_dir = get_dataset_dir(payload.dataset_id)
    labels_df = load_labels(dataset_dir)
    labels_df = labels_df[labels_df["patch_path"] != payload.patch_path]
    new_row = pd.DataFrame([{
        "dataset_id": payload.dataset_id,
        "patch_path": payload.patch_path,
        "label": payload.label,
        "reviewer": payload.reviewer,
        "labeled_at": datetime.now(tz=None).isoformat(timespec="seconds"),
    }])
    labels_df = pd.concat([labels_df, new_row], ignore_index=True)
    labels_df.to_parquet(dataset_dir / "_labels.parquet", index=False)
    return {"status": "success"}
