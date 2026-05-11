# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

- Geo-MLOps is a PEFT benchmark platform for geospatial foundation models built around Prithvi-100M.
- `geoadapter/` is the core Python library: PEFT adapters, Prithvi backbone loading, training/eval engine, dataset loaders, benchmark runners, and visualization utilities live here.
- `ae_backend/` is a thin FastAPI service that wraps `geoadapter` for dataset preparation, training job orchestration, model registry, and spatial lookup.
- `ae_frontend/` is a no-build Vue 3 + ECharts dashboard implemented as a single `index.html` and served by the backend.

## Commands

```bash
pip install -e .
pip install -r ae_backend/requirements.txt
python -m uvicorn app.main:app --app-dir ae_backend --host 127.0.0.1 --port 8087
python -m pytest tests -v
python -m pytest tests/test_trainer.py -v
python -m pytest tests/test_trainer.py::TestPEFTTrainer::test_one_step -v
python -m pytest tests/test_labels_api.py -v
docker build -t alphaearth-backend ae_backend
docker build -t alphaearth-frontend ae_frontend
```

- No repo-wide lint or typecheck command is configured in the checked-in files.
- There is no checked-in `docker-compose.yml`; build backend and frontend images separately.
- `ae_backend/test_pipeline.py` and `ae_backend/test_ws.py` are manual smoke scripts, not the maintained test suite.

## High-Level Architecture

- `geoadapter/` is intentionally independent of FastAPI. Most model-side changes belong here, not in `ae_backend/`.
- `ae_backend/app/main.py` mounts five routers under `/api/ae`: `pipeline`, `training`, `satellites`, `areas`, and `models`. It also serves `ae_frontend/index.html` at `/` when the frontend directory exists.
- `ae_backend/app/services/data_fusion.py` is the preprocessing core. It resolves ROI geometry from the `xiangzhen` PostGIS table, pulls public imagery from GEE or local/private `.tif` files, reprojects to a 10 m target grid, then slices fused rasters into 128x128 training patches.
- `ae_backend/app/core/memory.py` provides the in-memory dataset bridge. When `/api/ae/pipeline/prepare` runs with `in_memory=true`, tensors are stored there and later consumed by the trainer via `memory_{job_id}` dataset IDs.
- `ae_backend/app/services/trainer.py` owns the training runtime. `PrithviAlphaEarthEncoder` wraps the Prithvi backbone and switches among `linear_probe`, `bitfit`, `lora`, `houlsby`, and `geoadapter`. `AlphaEarthTrainer` streams logs/metrics over WebSocket, updates `ae_training_jobs`, and optionally uploads outputs to Huawei OBS.
- `ae_backend/app/models/domain.py` defines the persistent state: `AeDataset`, `AeTrainingJob`, `AeModel`, plus spatial lookup tables `Xiangzhen` and `SmSatellite`.
- `ae_frontend/index.html` talks directly to the backend with relative `/api/ae/...` requests and opens WebSockets at `/api/ae/training/ws/{job_id}`. There is no separate frontend build pipeline for local development.

## Constraints That Matter

- The root `.env` is authoritative and is loaded with `override=True` in `ae_backend/app/core/config.py`; do not auto-edit credentials.
- Full backend behavior depends on PostgreSQL/PostGIS. Spatial clipping and admin-area lookup come from the `xiangzhen` table.
- GEE and Huawei OBS are optional integrations. The code degrades when they are unavailable, but dataset download/upload features become partial or no-op.
- Training expects Prithvi weights at `data/weights/prithvi/Prithvi_100M.pt`.
- Dataset listing in `/api/ae/pipeline/datasets` only recognizes `data/weights/raw_data/dataset_*`, `data/linhe_patches/_index.parquet`, and in-memory datasets.
- The root `tests/` suite mostly validates `geoadapter` and small backend helpers; it is not a full end-to-end backend test harness.

## Current Documentation Gaps Fixed From The Previous Version

- Removed the stale `docker-compose up -d ae_backend ae_frontend` command; no compose file is checked in.
- Added file-scoped pytest commands, including a single-test invocation.
- Clarified that the frontend is a single static file served by the backend, and that the backend is mostly an orchestration layer around `geoadapter`.
- Marked the backend smoke scripts as non-canonical so future agents prefer the maintained pytest suite.
