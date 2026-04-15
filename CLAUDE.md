# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Geo-MLOps: PEFT Benchmark Platform for Geospatial Foundation Models — a systematic evaluation platform for parameter-efficient fine-tuning of Prithvi-100M on heterogeneous satellite inputs. Two-service architecture: FastAPI+PyTorch backend (`ae_backend/`) and Vue 3 dashboard (`ae_frontend/`). The `geoadapter/` package is an independent Python library (zero FastAPI dependency) containing 5 PEFT method implementations, Prithvi-100M backbone loader (149/149 weights), unified training engine, and benchmark runner.

75 experiments completed on EuroSAT (5 methods x 5 modalities x 3 seeds). Key findings: LoRA fails on Prithvi's fused-QKV architecture; Houlsby adapter dominates (+16-23%); input-stage adaptation (GeoAdapter) does not work — negative result documented in `docs/Experiment_Results_Analysis.md`.

Research context: Pivoted from "GeoAdapter as method contribution" to "PEFT benchmark with negative results." Paper target: GIScience & Remote Sensing or CVPR EarthVision Workshop.

## Commands

### Run backend
```bash
cd ae_backend
pip install -e ../.  # install geoadapter package
uvicorn app.main:app --host 127.0.0.1 --port 8087
```

### Run with Docker
```bash
docker-compose up -d ae_backend ae_frontend
```

### Run tests
```bash
cd ae_backend
python test_pipeline.py      # Integration test — hits /api/ae/pipeline/prepare
python test_ws.py            # WebSocket connection test
```

### Frontend
The frontend is a single `ae_frontend/index.html` file (Vue 3 + Tailwind CSS + ECharts, no build step). Served as static files by the backend or via nginx in Docker.

## Architecture

### Two Services

**`ae_backend/`** — FastAPI + PyTorch + Rasterio
- `app/main.py`: FastAPI app, CORS, router mounting, static file serving
- `app/api/`: 5 routers — `pipeline` (data fusion), `training` (PEFT jobs + WebSocket), `satellites` (catalog), `areas` (PostGIS admin boundaries), `models` (asset registry)
- `app/services/data_fusion.py`: `DataFusionPipeline` — GEE download, local .tif ingestion, spatial alignment, 10m normalization, 128x128 patch generation
- `app/services/trainer.py`: `AlphaEarthTrainer` — STP encoder training loop with WebSocket broadcast, OBS upload, PCA embedding visualization
- `app/models/domain.py`: SQLAlchemy models — `AeDataset`, `AeTrainingJob`, `AeModel`, `SmSatellite`, `Xiangzhen` (PostGIS admin boundaries)
- `app/core/config.py`: Pydantic Settings from `.env` (loads with `override=True` to prevent stale system env vars)
- `app/core/memory.py`: `IN_MEMORY_DATASETS` dict for zero-copy tensor pipelining
- `app/db/database.py`: SQLAlchemy engine + `SessionLocal` + `get_db()` dependency

**`ae_frontend/`** — Single HTML file with embedded Vue 3 + ECharts
- Real-time training dashboard via WebSocket (`ws://host:8085/api/ae/training/ws`)
- Tabs: data source selection, pipeline monitoring, training metrics (loss curves, ETA, GPU), model asset registry

### API Prefix
All backend routes use `/api/ae/` prefix (configured in `Settings.API_V1_STR`).

### Data Flow
1. User selects satellite sources + admin area → `POST /api/ae/pipeline/prepare`
2. `DataFusionPipeline` fetches GEE data + reads local .tif → spatial alignment → 128x128 patches (disk or in-memory)
3. `POST /api/ae/training/start` → `AlphaEarthTrainer` runs STP encoder PEFT loop in background task
4. Training metrics broadcast via WebSocket → frontend ECharts updates in real-time
5. Trained weights pushed to Huawei OBS → registered in `ae_models` table

### Zero-Copy Tensor Pipelining (Paper 9)
When `in_memory=True`, `DataFusionPipeline` stores tensors in `IN_MEMORY_DATASETS[job_id]` instead of writing to disk. `RealPatchDataset` in `trainer.py` detects `memory_` prefix in dataset_id and reads directly from RAM, eliminating I/O bottleneck.

### Database
PostgreSQL + PostGIS. Tables: `ae_datasets`, `ae_training_jobs`, `ae_models`, `xiangzhen` (admin boundaries with MULTIPOLYGON geometry), `sm_satellite` (legacy satellite catalog). Connection via SQLAlchemy with `pool_pre_ping=True`.

## Environment Variables

Configured in `.env` at project root (loaded by `app/core/config.py`):
- `POSTGRES_SERVER`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
- `HUAWEI_OBS_AK`, `HUAWEI_OBS_SK`, `HUAWEI_OBS_SERVER`, `HUAWEI_OBS_BUCKET`

**CRITICAL**: Never modify `.env` automatically — it contains manually configured credentials. Output suggested changes as text for the user to copy.

## Key Constraints

- GEE (`earthengine-api` + `geemap`) initialization may fail if no Google Cloud credentials are configured — the system logs a warning and continues without GEE support
- OBS SDK (`obs-sdk-python`) is optional — `ObsClient` import is wrapped in try/except throughout
- The `xiangzhen` table provides real Chinese admin boundary polygons for spatial clipping — queries use PostGIS `ST_Intersects`
- Frontend connects to backend WebSocket at the same host — no separate frontend dev server needed
- `data/` directory is gitignored (contains raw samples and model weights)
- Language: UI text and comments are primarily in Chinese; API code is in English
