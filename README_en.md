# 🌍 AlphaEarth Local Fine-Tuning MLOps Pipeline

**English** | [简体中文](README.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)](https://fastapi.tiangolo.com)
[![Vue3](https://img.shields.io/badge/Vue.js-3.0-4FC08D.svg)](https://vuejs.org/)

An enterprise-grade, end-to-end MLOps platform designed for the localized fine-tuning of **AlphaEarth Foundation Models**. This system bridges the gap between raw, multi-modal remote sensing data (optical, SAR) and downstream AI tasks through automated data fusion, real-time training, and cloud-native asset management.

## ✨ Key Features

*   **🛰️ Multi-Modal Data Fusion**: Automatically aligns and fuses open-source cloud data (Sentinel-1/2, Landsat via Google Earth Engine) with local, highly-classified high-resolution imagery (GF-1, GF-2, ZY-3).
*   **🗺️ Dynamic Spatial Cropping**: Leverages native PostGIS databases to perform precise irregular polygon cropping based on real-world administrative boundaries. Supports real-time Orthorectification via RPC parameters for L1A imagery without native CRS.
*   **🧠 Space-Time-Precision (STP) Encoder**: Integrates a PyTorch-based training loop that learns geographical representations via Reconstruction Loss and Contrastive Uniformity Penalty, transforming raw pixels into 64-dimensional semantic embeddings.
*   **📊 Real-Time Training Dashboard**: A responsive Vue 3 + Tailwind CSS + ECharts frontend that connects to the FastAPI backend via WebSockets to stream live training metrics (Epochs, Loss curves, ETA, and GPU memory usage).
*   **☁️ Cloud-Native Asset Registry**: Automatically pushes generated tensor patches and trained PyTorch weights (`.pt`) to Huawei Cloud OBS, decoupling compute from storage, and registers them in a centralized Model Asset Library for downstream evaluation.

## 🏗️ Architecture

The project is cleanly decoupled into two microservices:

1.  **`ae_backend/` (FastAPI + PyTorch + Rasterio)**
    *   Handles spatial data processing (`geopandas`, `rasterio`), GEE streaming downloads, and tensor generation.
    *   Manages the PyTorch training loop asynchronously with WebSocket broadcasting to prevent event loop blocking.
    *   Interacts with PostgreSQL/PostGIS for boundary querying and Huawei Cloud OBS for object storage.
2.  **`ae_frontend/` (Vue 3 + Tailwind CSS + ECharts)**
    *   Provides an interactive dashboard for dataset preparation, real-time training monitoring, and model evaluation.

## 🚀 Quick Start (Docker)

The easiest way to run the system is via Docker Compose.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/zhouning/alphaearth-training-system.git
    cd alphaearth-training-system
    ```

2.  **Configure Environment**:
    Create a `.env` file in the root directory containing your PostgreSQL, Huawei Cloud OBS, and Google Cloud credentials.

3.  **Launch the stack**:
    ```bash
    docker-compose up -d ae_backend ae_frontend
    ```

4.  **Access the Dashboard**:
    Open your browser and navigate to `http://localhost` (or `http://localhost:8085` if running the backend standalone with static file serving enabled).

## 🧪 Manual Development Setup

If you wish to run the backend service manually without Docker:

```bash
cd ae_backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8085 --reload
```

## 📝 License

This project is licensed under the MIT License.