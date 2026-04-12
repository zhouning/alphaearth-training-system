# 🌍 AlphaEarth 本地化训练 MLOps 平台 (AlphaEarth Local Fine-Tuning MLOps Pipeline)

[English](README_en.md) | **简体中文**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)](https://fastapi.tiangolo.com)
[![Vue3](https://img.shields.io/badge/Vue.js-3.0-4FC08D.svg)](https://vuejs.org/)

这是一个企业级的端到端 MLOps 平台，专为 **AlphaEarth 地理空间基础大模型**的本地化微调（Fine-Tuning）而设计。该系统通过自动化的多源数据融合、实时的训练监控以及云原生的资产管理，打通了从原始多模态遥感数据（光学、SAR 雷达）到下游 AI 业务任务的壁垒。

## ✨ 核心特性

*   **🛰️ 多模态数据融合 (Multi-Modal Data Fusion)**: 自动提取、对齐并融合来自云端的开源数据（通过 GEE 提取的 Sentinel-1/2, Landsat）与本地的保密高分影像（GF-1, GF-2, ZY-3）。
*   **🗺️ 动态空间裁剪与正射校正 (Dynamic Spatial Cropping)**: 基于 PostGIS 数据库中的真实行政区划边界，执行高精度的不规则多边形裁剪。支持在内存中读取 RPC 参数，对无坐标系的 L1A 级影像进行实时正射校正与投影变换。
*   **🧠 时空精度编码器 (STP Encoder)**: 集成了原生的 PyTorch 训练流。模型通过重构误差 (Reconstruction Loss) 和对比均匀性惩罚 (Uniformity Penalty) 学习地球的物理规律，将原始像素矩阵转化为 64 维的高维语义向量 (Embeddings)。
*   **📊 实时训练监控大屏 (Real-Time Dashboard)**: 采用 Vue 3 + Tailwind CSS + ECharts 构建的响应式前端，通过 WebSocket 与 FastAPI 后端连接，毫秒级流式推送训练指标（Epoch、Loss 曲线、ETA、显存占用等）。
*   **☁️ 云原生资产管理 (Cloud-Native Asset Registry)**: 切片完成后自动将张量数据集以及训练好的 PyTorch 权重文件 (`.pt`) 推送至华为云 OBS（对象存储），实现存算分离，并在“模型资产库”中进行统一的评估与默认推理版本激活。

## 🏗️ 系统架构

本项目被清晰地解耦为两个微服务：

1.  **`ae_backend/` (FastAPI + PyTorch + Rasterio)**
    *   负责空间数据处理 (`geopandas`, `rasterio`)、GEE 流式下载及模型张量 (Tensor) 生成。
    *   在后台异步调度 PyTorch 训练循环，并通过 WebSocket 广播防止事件循环阻塞。
    *   连接 PostgreSQL/PostGIS 进行空间查询，连接华为云 OBS 进行对象存储。
2.  **`ae_frontend/` (Vue 3 + Tailwind CSS + ECharts)**
    *   提供美观的交互式大屏，用于数据源准备、实时训练监控及模型资产评测。

## 🚀 快速启动 (Docker 推荐)

使用 Docker Compose 是运行本系统最简单的方式。

1.  **克隆仓库**:
    ```bash
    git clone https://github.com/zhouning/alphaearth-training-system.git
    cd alphaearth-training-system
    ```

2.  **配置环境变量**:
    在项目根目录创建一个 `.env` 文件，填入你的 PostgreSQL 凭证、华为云 OBS 密钥以及 Google Cloud 认证信息。

3.  **一键拉起服务**:
    ```bash
    docker-compose up -d ae_backend ae_frontend
    ```

4.  **访问大屏系统**:
    打开浏览器并访问 `http://localhost` (若以独立脚本启动后端并挂载了静态文件，则访问 `http://localhost:8085`)。

## 🧪 手动开发环境配置

如果你希望在没有 Docker 的情况下手动运行后端：

```bash
cd ae_backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8085 --reload
```

## 📝 许可证

本项目采用 MIT License 开源许可证。