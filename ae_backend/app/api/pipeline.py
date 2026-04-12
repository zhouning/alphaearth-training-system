from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional
from pydantic import BaseModel
from app.schemas.pipeline import DataSourceRequest, DataSourceEvaluation
from app.services.data_fusion import DataFusionPipeline
import uuid

router = APIRouter()

# In-memory store for task progress (prototype only)
tasks_db = {}

class PrepareRequest(BaseModel):
    satellite_sources: List[str]
    area_code: str

@router.post("/analyze", response_model=DataSourceEvaluation)
async def analyze_data_sources(request: DataSourceRequest):
    """
    Evaluates the suitability of selected satellite data sources for AlphaEarth training.
    """
    sources = [s.lower().replace(" ", "").replace("-", "") for s in request.satellite_sources]
    
    # 1. Spectral Overlap (Needs Optical B, G, R, NIR, SWIR)
    has_optical = any("sentinel2" in s or "landsat" in s or "gaofen" in s or "gf" in s or "zy" in s or "hj" in s or ".tif" in s for s in sources)
    spectral_score = 90 if has_optical else 20
    
    # 2. Spatial Resolution (Needs 10m)
    has_high_res = any("sentinel2" in s or "gaofen" in s or "gf" in s or "jilin" in s or "zy" in s or "hj" in s or ".tif" in s for s in sources)
    has_modis = any("modis" in s for s in sources)
    resolution_score = 100 if has_high_res else (10 if has_modis else 50)
    
    # 3. Temporal Density (Needs frequent revisit)
    temporal_score = 85 if any("sentinel" in s for s in sources) else 60
    
    # 4. Modality Richness (Needs Optical + SAR + Elevation)
    has_sar = any("sentinel1" in s or "gaofen3" in s or "gf3" in s or "alos" in s for s in sources)
    modality_score = 50
    if has_optical: modality_score += 20
    if has_sar: modality_score += 30
    
    # 5. Data Readiness (Local vs Cloud Download)
    readiness_score = 95 # With local .tif support, readiness is high
    
    overall_score = int((spectral_score + resolution_score + temporal_score + modality_score + readiness_score) / 5)
    
    recommendations = []
    if not has_optical:
        recommendations.append("AlphaEarth 的 STP 结构强烈依赖多光谱上下文，建议补充多光谱光学数据源 (如 Sentinel-2 或 GF-2)。")
    if not has_sar:
        recommendations.append("缺少 SAR 数据(如 Sentinel-1)，模型可能无法有效穿透云层提取地表纹理。")
    if has_modis:
        recommendations.append("MODIS 空间分辨率过低 (250m)，难以重采样至 10m 的目标尺度，建议移除该数据源。")
        
    if not recommendations:
        recommendations.append("数据源组合优异（支持公开与私有数据融合），符合 AlphaEarth 本地化训练的最佳要求！")

    return DataSourceEvaluation(
        spectral_overlap_score=spectral_score,
        spatial_resolution_score=resolution_score,
        temporal_density_score=temporal_score,
        modality_richness_score=modality_score,
        data_readiness_score=readiness_score,
        overall_score=overall_score,
        recommendations=recommendations
    )

def background_prepare_task(task_id: str, area_code: str, data_sources: List[str]):
    pipeline = DataFusionPipeline()
    
    def update_callback(progress: int, message: str):
        tasks_db[task_id] = {
            "status": "PROCESSING",
            "progress": progress,
            "message": message
        }
        
    try:
        tasks_db[task_id] = {"status": "PROCESSING", "progress": 0, "message": "任务已启动"}
        result = pipeline.prepare_dataset(area_code, data_sources, update_callback=update_callback)
        
        if isinstance(result, dict) and result.get("status") == "error":
            tasks_db[task_id] = {
                "status": "FAILED",
                "progress": 100,
                "message": result.get("message", "处理失败")
            }
        else:
            tasks_db[task_id] = {
                "status": "COMPLETED",
                "progress": 100,
                "message": "数据融合流水线处理完成",
                "result": result
            }
    except Exception as e:
        tasks_db[task_id] = {
            "status": "FAILED",
            "progress": 0,
            "message": f"流水线执行失败: {str(e)}"
        }

@router.post("/prepare")
async def prepare_dataset(request: PrepareRequest, background_tasks: BackgroundTasks):
    """
    触发真实的数据清洗流水线，支持融合公开 GEE 卫星流和本地/私有的 TIF 文件。
    """
    task_id = str(uuid.uuid4())
    tasks_db[task_id] = {"status": "PENDING", "progress": 0, "message": "准备中..."}
    
    background_tasks.add_task(
        background_prepare_task, 
        task_id=task_id,
        area_code=request.area_code, 
        data_sources=request.satellite_sources
    )
    
    return {
        "message": "数据融合流水线已在后台启动 (Data Fusion Pipeline started)",
        "task_id": task_id,
        "sources": request.satellite_sources,
        "area_code": request.area_code,
        "status": "PROCESSING"
    }

import os

@router.get("/datasets")
async def list_available_datasets():
    """
    列出所有已生成的有效训练数据集 (包含 .pt 文件的目录)
    """
    work_dir = "D:/adk/data_agent/weights/raw_data"
    datasets = []
    
    if os.path.exists(work_dir):
        for item in os.listdir(work_dir):
            item_path = os.path.join(work_dir, item)
            if os.path.isdir(item_path) and item.startswith("dataset_"):
                # Check if it has any .pt files
                pt_files = [f for f in os.listdir(item_path) if f.endswith('.pt')]
                if pt_files:
                    # Get modification time of the directory
                    mtime = os.path.getmtime(item_path)
                    datasets.append({
                        "id": item,
                        "name": f"数据集 {item[-8:]} ({len(pt_files)} 个切片)",
                        "mtime": mtime
                    })
                    
    # Sort by modification time, newest first
    datasets.sort(key=lambda x: x["mtime"], reverse=True)
    return {"status": "success", "data": datasets}

@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    获取后台流水线任务的状态
    """
    task = tasks_db.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task
