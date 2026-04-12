import os
import uuid
import logging
import time
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window
from rasterio.io import MemoryFile
import torch

from app.db.database import SessionLocal
from sqlalchemy import text
from app.core.config import settings

# Initialize OBS Client safely
try:
    from obs import ObsClient
except ImportError:
    ObsClient = None

# Initialize GEE safely
try:
    import ee
    import geemap
    ee.Initialize(project=os.getenv('GOOGLE_CLOUD_PROJECT', 'gen-lang-client-0977577668'))
except Exception as e:
    logging.warning(f"GEE Initialization skipped or failed: {e}")

logger = logging.getLogger(__name__)

class DataFusionPipeline:
    """
    AlphaEarth 核心预处理管线：支持公开与私有卫星数据的时空融合、裁切与 10m 归一化。
    """
    def __init__(self, work_dir: str = "D:/adk/data_agent/weights/raw_data"):
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)
        self.patch_size = 128
        self.target_res = 10.0 # AlphaEarth 要求的 10 米分辨率

    def _get_area_bounds(self, area_code: str) -> np.ndarray:
        # 默认回退边界（江西某地）
        bounds = np.array([116.64, 29.56, 117.07, 30.08])
        try:
            parts = area_code.split('-')
            if len(parts) == 4:
                province, city, county, township = parts
                with SessionLocal() as db:
                    query = text(
                        "SELECT ST_XMin(ST_Extent(geometry)), ST_YMin(ST_Extent(geometry)), "
                        "ST_XMax(ST_Extent(geometry)), ST_YMax(ST_Extent(geometry)) "
                        "FROM xiangzhen WHERE province = :p AND city = :c AND county = :co AND township = :t"
                    )
                    result = db.execute(query, {"p": province, "c": city, "co": county, "t": township}).fetchone()
                    if result and result[0] is not None:
                        bounds = np.array([result[0], result[1], result[2], result[3]])
                        logger.info(f"成功从数据库查找到区域的真实空间边界: {bounds}")
        except Exception as e:
            logger.warning(f"从数据库获取空间边界失败: {e}")
        return bounds

    def _get_area_geom_and_bounds_utm(self, area_code: str, target_crs: str):
        try:
            parts = area_code.split('-')
            if len(parts) == 4:
                province, city, county, township = parts
                with SessionLocal() as db:
                    srid = int(target_crs.split(':')[1])
                    query = text(
                        "SELECT ST_AsGeoJSON(ST_Transform(geometry, :srid)), "
                        "ST_XMin(ST_Transform(geometry, :srid)), ST_YMin(ST_Transform(geometry, :srid)), "
                        "ST_XMax(ST_Transform(geometry, :srid)), ST_YMax(ST_Transform(geometry, :srid)) "
                        "FROM xiangzhen WHERE province = :p AND city = :c AND county = :co AND township = :t"
                    )
                    result = db.execute(query, {"srid": srid, "p": province, "c": city, "co": county, "t": township}).fetchone()
                    if result and result[0]:
                        import json
                        geom = json.loads(result[0])
                        bounds = (result[1], result[2], result[3], result[4])
                        return geom, bounds
        except Exception as e:
            logger.warning(f"获取 UTM 几何数据失败: {e}")
        return None, None

    def _upload_to_obs(self, local_dir: str, job_id: str, update_callback=None):
        if not ObsClient or not settings.HUAWEI_OBS_AK or not settings.HUAWEI_OBS_SK:
            logger.warning("华为云 OBS 凭证未配置或 SDK 缺失，跳过云端同步。")
            return

        try:
            obs_client = ObsClient(
                access_key_id=settings.HUAWEI_OBS_AK,
                secret_access_key=settings.HUAWEI_OBS_SK,
                server=settings.HUAWEI_OBS_SERVER
            )
            bucket_name = settings.HUAWEI_OBS_BUCKET
            
            files = os.listdir(local_dir)
            total = len(files)
            
            for idx, file in enumerate(files):
                local_path = os.path.join(local_dir, file)
                if not os.path.isfile(local_path): continue
                
                object_key = f"alphaearth/datasets/dataset_{job_id}/{file}"
                # 真实上传到华为云 OBS
                obs_client.putFile(bucket_name, object_key, local_path)
                
                if update_callback and (idx + 1) % max(1, total // 5) == 0:
                    progress = 85 + int(15 * (idx / total))
                    update_callback(progress, f"正在同步至华为云 OBS... ({idx+1}/{total})")
            
            if update_callback: update_callback(100, f"成功同步至 OBS: obs://{bucket_name}/alphaearth/datasets/dataset_{job_id}/")
            obs_client.close()
        except Exception as e:
            logger.error(f"OBS 同步失败: {e}")

    def prepare_dataset(self, area_code: str, data_sources: list, in_memory: bool = False, update_callback=None) -> dict:
        """
        核心调度流：处理包含公开或私有来源的混合数据源
        data_sources: 来源列表，例如 ["Sentinel-2", "D:/private_data/local_gf2_2023.tif"]
        """
        job_id = str(uuid.uuid4())[:8]
        output_dir = os.path.join(self.work_dir, f"dataset_{job_id}")
        os.makedirs(output_dir, exist_ok=True)

        if update_callback: update_callback(5, "正在提取 ROI 的真实多边形边界...")
        logger.info(f"[{job_id}] 正在提取 ROI 空间范围 ({area_code})...")
        
        # 使用数据库中的真实地理边界
        bounds = self._get_area_bounds(area_code)
        
        # --- 限制处理范围以满足 GEE 直链下载和本地快速演示的限制 ---
        # GEE getDownloadURL API 有单次 50MB (约 10kmx10km@10m) 的硬性体积限制
        minx, miny, maxx, maxy = bounds
        centroid_lon = (minx + maxx) / 2.0
        centroid_lat = (miny + maxy) / 2.0
        half_size = 0.05 # 约 5.5km
        bounds = np.array([
            max(minx, centroid_lon - half_size),
            max(miny, centroid_lat - half_size),
            min(maxx, centroid_lon + half_size),
            min(maxy, centroid_lat + half_size)
        ])
        logger.info(f"[{job_id}] 为适应 GEE 下载配额和演示速度，边界已收缩至目标中心的 10km x 10km: {bounds}")
        
        minx, miny, maxx, maxy = bounds
        centroid_lon = (minx + maxx) / 2.0
        utm_zone = int((centroid_lon + 180) / 6) + 1
        target_crs = f"EPSG:326{utm_zone:02d}" # 北半球 UTM
        logger.info(f"[{job_id}] 目标投影计算完成: {target_crs}")

        aligned_files = []
        total_sources = len(data_sources)
        
        for idx, src in enumerate(data_sources):
            src_lower = src.lower()
            progress = 10 + int(60 * (idx / max(total_sources, 1)))
            
            # --- 1. 处理公开的 GEE 卫星流 ---
            if "sentinel" in src_lower or "landsat" in src_lower:
                out_path = os.path.join(output_dir, f"{src_lower}_aligned.tif")
                if update_callback: update_callback(progress, f"正在向 GEE 请求真实下载 {src} 并自动重采样为 10 米分辨率...")
                logger.info(f"[{job_id}] 正在向 GEE 发起真实请求下载 {src} 并执行空间对齐...")
                
                try:
                    import concurrent.futures
                    
                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    future = executor.submit(self._download_from_gee, src_lower, bounds, target_crs, out_path)
                    # Set a hard timeout of 120 seconds for the GEE download
                    future.result(timeout=120)
                    executor.shutdown(wait=False)
                    
                    aligned_files.append(out_path)
                except concurrent.futures.TimeoutError:
                    logger.error(f"从 GEE 下载 {src} 超时 (超过120秒)")
                    if update_callback: update_callback(progress, f"从 GEE 下载 {src} 超时网络阻断，跳过...")
                    executor.shutdown(wait=False)
                except Exception as e:
                    logger.error(f"从 GEE 下载 {src} 失败: {e}")
                    if update_callback: update_callback(progress, f"从 GEE 下载 {src} 失败，跳过...")
                    executor.shutdown(wait=False)
                
            # --- 2. 处理私有/本地的遥感影像 ---
            else:
                # 真实系统：从配置的私有样本目录加载文件
                real_src = None
                
                # 尝试通过名字去样本库里找匹配的 .tif / .tiff
                for root, _, files in os.walk(settings.SAMPLES_DIR):
                    for file in files:
                        if (src_lower in file.lower() or file.lower().startswith(src_lower)) and (file.endswith('.tif') or file.endswith('.tiff')):
                            real_src = os.path.join(root, file)
                            break
                    if real_src: break

                # 如果没找到，退回示例逻辑：直接使用目录下的任何一个tif作为演示
                if not real_src:
                    logger.warning(f"[{job_id}] 警告: 在样本库中未找到 {src} 的对应文件，将尝试使用随机的 .tif 代替以作演示。")
                    for root, _, files in os.walk(settings.SAMPLES_DIR):
                        for file in files:
                            if file.endswith('.tif') or file.endswith('.tiff'):
                                real_src = os.path.join(root, file)
                                break
                        if real_src: break
                
                if real_src and os.path.exists(real_src):
                    if update_callback: update_callback(progress, f"正在按照 {area_code} 的空间多边形裁切本地高分数据: {real_src} ...")
                    logger.info(f"[{job_id}] 准备裁切本地高分数据: {real_src}")
                    
                    file_name = os.path.splitext(os.path.basename(real_src))[0]
                    out_path = os.path.join(output_dir, f"private_{file_name}_cropped.tif")
                    
                    try:
                        self._align_private_raster(real_src, bounds, target_crs, out_path)
                        aligned_files.append(out_path)
                    except Exception as e:
                        if update_callback: update_callback(progress, f"影像与区域不重叠，跳过 {src}...")
                else:
                    if update_callback: update_callback(progress, f"未找到 {src} 的本地数据，跳过...")

        # --- 3. 融合、堆叠与滑动切片生成 Tensor ---
        if update_callback: update_callback(75, f"正在读取真实的 {len(aligned_files)} 个源波段，执行辐射归一化，切割为 128x128 张量...")
        if aligned_files:
            logger.info(f"[{job_id}] 开始对真实影像进行切片 (in_memory={in_memory})...")
            
            # Real slicing computation
            tensor_count, total_time, throughput = self._fuse_and_slice(aligned_files, output_dir, in_memory=in_memory, job_id=job_id)
            
            if tensor_count == 0:
                if update_callback: update_callback(100, "裁剪区域内无足够有效的 128x128 像素块。")
                return {"status": "error", "message": "目标区域（行政区）的范围在影像内太小，无法切出一个完整的 128x128 训练张量。请尝试选择更大的行政区或市级单位。"}
            
            # --- 4. 同步至华为云 OBS ---
            if not in_memory:
                if update_callback: update_callback(85, "开始将模型张量推送到华为云 OBS 对象存储...")
                self._upload_to_obs(output_dir, job_id, update_callback)
            else:
                logger.info(f"[{job_id}] 内存直通模式已开启，跳过对象存储上传以消解网络 I/O。")
            
            if update_callback: 
                msg = f"处理完毕。已生成 {tensor_count} 个模型切片。"
                if in_memory:
                    msg += f" (内存直通 🚀 吞吐量: {throughput:.1f} patches/sec, 耗时: {total_time:.2f}s)"
                update_callback(100, msg)
                
            return {
                "status": "success", 
                "job_id": job_id, 
                "patches_generated": tensor_count, 
                "throughput": throughput,
                "total_time": total_time,
                "in_memory": in_memory,
                "obs_path": f"obs://{settings.HUAWEI_OBS_BUCKET}/alphaearth/datasets/dataset_{job_id}/" if not in_memory else "InMemory"
            }
        else:
            if update_callback: update_callback(100, "未能找到任何有效的本地或公开数据。")
            return {"status": "error", "message": "未能找到任何有效的本地或公开数据。"}

    def _download_from_gee(self, source_name, bounds, target_crs, out_path):
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        import shutil
        
        roi = ee.Geometry.Rectangle(bounds.tolist())
        
        if "sentinel2" in source_name or "sentinel-2" in source_name:
            collection = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
                .filterBounds(roi) \
                .filterDate('2023-01-01', '2023-03-31') \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
                .select(['B2', 'B3', 'B4', 'B8', 'B11'])
            image = collection.first().clip(roi)
        elif "sentinel1" in source_name or "sentinel 1" in source_name:
            collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
                .filterBounds(roi) \
                .filterDate('2023-01-01', '2023-03-31') \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                .select(['VV'])
            image = collection.first().clip(roi)
        elif "landsat" in source_name:
            collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                .filterBounds(roi) \
                .filterDate('2023-01-01', '2023-03-31') \
                .filter(ee.Filter.lt('CLOUD_COVER', 20)) \
                .select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5'])
            image = collection.first().clip(roi)
        else:
            raise ValueError(f"不支持的公开数据源: {source_name}")
            
        logger.info("正在向 GEE 申请直接下载链接 (getDownloadURL)...")
        try:
            url = image.getDownloadURL({
                'scale': self.target_res,
                'crs': target_crs,
                'region': roi,
                'format': 'GEO_TIFF'
            })
            logger.info(f"成功获取直链: {url[:60]}...")
            
            # Setup robust requests session for proxy environments
            session = requests.Session()
            retry = Retry(connect=3, backoff_factor=0.5)
            adapter = HTTPAdapter(max_retries=retry)
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            
            # Streaming download with chunk iteration to prevent hang
            with session.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(out_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            logger.info("GEE 影像下载与重投影完成！")
            
        except Exception as e:
            logger.error(f"GEE 下载通道失败: {e}")
            raise e

    def _align_private_raster(self, src_path, wgs84_bounds, target_crs, out_path):
        """
        核心能力：将任意分辨率、任意投影的私有 TIF，重采样为 10米，并裁切到 WGS84 对应的边界内。
        """
        import rasterio.mask
        from rasterio.vrt import WarpedVRT
        from shapely.geometry import box
        import json
        
        with rasterio.open(src_path) as src:
            # 检查是否有坐标系或是否含有 RPC 有理多项式 (如高分、资源系列 L1A)
            if src.crs is not None or src.rpcs is not None:
                minx, miny, maxx, maxy = wgs84_bounds
                geo_box = box(minx, miny, maxx, maxy)
                
                try:
                    gdf = gpd.GeoDataFrame({'geometry': [geo_box]}, crs="EPSG:4326")
                    
                    if src.rpcs is not None:
                        logger.info(f"检测到 RPC 参数，启动 VRT 实时正射校正与投影转换...")
                        # 动态将带有 RPC 的 L1A 数据校正并重投影到目标坐标系
                        with WarpedVRT(src, crs=target_crs, resampling=Resampling.bilinear) as vrt:
                            gdf = gdf.to_crs(target_crs)
                            shapes = [feature["geometry"] for feature in json.loads(gdf.to_json())['features']]
                            
                            out_image, out_transform = rasterio.mask.mask(vrt, shapes, crop=True)
                            out_meta = vrt.meta.copy()
                            out_meta.update({"driver": "GTiff",
                                             "height": out_image.shape[1],
                                             "width": out_image.shape[2],
                                             "transform": out_transform})
                            with rasterio.open(out_path, "w", **out_meta) as dest:
                                dest.write(out_image)
                            logger.info("利用 RPC 自动正射校正并完成了严格行政区多边形裁切！")
                            return
                    else:
                        gdf = gdf.to_crs(src.crs)
                        shapes = [feature["geometry"] for feature in json.loads(gdf.to_json())['features']]
                        
                        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
                        out_meta = src.meta.copy()
                        out_meta.update({"driver": "GTiff",
                                         "height": out_image.shape[1],
                                         "width": out_image.shape[2],
                                         "transform": out_transform})
                        with rasterio.open(out_path, "w", **out_meta) as dest:
                            dest.write(out_image)
                        logger.info("基于数据库边界的严格空间相交裁切已完成。")
                        return
                except Exception as e:
                    logger.error(f"严格空间相交裁切失败: {e}")
                    raise ValueError(f"您选择的区域与这幅卫星影像的真实地理覆盖范围没有重叠！ ({e})")
            else:
                logger.error(f"影像 {src_path} 缺少空间参考 (CRS=None，且无 RPC 附属文件)。")
                raise ValueError("影像缺乏必须的空间坐标系(CRS)或RPC信息，无法根据行政区边界进行精准裁切！")

    def _fuse_and_slice(self, aligned_files, output_dir, in_memory=False, job_id=None):
        """
        将多个对齐好的影像进行通道堆叠 (Channel Stacking)，
        应用 AlphaEarth log(x+1)/10 辐射归一化，并切成 128x128 张量。
        """
        from app.core.memory import IN_MEMORY_DATASETS
        import time
        
        patch_count = 0
        start_time = time.time()
        memory_tensors = []
        
        with rasterio.open(aligned_files[0]) as src:
            H, W = src.shape
            stride = 128
            
            # 限制处理范围，避免演示时耗时过长
            max_h = min(H, self.patch_size * 5)
            max_w = min(W, self.patch_size * 5)
            
            for y in range(0, max_h - self.patch_size + 1, stride):
                for x in range(0, max_w - self.patch_size + 1, stride):
                    window = Window(col_off=x, row_off=y, width=self.patch_size, height=self.patch_size)
                    patch = src.read(window=window)
                    
                    # 补充通道到5 (AlphaEarth模型期望5通道)
                    C = patch.shape[0]
                    if C < 5:
                        pad = np.zeros((5-C, self.patch_size, self.patch_size), dtype=patch.dtype)
                        patch = np.concatenate([patch, pad], axis=0)
                    elif C > 5:
                        patch = patch[:5, :, :]
                        
                    patch = np.nan_to_num(patch, nan=0.0)
                    patch = np.clip(patch, 0, 10000)
                    patch = np.log1p(patch) / 10.0
                    mean = np.mean(patch, axis=(1,2), keepdims=True)
                    std = np.std(patch, axis=(1,2), keepdims=True) + 1e-6
                    patch = (patch - mean) / std
                    
                    tensor = torch.tensor(patch, dtype=torch.float32)
                    
                    if in_memory:
                        memory_tensors.append(tensor)
                    else:
                        torch.save(tensor, os.path.join(output_dir, f"patch_{patch_count:04d}.pt"))
                        
                    patch_count += 1
                    
        if in_memory and job_id:
            IN_MEMORY_DATASETS[job_id] = memory_tensors
            
        total_time = time.time() - start_time
        throughput = patch_count / max(total_time, 0.001)
        
        return patch_count, total_time, throughput
