# 临河样例数据接入说明

## 当前结论

这批 `D:/image/临河` 数据更适合作为 **区域私有数据源 + PEFT 微调数据**，而不是从零训练一个新的基座模型。

原因：
- 数据范围集中在临河局部，不具备 foundation model 预训练所需的跨区域泛化覆盖
- 当前主数据是 3 波段 RGB 成品影像，没有 NIR/SWIR，光谱信息不足
- 更合理的路线是：`RGB 私有影像 -> GeoAdapter(3→6) -> Prithvi-100M -> Houlsby Adapter`

## 已接入内容

### 1. 数据扫描与清单
- `scripts/linhe_scan_catalog.py`
- 输出：
  - `results/linhe/linhe_scenes.geojson`
  - `results/linhe/linhe_scenes.csv`
  - `results/linhe/linhe_summary.md`

作用：统一整理季度、卫星、footprint、日期、分辨率、文件大小。

### 2. 离线 Patch 构建
- `scripts/linhe_build_patches.py`
- 输出目录：`data/linhe_patches/`

当前默认流程：
- 将原始高分影像重投影到 EPSG:3857
- 重采样到 10m 网格
- 切成 128x128 patch
- 保存为 `.npz`

这一步是为了和 AlphaEarth / Sentinel-2 级别网格对齐，便于后续做融合与对照实验。

### 3. PEFT 微调脚手架
- `scripts/linhe_finetune.py`

当前实现：
- 使用 `PrithviBackbone`
- 注入 `Houlsby Adapter`
- 用 `GeoAdapter` 把 RGB 映射到 6 通道输入
- 在缺少人工标注时，先用 patch 统计特征做 pseudo-label 代理任务，验证训练链路

说明：这不是最终业务模型，而是“临河区域专家模型”的第一版训练入口。

### 4. 平台可见数据集
后端 `GET /api/ae/pipeline/datasets` 现在会自动识别：
- `data/linhe_patches/_index.parquet`

并将其暴露为：
- `linhe_patches`

前端训练页会自动把它显示到数据集下拉框中。

### 5. 训练链路兼容
- `AlphaEarthTrainer` 已支持 `dataset_id == "linhe_patches"`，会解析到 `data/linhe_patches`
- `RealPatchDataset` 已支持递归读取 `.npz` RGB patch
- `.npz` patch 会按 `[3, H, W] -> [5, H, W]` 零填充到现有训练输入形状
- 已新增聚焦回归测试：
  - `tests/test_linhe_backend_integration.py`
  - `tests/test_linhe_scripts.py`


### 6. 系统内标注能力
- 前端新增 `Linhe 标注` tab
- 第一版固定类别：耕地 / 林地 / 水体 / 建筑 / 道路 / 裸地 / 其他
- 标签结果保存到 `data/linhe_patches/_labels.parquet`
- 标注结果可直接与 `_index.parquet` 关联，供后续监督训练使用

## 这批数据在 AlphaEarth-System 里的两种用途

### 用途 A：区域 PEFT 微调
目标：训练一个只服务临河或相近场景的区域模型。

推荐任务：
- 地类分类
- 地块变化检测
- 农业地物识别
- 小范围时序监测

### 用途 B：AlphaEarth 叠加数据源
目标：把临河高分 RGB 作为 AlphaEarth 全球底座之上的私有增强层。

推荐方式：
- AlphaEarth / Sentinel-2 提供统一低分辨率语义底座
- 临河样例提供高分辨率细节纹理和局部判别信息
- 在 ROI 内做 patch-level 融合、对齐与下游检索/分类

### 7. 复核 / 重标模式
- `Linhe 标注` 支持 `未标注模式` 与 `已标注复核模式`
- 在复核模式下可按现有标签筛选：耕地 / 林地 / 水体 / 建筑 / 道路 / 裸地 / 其他
- 页面显示 `当前标签`
- 点击新标签会覆盖更新 `_labels.parquet` 中该 patch 的旧标签
- 推荐优先复核：其他 / 道路 / 水体 / 林地

## 当前限制

- 还没有 Sentinel-2 / AlphaEarth embedding 的自动配对拉取流程
- 还没有人工标注监督任务
- `linhe_finetune.py` 当前是代理任务验证，不代表最终精度上限

## 下一步建议

1. 先批量生成 `linhe_patches`
2. 补一个 Sentinel-2 配对脚本，形成 `RGB + S2` 双模态 patch 数据集
3. 选一个真实任务做少量人工标注
4. 用 Houlsby 做正式 PEFT 对比实验：
   - linear_probe
   - bitfit
   - houlsby
   - geoadapter-only
5. 把结果沉淀成“区域版 AlphaEarth 增强方案”
