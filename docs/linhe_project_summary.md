# 临河区域版 AlphaEarth 增强项目总结

日期：2026-04-18

## 1. 项目背景与目标

本项目的目标是验证一条工程路线：

> 能否将小范围私有高分遥感影像，通过 AlphaEarth-System 平台，转化为一个可训练、可评估、可部署的区域增强能力？

数据来源是巴彦淖尔市临河区的高分卫星影像（`D:/image/临河`），覆盖 2019Q4 至 2022Q4 共 14 个季度。项目不追求训练新的基座模型，也不以论文发表为主目标，而是聚焦于工程落地验证。

## 2. 数据清单

| 项目 | 数值 |
|---|---|
| 原始场景数 | 53 |
| 总文件大小 | 224.1 GB |
| 卫星源 | GF-1/1C/1D, GF-6, ZY-3/ZY-02C, JKF-01 |
| 波段 | 3 波段 RGB (uint8)，无 NIR/SWIR |
| 空间分辨率 | 0.58m (JKF01) ~ 4.3m (ZY02C) |
| 覆盖范围 | lon [106.51, 108.17], lat [39.75, 41.69] |
| 切片后 patch 数 | 1293 (128x128, 10m GSD) |
| 已标注 patch 数 | 1293 (100%) |

标注类别分布（第二轮清洗后）：

| 类别 | 数量 | 占比 |
|---|---:|---:|
| 耕地 | 842 | 65.1% |
| 建筑 | 226 | 17.5% |
| 裸地 | 134 | 10.4% |
| 水体 | 63 | 4.9% |
| 道路 | 28 | 2.2% |

数据清单文件：`results/linhe/linhe_scenes.geojson`, `results/linhe/linhe_summary.md`
标签文件：`data/linhe_patches/_labels.parquet`

## 3. 构建内容

本项目在 AlphaEarth-System 平台上新增了以下工程能力：

### 3.1 数据扫描与 Patch 构建
- `scripts/linhe_scan_catalog.py`：扫描原始影像目录，生成统一场景目录
- `scripts/linhe_build_patches.py`：将高分影像重投影到 10m 网格，切成 128x128 patch
- 输出：`data/linhe_patches/_index.parquet`

### 3.2 系统内标注工作台
- 后端 API：`ae_backend/app/api/labels.py`（6 个端点）
- 前端：`ae_frontend/index.html` 中的 `Linhe 标注` tab
- 支持未标注模式和已标注复核模式
- 支持按类别筛选复核和标签覆盖更新
- 标签持久化：`data/linhe_patches/_labels.parquet`

### 3.3 PEFT 训练脚本
- `scripts/linhe_finetune.py`：支持 `--peft-method` 切换（linear_probe / bitfit / houlsby）
- 管线：RGB patch → GeoAdapter(3→6) → frozen Prithvi-100M → PEFT adapter → ClassificationHead
- 支持真实标签和 pseudo-label 两种模式
- 输出：`results/linhe/linhe_{method}_rgb.pt`

### 3.4 Benchmark 对比框架
- `scripts/linhe_benchmark.py`：统一超参下跑多方法对比
- 输出：`results/linhe/benchmark_real_labels.json`, `results/linhe/benchmark_real_labels.csv`

### 3.5 平台集成
- 后端 `GET /api/ae/pipeline/datasets` 自动识别 `linhe_patches`
- `AlphaEarthTrainer` 和模型评估均通过 `resolve_dataset_path()` 统一路径解析
- 前端训练页数据集下拉框自动显示临河数据集
- 模型注册脚本：`scripts/linhe_register_model.py`

## 4. 核心结果

### Benchmark 对比表（第二轮，5 类，10 epoch）

| 方法 | best_val_acc | weighted_f1 | macro_f1 | 参数量 |
|---|---:|---:|---:|---:|
| linear_probe | 0.6950 | **0.6683** | **0.4287** | 3,987 |
| bitfit | 0.6216 | 0.5073 | 0.3499 | 106,899 |
| houlsby | **0.7027** | 0.6425 | 0.4251 | 1,193,619 |

### Per-class F1

| 方法 | 建筑 | 水体 | 耕地 | 裸地 | 道路 |
|---|---:|---:|---:|---:|---:|
| linear_probe | 0.65 | 0.25 | **0.75** | 0.50 | 0.00 |
| bitfit | 0.58 | **0.27** | 0.54 | 0.36 | 0.00 |
| houlsby | **0.69** | 0.21 | 0.68 | **0.53** | 0.00 |

### 两轮对比（标签清洗前后）

| 指标 | 第一轮 linear_probe | 第二轮 linear_probe | 第一轮 houlsby | 第二轮 houlsby |
|---|---:|---:|---:|---:|
| best_val_acc | 0.6641 | **0.6950** | 0.5714 | **0.7027** |
| weighted_f1 | 0.5389 | **0.6683** | 0.4156 | **0.6425** |
| macro_f1 | 0.3481 | **0.4287** | 0.1212 | **0.4251** |

## 5. 工程决策

### 5.1 为什么选 linear_probe 作为当前生产候选
- 和 Houlsby 的 val_acc 差距仅 0.8%（0.6950 vs 0.7027）
- weighted_f1 反而更高（0.6683 vs 0.6425）
- 参数量只有 4K vs 1.2M，训练更快更稳
- 在当前样本规模下，复杂 PEFT 的额外自由度不值得

### 5.2 为什么去掉"其他"类
- 第一轮标注中"其他"占 16.3%，是一个语义模糊的兜底类
- 它会污染分类边界，导致 Houlsby 塌到主类
- 第二轮清洗后去掉"其他"，所有方法的 macro_f1 都显著提升

### 5.3 为什么不从零训练基座模型
- 临河数据只覆盖一个区县，不具备 foundation model 预训练所需的跨区域泛化覆盖
- 当前主数据是 3 波段 RGB，光谱维度不足
- 更合理的路线是在已有 Prithvi-100M 上做区域 PEFT 微调

### 5.4 为什么道路类 F1 为 0
- 道路样本仅 28 张（2.2%），严重不足
- 这不是模型问题，而是数据问题
- 需要定向补标或接受道路不是本区核心任务

## 6. 当前系统状态

### 文件位置
| 产物 | 路径 |
|---|---|
| 场景目录 | `results/linhe/linhe_scenes.geojson` |
| Patch 索引 | `data/linhe_patches/_index.parquet` |
| 标签文件 | `data/linhe_patches/_labels.parquet` |
| 最佳 checkpoint | `results/linhe/linhe_linear_probe_rgb.pt` |
| Benchmark 结果 | `results/linhe/benchmark_real_labels.csv` |
| 模型注册脚本 | `scripts/linhe_register_model.py` |

### 服务入口
- 后端：`uvicorn app.main:app --host 127.0.0.1 --port 8087`（需设置 PYTHONPATH 到 ae_backend）
- 前端：`http://127.0.0.1:8087/`
- 标注台：前端 `Linhe 标注` tab
- 模型资产库：前端 `模型资产库` tab

### 自动化测试
- `tests/test_labels_api.py`：8 条标注 API 测试
- `tests/test_linhe_backend_integration.py`：4 条后端集成测试
- `tests/test_linhe_scripts.py`：2 条脚本工具测试
- 全部通过

## 7. 推荐下一步

1. **定向补标道路样本**：当前 28 张完全不够，建议补到 80+ 再重跑 benchmark
2. **扩大 patch 覆盖范围**：当前只切了 1-2 个场景，53 个场景全量切片后可以验证空间泛化能力
3. **Sentinel-2 配对**：拉取对应位置的 Sentinel-2 多光谱数据，形成 RGB + S2 双模态 patch，突破 RGB-only 天花板
4. **如果 Houlsby 在更大数据量下持续领先**：考虑升级为主路线
5. **如果需要对外展示**：运行 `scripts/linhe_register_model.py` 注册模型到平台，在"模型资产库"中可见
