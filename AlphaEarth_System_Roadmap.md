# 🗺️ AlphaEarth 本地化训练管理系统 - 开发路线图 (Roadmap)

本项目旨在将单一的卫星轨道预测系统升级为**多源遥感数据融合与本地 AI 大模型微调的 MLOps 平台**。整个开发周期预计为 **12 周 (约 3 个月)**，分为 5 个核心阶段。

---

## 阶段一：基础设施搭建与架构“并轨” (第 1-2 周)
**目标**：完成独立微服务骨架的搭建，打通新旧数据链路，实现前端大屏的原型交付。

*   **[Milestone 1.1] 后端基建 (FastAPI)**
    *   搭建 FastAPI 工程目录结构 (`/api`, `/core`, `/models`, `/services`)。
    *   配置 SQLAlchemy 连接旧版 PostGIS 数据库 (`119.3.175.198`)。
    *   配置华为云 OBS SDK 连接池。
*   **[Milestone 1.2] 前端基建 (Vue 3 SPA)**
    *   初始化 Vue 3 + Vite + Tailwind CSS 工程。
    *   引入 Element Plus 和 ECharts。
    *   完成基于 `Data-Dense Dashboard` 规范的全局样式和组件封装。
*   **[Milestone 1.3] 资产整合与测试**
    *   实现 `/api/ae/satellites/available` 接口，打通老库的 `sm_satellite` 表。
    *   完成 WebSocket 通信通道的压力测试。

## 阶段二：多模态数据流转与清洗管线 (第 3-5 周)
**目标**：实现从用户框选 ROI 到生成标准化 PyTorch 训练张量 (`.pt`) 的全自动化数据流水线。

*   **[Milestone 2.1] 数据源智能体检模块**
    *   实现基于用户选择卫星源的雷达图评分逻辑（光谱、分辨率、重访周期等）。
*   **[Milestone 2.2] 异步任务调度中心**
    *   引入 Celery + Redis，构建可长时间运行的数据清洗后台任务。
*   **[Milestone 2.3] 空间裁剪与 GEE 抓取**
    *   基于 PostGIS 查询行政边界 (如 `和平村.shp`) 的 WKT/GeoJSON。
    *   利用 `geemap` 异步从 Google Earth Engine 下载指定区域、指定时间窗口的无云遥感影像 (Sentinel-2/Landsat等) 到本地/OBS。
*   **[Milestone 2.4] 张量化与归一化引擎**
    *   利用 `rasterio` 实现大图的 10m 分辨率重采样和投影对齐 (UTM)。
    *   实现 $log(x+1)/10$ 辐射归一化。
    *   滑动窗口切割为 `[128, 128]` 形状，并批量存储至华为云 OBS 的 `2_processed_tensors` 温数据区。

## 阶段三：AlphaEarth 核心训练引擎接入 (第 6-8 周)
**目标**：将 AlphaEarth 论文中的核心网络结构 (STP Encoder) 本地化，并与前端监控面板深度绑定。

*   **[Milestone 3.1] PyTorch 模型结构编写**
    *   实现 `Precision Branch` (卷积)、`Space Branch` (自注意力)、`Time Branch` (时序注意力)。
    *   实现 `Implicit Decoder` 用于重构误差计算。
    *   实现 `Batch Uniformity Loss` 确保特征在超球面上均匀分布。
*   **[Milestone 3.2] DataLoader 与硬件适配**
    *   编写从 OBS 流式读取/缓存张量切片的 `Dataset` 类。
    *   适配本地或 Colab 上的 GPU (CUDA) 加速。
*   **[Milestone 3.3] 实时训练监控闭环**
    *   将训练 Loop 的 Metrics (重构损失、均匀性损失、GPU负载) 实时推送到 WebSocket。
    *   前端 ECharts 动态渲染双线图。

## 阶段四：模型资产注册与管理 (第 9-10 周)
**目标**：将训练好的 AI 模型转化为可被下游系统调用的标准化资产。

*   **[Milestone 4.1] 模型持久化与元数据记录**
    *   训练完成后，自动将 `encoder.pth` 保存至 OBS 的 `3_model_registry`。
    *   向 PostGIS 的 `ae_models` 表写入模型得分、超参数、关联的区域和卫星源等元数据。
*   **[Milestone 4.2] 前端资产库面板**
    *   开发“模型资产库”页面，以网格卡片形式展示所有可用的本地化大模型。
    *   提供模型详情查看、训练报告下载、和一键激活为“默认预测模型”的操作。
*   **[Milestone 4.3] 特征缓存 (Optional)**
    *   利用 `pgvector` 将模型推理后的 64 维向量特征入库，为未来的“以图搜图”或“违建检测”打下基础。

## 阶段五：生产环境部署与全链路验收 (第 11-12 周)
**目标**：在新域名下完成独立部署，进行端到端的可用性与性能验收。

*   **[Milestone 5.1] 独立域名与网关配置**
    *   配置 Nginx 反向代理，将新申请的域名 (如 `ae.img.net`) 指向 FastAPI 的 8000 端口。
    *   配置 HTTPS 证书及完善的 CORS 跨域策略。
*   **[Milestone 5.2] 生产级容器化编排**
    *   编写 `Dockerfile` 和 `docker-compose.yml`，将 FastAPI 后端、Celery Worker 容器化。
    *   前端使用 Nginx 静态托管发布。
*   **[Milestone 5.3] 全链路真实业务演练**
    *   选取真实的“和平村”等区域，从数据勾选 -> GEE下载 -> 自动切片 -> 模型训练 -> 面板监控 -> 资产入库，执行无人工干预的端到端演练。
    *   修复极端异常（如网络中断、GPU OOM 恢复机制）。

---
**🚀 长期愿景 (Future Outlook)**：
在 V1.0 本地化训练系统平稳运行后，V2.0 可进一步接入更丰富的下游任务头 (Task Heads)，例如：基于本地特征的**农作物分类**、**非法建筑提取**、**地表沉降分析**等。