# Paper12 相关算法实现 ArcGIS GeoVLM 的可行性评估

**评估日期：** 2026 年 7 月 18 日

**评估对象：** Paper12 相关算法模型与 ArcGIS Geospatial Vision Language Model（GeoVLM）

**参考文档：** [Concepts of Geospatial Vision Language Model](https://doc.arcgis.com/en/pretrained-models/latest/imagery/concepts-of-geospatial-vision-language-model.htm)

## 一、结论

Paper12 当前的算法可以复用为 GeoVLM 的遥感视觉底座，并已具备固定类别分类、固定类别语义分割、传感器通道适配和参数高效微调能力，但不能直接实现 ArcGIS 文档中的完整 GeoVLM。

Paper12 本质上是“遥感视觉编码器 + 参数高效适配模块 + 固定任务头”。当前主干仍是 Prithvi-100M，最新代码还加入了 SatMAE 兼容的第二骨干验证。ArcGIS GeoVLM 则是“视觉编码器 + 语言模型 + 多模态对齐模块 + 语言条件化像素解码器 + GIS 部署组件”。两者目前只在视觉编码和部分像素级任务上重合。

因此，准确的技术判断是：Paper12 已经具备 GeoVLM 的遥感视觉编码、传感器适配和 PEFT 实验基础，但要完整对标 ArcGIS GeoVLM，仍需新增语言通路、图文对齐、提示词条件化解码器、指令数据训练以及 ArcGIS 部署层。这属于一项新的模型研发工作，而不是给现有分类头增加一个文本参数。

## 二、能力对照

| ArcGIS GeoVLM 能力 | Paper12 当前状态 | 说明 |
|---|---|---|
| 固定类别场景分类 | 已实现 | 支持单标签和多标签分类，但类别空间在训练时固定。 |
| 固定类别语义分割 | 已实现 | 支持固定类别的像素级预测。 |
| 自然语言提示分割 | 未实现 | 当前分割头不接收文本提示。 |
| 图像描述 | 未实现 | 没有语言生成模型和自回归文本解码器。 |
| 视觉问答 | 未实现 | 没有图文交叉注意力和问答训练通路。 |
| 目标计数 | 未实现 | 可以对特定分割结果做后处理计数，但不具备开放式语言计数能力。 |
| 指代表达分割 | 未实现 | 无法处理“图像顶部的飞机”等语言和空间联合条件。 |
| 目标级实例提取 | 未完整实现 | 当前主要输出语义掩膜，尚缺实例拆分、目标级掩膜和检测后处理。 |
| ArcGIS `.dlpk` 推理 | 未实现 | 本地已有检查点驱动的 LULC 滑窗推理和 ArcGIS-style 农作物演示，但没有 GeoVLM `.dlpk` 或 ArcPy 兼容推理。 |

## 三、现有代码基础

### 3.1 可以复用的部分

1. **遥感视觉编码器与第二骨干支持**

   当前 Prithvi-100M 模型能够将六通道遥感影像编码为全局 CLS 特征或空间 patch token，可以继续作为 GeoVLM 的视觉分支。最新代码还通过 backbone factory 加入了 SatMAE 兼容实现和第二骨干 EuroSAT 验证，为后续比较不同视觉编码器提供了基础。

2. **GeoAdapter 传感器通道适配**

   GeoAdapter 可以将 RGB、多光谱或其他通道配置映射到 Prithvi 所需的六通道输入空间，适合继续承担跨传感器和跨模态输入桥接。

3. **Houlsby 参数高效适配**

   Paper12 的实验表明，Houlsby adapter 是当前 Prithvi-100M 设置中最可靠的参数高效适配方法。该模块可以用于新 GeoVLM 的视觉编码分支微调。

4. **分类和语义分割训练框架**

   当前训练器已经支持单标签分类、多标签分类和语义分割。分割头同时提供线性解码器和浅层 `conv_lite` 解码器，可作为后续多任务训练框架的基础。

5. **GIS 栅格处理与 Model Hub 基础设施**

   现有系统已经具备栅格输入检查、滑窗切片、本地 LULC 检查点推理、结果拼接、GeoTIFF/GeoJSON/CSV 输出和模型资产就绪性检查。ArcGIS-style 农作物模型目前仍是确定性产品契约演示，真实 TerraTorch 推理适配器尚未完成本地验证。

### 3.2 当前缺失的核心组件

1. **文本 tokenizer、文本编码器或大语言模型。**
2. **视觉 token 与语言 token 的跨模态对齐模块。**
3. **支持图像描述和问答的语言生成解码器。**
4. **语言条件化掩膜解码器以及 `[SEG]` 一类分割信号。**
5. **面向指代分割的空间关系建模。**
6. **实例分割、目标拆分、计数与矢量化后处理。**
7. **GeoVLM 多任务指令数据和联合训练流程。**
8. **ArcGIS EMD 推理定义、Python Raster Function 和 `.dlpk` 打包。**

## 四、需要优先修复的视觉骨干问题

当前本地 `PrithviBackbone` 包装实现包含 patch embedding、CLS token 和 Transformer blocks，但没有加入或加载位置编码。模型能够按照原始 token 顺序恢复规则网格并完成固定类别分割，却缺少可靠理解绝对位置和空间关系的机制。SatMAE 兼容骨干应单独检查其位置嵌入加载和插值行为，不能用第二骨干实验替代这一审计。

这会直接影响以下提示：

- “分割图像顶部的飞机”；
- “水体附近是否存在道路”；
- “提取左下角最大的建筑”；
- “统计跑道两侧的飞机数量”。

在接入语言模型之前，应先恢复与 Prithvi 检查点一致的位置编码，并验证空间 token、输入尺寸变化和位置插值的正确性。对于小目标、道路和精细边界，还需要增加多尺度特征或特征金字塔。最新 `conv_lite` 解码器已经明显改善固定类别分割，但它仍是单尺度、非语言条件化的浅层解码器，不能替代 GeoVLM 所需的 grounded pixel decoder。

## 五、推荐实施路线

### 阶段一：实现提示词驱动分割 MVP

目标是优先实现最接近 GIS 生产需求的功能，例如：

- `segment buildings`；
- `segment roads`；
- `segment water bodies`。

推荐结构：

1. 使用 Prithvi-100M 作为遥感视觉编码器；
2. 使用 GeoAdapter 处理 RGB 和多光谱通道差异；
3. 使用 Houlsby adapter 微调视觉骨干；
4. 接入 SigLIP 或 CLIP 文本编码器；
5. 增加图文交叉注意力；
6. 使用 SAM、Mask2Former 或同类结构构建语言条件化掩膜解码器；
7. 输出带地理参考的二值掩膜和矢量面。

这一阶段可以实现自然语言类别提示下的语义分割，但还不等同于完整的图像描述和视觉问答模型。

### 阶段二：升级为完整 GeoVLM

在阶段一基础上接入 Qwen-VL、LLaVA 或同类视觉语言模型，增加以下能力：

1. 图像描述；
2. 场景分类和候选类别回答；
3. 遥感视觉问答；
4. 目标计数；
5. 指代表达分割；
6. 通过 `[SEG]` token 联动语言输出和像素解码器。

训练数据需要同时覆盖影像描述、问答、计数、类别标签、语义掩膜和指代掩膜。仅使用 EuroSAT、BigEarthNet、LandCover.ai 和 LoveDA 的固定标签数据不足以训练完整语言能力，还需要构建遥感图文指令数据集。

### 阶段三：GIS 产品化与 ArcGIS 集成

1. 实现大影像滑窗、重叠区域融合和批量推理；
2. 实现实例拆分、连通域处理、面积过滤和矢量化；
3. 保持投影、分辨率、范围和像元对齐；
4. 对接 Detect Objects Using Deep Learning；
5. 对接 Classify Pixels Using Deep Learning；
6. 对接 Classify Objects Using Deep Learning；
7. 编写 EMD 推理定义并生成真实 `.dlpk`；
8. 在 ArcGIS Pro 中验证文本提示、输出图层和空间精度。

## 六、当前 GIS 部署状态

当前 Model Hub 中的 `lulc_6class_prithvi_houlsby` 已配置本地检查点，并支持三波段 GeoTIFF 的滑窗神经网络推理、结果拼接、分类 GeoTIFF、GeoJSON、CSV 和预览输出。这说明 Paper12 相关模型已经具备从实验代码走向本地 GIS 推理服务的实际基础。

但是，`prithvi_crop_classification_arcgis_style` 仍标记为 `demo_only`，其上传栅格运行模式明确不加载真实 Prithvi 农作物检查点；真实 TerraTorch 路径目前只完成权重和依赖检查，尚未完成推理适配器验证。整个仓库也没有 GeoVLM 文本提示接口、EMD 推理定义或 `.dlpk` 兼容声明。因此，本地任务专用 LULC 推理可用，并不等于 ArcGIS GeoVLM 已经实现。

## 七、Paper12 实验结果的作用边界

Paper12 在 LandCover.ai 上的固定六类别分割实验中，Houlsby adapter 使用原始线性解码器达到约 `0.641 mIoU`。最新解码器容量消融中，`conv_lite` 将 Houlsby 提升到约 `0.7246 mIoU`，同时将线性探测和 LoRA 提升到约 `0.6539` 和 `0.6547 mIoU`。这证明了 Prithvi 空间 token、Houlsby adapter 和更强任务头可以承担固定类别密集预测，也表明绝对精度同时受适配器和解码器容量影响。

但该结果不能直接证明以下能力已经实现：

- 开放词汇识别；
- 未见类别的零样本分割；
- 自然语言空间推理；
- 图像描述或视觉问答；
- 提示词变化时的稳定泛化；
- ArcGIS GeoVLM 的整体精度和部署兼容性。

SatMAE 兼容的第二骨干结果扩大了 Paper12 的视觉 PEFT 证据范围，但仍然没有覆盖图文对齐或语言生成。Paper12 中关于 LoRA 的主要机制结论也限定于 Prithvi-100M 的融合 QKV 视觉骨干。未来在语言模型分支使用 LoRA 或 QLoRA 时，应重新检查目标层、实际可训练参数和模型架构，不能将视觉骨干上的负面结果直接推广到所有语言模型。

## 八、最终建议

最合理的第一阶段产品目标不是一次性复制 ArcGIS GeoVLM 的全部能力，而是先完成一个真实可验证的“遥感文本提示分割模型”：支持建筑、道路和水体提示，输出地理配准二值掩膜与矢量图层。

在该 MVP 达到可靠精度后，再加入语言生成、视觉问答、计数和指代分割。这样能够最大程度复用 Paper12 已验证的 Prithvi、GeoAdapter 和 Houlsby 资产，同时把技术风险集中在图文对齐和提示词掩膜解码两个核心问题上。
