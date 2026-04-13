# GeoAdapter 系统验证指南

本指南帮助你在本地逐步验证 GeoAdapter 系统的各项功能。

## 前置条件

你已经具备以下条件（无需额外准备）：

- Python 环境：`D:\adk\.venv\Scripts\python.exe` (Python 3.13.7)
- Prithvi-100M 权重：`data/weights/prithvi/Prithvi_100M.pt` (432.7 MB, 已存在)
- ZY3 样本数据：`data/raw_samples/` 下的 4 波段 L1A 影像 (619 MB, 已存在, 带 RPC)
- PostgreSQL 数据库：`119.3.175.198` (已配置在 .env)

## 验证 1：单元测试（2 分钟）

```bash
cd D:\adk\AlphaEarth-System
D:\adk\.venv\Scripts\python.exe -m pytest tests/ -v
```

预期结果：37 passed，约 12 秒。如果全部通过，说明 geoadapter 包的核心模块（adapters、backbone、heads、trainer、evaluator）都正常。

## 验证 2：GeoAdapter 核心功能（3 分钟）

打开命令行，逐段运行以下 Python 代码：

```bash
cd D:\adk\AlphaEarth-System
D:\adk\.venv\Scripts\python.exe
```

### 2.1 验证 GeoAdapter 三层适配器

```python
import torch
from geoadapter.adapters.geo_adapter import GeoAdapter

# 模拟 GF-2 四波段输入 → 映射到 Prithvi 的 6 波段
adapter = GeoAdapter(in_channels=4, out_channels=6)
x = torch.randn(1, 4, 128, 128)  # 1张图, 4波段, 128x128
out = adapter(x)
print(f"输入: {x.shape} → 输出: {out.shape}")
print(f"可训练参数: {sum(p.numel() for p in adapter.parameters() if p.requires_grad)}")
# 预期: 输入 [1,4,128,128] → 输出 [1,6,128,128], 参数约 147
```

### 2.2 验证 Prithvi-100M 加载真实权重

```python
from geoadapter.models.prithvi import PrithviBackbone

# 加载真实的 Prithvi-100M 权重（432 MB）
model = PrithviBackbone(
    pretrained=True,
    checkpoint_path="data/weights/prithvi/Prithvi_100M.pt"
)
print(f"Backbone 参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"Transformer 层数: {len(model.blocks)}")
print(f"所有参数已冻结: {all(not p.requires_grad for p in model.parameters())}")

# 前向传播测试
x = torch.randn(1, 6, 64, 64)
features = model(x)
print(f"输入: {x.shape} → 特征: {features.shape}")
# 预期: 86,237,184 参数, 12 层, 全部冻结, 输出 [1, 768]
```

### 2.3 验证完整管线：GeoAdapter + Prithvi + 分类 Head

```python
from geoadapter.adapters.geo_adapter import GeoAdapter
from geoadapter.models.prithvi import PrithviBackbone
from geoadapter.models.heads import ClassificationHead
from geoadapter.engine.trainer import PEFTTrainer

# 构建完整管线
backbone = PrithviBackbone(pretrained=True, checkpoint_path="data/weights/prithvi/Prithvi_100M.pt")
adapter = GeoAdapter(in_channels=4, out_channels=6)  # 4波段输入
head = ClassificationHead(in_dim=768, num_classes=10)
trainer = PEFTTrainer(backbone, adapter, head, lr=1e-3)

# 模拟一步训练
x = torch.randn(4, 4, 64, 64)   # batch=4, 4波段
y = torch.randint(0, 10, (4,))   # 10类标签
loss = trainer.train_step(x, y)
print(f"训练 loss: {loss:.4f}")

# 模拟预测
logits = trainer.predict(x)
preds = logits.argmax(dim=1)
print(f"预测结果: {preds.tolist()}")
# 预期: loss 约 2-4, 预测为 0-9 的整数列表
```

### 2.4 验证所有 5 种 PEFT 方法

```python
from geoadapter.adapters.lora import inject_lora
from geoadapter.adapters.bitfit import configure_bitfit
from geoadapter.adapters.houlsby import inject_houlsby_adapters

# LoRA
backbone_lora = PrithviBackbone(pretrained=False)
for block in backbone_lora.blocks:
    inject_lora(block, rank=8)
n_lora = sum(p.numel() for p in backbone_lora.parameters() if p.requires_grad)
print(f"LoRA 可训练参数: {n_lora:,}")

# BitFit
backbone_bitfit = PrithviBackbone(pretrained=False)
configure_bitfit(backbone_bitfit)
n_bitfit = sum(p.numel() for p in backbone_bitfit.parameters() if p.requires_grad)
print(f"BitFit 可训练参数: {n_bitfit:,}")

# Houlsby
backbone_houlsby = PrithviBackbone(pretrained=False)
for block in backbone_houlsby.blocks:
    inject_houlsby_adapters(block, bottleneck_dim=64)
n_houlsby = sum(p.numel() for p in backbone_houlsby.parameters() if p.requires_grad)
print(f"Houlsby 可训练参数: {n_houlsby:,}")

# GeoAdapter (已在 2.3 验证)
print(f"GeoAdapter 可训练参数: 147 (adapter) + 7690 (head)")
print(f"Linear Probe 可训练参数: 0 (adapter) + 7690 (head only)")
```

退出 Python：`exit()`

## 验证 3：Benchmark Runner（5 分钟）

### 3.1 Dry-run 查看实验矩阵

```bash
D:\adk\.venv\Scripts\python.exe -m geoadapter.bench.run_benchmark ^
    --config geoadapter/bench/configs/eurosat_default.yaml ^
    --dry-run
```

预期：打印 75 个实验组合（5 方法 × 5 模态 × 3 seeds）。

### 3.2 用合成数据跑 2 个 epoch 的快速验证

```bash
D:\adk\.venv\Scripts\python.exe -m geoadapter.bench.run_benchmark ^
    --config geoadapter/bench/configs/eurosat_default.yaml ^
    --epochs 2 ^
    --output results_smoke.json
```

预期：每个实验打印 loss 值，最终生成 `results_smoke.json`。因为没有真实 EuroSAT 数据，会自动回退到合成数据（"Dataset not available, using synthetic data"）。这是正常的——验证的是管线能跑通，不是结果准确性。

注意：75 个实验在 CPU 上可能需要 5-10 分钟。如果想快速验证，可以手动编辑 `geoadapter/bench/configs/eurosat_default.yaml`，临时把 `seeds` 改为 `[42]`，`methods` 只保留 `geoadapter` 和 `linear_probe`，这样只跑 10 个实验。

## 验证 4：用真实 ZY3 数据测试数据融合（5 分钟）

这一步验证 ae_backend 的 DataFusionPipeline 能否处理你的 ZY3 私有影像。

```bash
D:\adk\.venv\Scripts\python.exe
```

```python
import sys
sys.path.insert(0, "ae_backend")
import rasterio
import torch
import numpy as np

# 读取真实 ZY3 影像
tif_path = "data/raw_samples/zy302a_mux_004562_878150_20170326104452_01_sec_0004_1703290518/zy302a_mux_004562_878150_20170326104452_01_sec_0004_1703290518.tif"

with rasterio.open(tif_path) as src:
    print(f"CRS: {src.crs}")
    print(f"尺寸: {src.shape}")
    print(f"波段数: {src.count}")
    print(f"有 RPC 参数: {src.rpcs is not None}")

    # 读取左上角 128x128 区域
    from rasterio.windows import Window
    window = Window(0, 0, 128, 128)
    patch = src.read(window=window)  # [4, 128, 128]
    print(f"切片形状: {patch.shape}, 数据类型: {patch.dtype}")
    print(f"像素值范围: [{patch.min()}, {patch.max()}]")

# 归一化 + 转 tensor
patch = np.clip(patch.astype(np.float32), 0, 10000)
patch = np.log1p(patch) / 10.0
mean = np.mean(patch, axis=(1,2), keepdims=True)
std = np.std(patch, axis=(1,2), keepdims=True) + 1e-6
patch = (patch - mean) / std
tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)  # [1, 4, 128, 128]
print(f"归一化后 tensor: {tensor.shape}, 范围: [{tensor.min():.2f}, {tensor.max():.2f}]")

# 通过 GeoAdapter + Prithvi 提取特征
from geoadapter.adapters.geo_adapter import GeoAdapter
from geoadapter.models.prithvi import PrithviBackbone

adapter = GeoAdapter(in_channels=4, out_channels=6)
backbone = PrithviBackbone(pretrained=True, checkpoint_path="data/weights/prithvi/Prithvi_100M.pt")

adapted = adapter(tensor)
print(f"GeoAdapter 输出: {adapted.shape}")  # [1, 6, 128, 128]

features = backbone(adapted)
print(f"Prithvi 特征: {features.shape}")  # [1, 768]
print(f"特征 L2 范数: {torch.norm(features).item():.4f}")
print("真实 ZY3 数据 → GeoAdapter → Prithvi 管线验证通过！")
```

预期：ZY3 的 4 波段 uint16 数据被成功读取、归一化、通过 GeoAdapter 映射到 6 通道、再通过 Prithvi 提取出 768 维特征。

## 验证 5：ae_backend 平台集成（3 分钟）

验证 ae_backend 的 trainer 模块已正确使用 geoadapter。

```bash
D:\adk\.venv\Scripts\python.exe
```

```python
import sys
sys.path.insert(0, "ae_backend")
from app.services.trainer import PrithviAlphaEarthEncoder
import torch

# 使用真实 Prithvi 权重
model = PrithviAlphaEarthEncoder(
    weight_path="data/weights/prithvi/Prithvi_100M.pt",
    in_channels=5,
    hidden_dim=64
)

x = torch.randn(2, 5, 128, 128)
rec, z = model(x)
print(f"重构输出: {rec.shape}")   # [2, 5, 128, 128]
print(f"Embedding: {z.shape}")     # [2, 64]
print(f"Backbone 类型: {type(model.backbone).__name__}")
print(f"Adapter 类型: {type(model.modality_adapter).__name__}")
print("ae_backend 集成验证通过！")
```

预期：`PrithviAlphaEarthEncoder` 内部使用 `PrithviBackbone` 和 `GeoAdapter`，输出形状与原来一致。

## 验证 6：前端平台启动（可选，需要数据库）

如果你的 PostgreSQL 数据库（119.3.175.198）可以连接：

```bash
cd D:\adk\AlphaEarth-System\ae_backend
D:\adk\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8085 --reload
```

然后浏览器打开 `http://localhost:8085`，应该能看到前端界面。

如果数据库不可用，这一步可以跳过——核心的 geoadapter 功能不依赖数据库。

---

## 验证结果汇总

完成以上步骤后，你应该确认了：

| 验证项 | 预期结果 |
|---|---|
| 37 个单元测试 | 全部 PASS |
| GeoAdapter 三层适配器 | 4ch→6ch 映射正常，~147 参数 |
| Prithvi-100M 真实权重加载 | 86M 参数，12 层，全冻结 |
| 完整训练管线 | train_step 返回 loss，predict 返回 logits |
| 5 种 PEFT 方法 | 各自参数量不同，均可注入 |
| Benchmark runner | 75 实验矩阵，合成数据可跑通 |
| 真实 ZY3 数据 | 4 波段 → GeoAdapter → Prithvi → 768 维特征 |
| ae_backend 集成 | PrithviAlphaEarthEncoder 使用 geoadapter 内部模块 |

## 下一步：在 Colab 上跑真实 Benchmark

本地验证通过后，下一步是在 Colab Pro+ A100 上用真实数据集（EuroSAT）跑完整实验。需要：

1. 将本项目上传到 Google Drive 或 GitHub
2. 在 Colab 中 `pip install -e /content/AlphaEarth-System/[bench]`
3. EuroSAT 数据集会通过 torchgeo 自动下载（约 2.5 GB）
4. 运行 `notebooks/01_benchmark_eurosat.py` 中的代码
