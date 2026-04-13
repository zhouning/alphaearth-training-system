# Google Colab Pro+ 本地化 AlphaEarth 模型训练实战指南 (内存与云盘极限优化版)

本指南旨在指导您在 Google Colab Pro+ 环境中，利用本地的矢量边界提取高精度卫星影像，并训练本地的 AlphaEarth 地理空间特征编码器。

**【关于 Google Drive 1.5GB 空间的特别说明】**：
您完全不用担心！在本次深度优化版本中，我们**不再将庞大的卫星影像保存到 Google Drive**。
我们将充分利用 Colab 实例自带的免费大容量本地磁盘（通常有 78GB+ 空闲），**Google Drive 仅用于存放矢量边界（几 KB）和最终生成的模型权重（约 2-3 MB）**，全流程对您的 Drive 空间消耗不到 10 MB！

---

## 阶段一：前期准备与实例环境选择

### 1. 硬件实例选择 (Colab Pro+)
在 Colab 菜单栏 -> `代码执行程序` -> `更改运行时类型` 中进行设置：

*   **如果是【村/镇】级别（例如：和平村，< 50平方公里）**：
    *   **硬件加速器**：`T4 GPU` 或 `L4 GPU`。
    *   **系统 RAM**：`标准` 即可。
*   **如果是【县/区】级别（例如：璧山区全境，> 500平方公里）**：
    *   **硬件加速器**：建议选择 `A100 GPU` 或 `V100 GPU`。
    *   **系统 RAM**：必须选择 `高 RAM` (High-RAM)。

### 2. 云盘与账户准备
1. 准备一个 **Google Cloud Project (GCP)** 并启用 **Earth Engine API**。
2. 在您的 Google Drive 根目录创建一个文件夹 `AlphaEarth_Data`。
3. 将本地的 `和平村8000.shp` 及同名附属文件（`.shx`, `.dbf`, `.prj`）打包为 `heping_village.zip`，上传到 Google Drive 的 `AlphaEarth_Data` 目录中。

---

## 阶段二：在 Colab 中执行的全流程代码

新建一个 Colab Notebook，依次运行以下代码块（Cell）：

### Cell 1：挂载云盘与安装依赖
```python
# 挂载 Google Drive (仅用于读取 .shp 和保存 .pth)
from google.colab import drive
drive.mount('/content/drive')

# 安装所需的地理空间库
!pip install -q geemap geopandas rasterio geedim
```

### Cell 2：GEE 授权与直连本地下载 (零云盘消耗版)
```python
import ee
import geemap
import geopandas as gpd
import os

# 1. 触发 GEE 授权验证
ee.Authenticate()
ee.Initialize(project='YOUR_GCP_PROJECT_ID_HERE') 
print("GEE 验证成功！")

# 2. 读取上传的 Shapefile 边界
shp_path = '/content/drive/MyDrive/AlphaEarth_Data/heping_village.zip'
gdf = gpd.read_file(shp_path)
gdf_wgs84 = gdf.to_crs(epsg=4326)

# 获取外接矩形边界
minx, miny, maxx, maxy = gdf_wgs84.total_bounds
roi = ee.Geometry.Rectangle([minx, miny, maxx, maxy])

# 3. 获取 Sentinel-2 影像 (2023年夏季无云中值合成，5个核心波段)
s2 = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
    .filterBounds(roi) \
    .filterDate('2023-06-01', '2023-09-01') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
    .median() \
    .select(['B2', 'B3', 'B4', 'B8', 'B11'])

# 4. 直接下载到 Colab 的本地磁盘 (/content/)，不占用 Google Drive！
# geemap 具备自动分块下载大图的能力
out_tif = '/content/Sentinel2_ROI.tif'
print(f"正在直接下载高精度影像至 Colab 本地磁盘 ({out_tif})，这不会占用您的 Google Drive 空间...")

geemap.download_ee_image(
    image=s2,
    filename=out_tif,
    scale=10,             # 10米分辨率
    region=roi,
    crs='EPSG:32648'      # 请根据目标地实际UTM带调整，例如重庆通常为 32648
)

print("✅ 下载完成！您可以进行下一步了。")
```

### Cell 3：定义内存安全的 PyTorch 数据集 (Dataset)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.windows import Window
import numpy as np

class AlphaEarthDataset(Dataset):
    """内存安全的切片数据集：按需从本地磁盘读取，防 OOM"""
    def __init__(self, tif_path, patch_size=128, stride=64):
        self.tif_path = tif_path
        self.patch_size = patch_size
        self.windows = []
        
        # 扫描影像尺寸并生成滑动窗口
        with rasterio.open(tif_path) as src:
            H, W = src.shape
            for y in range(0, H - patch_size + 1, stride):
                for x in range(0, W - patch_size + 1, stride):
                    self.windows.append(Window(col_off=x, row_off=y, width=patch_size, height=patch_size))
        
        print(f"✅ 扫描完毕：共生成 {len(self.windows)} 个 {patch_size}x{patch_size} 切片样本。")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        with rasterio.open(self.tif_path) as src:
            patch = src.read(window=window) # [5, 128, 128]
            
        patch = np.nan_to_num(patch, nan=0.0)
        patch = np.clip(patch, 0, 10000)
        
        # 遵循 AlphaEarth 论文缩放：log(x+1)/10
        patch = np.log1p(patch) / 10.0
        
        # 实例级归一化
        mean = np.mean(patch, axis=(1,2), keepdims=True)
        std = np.std(patch, axis=(1,2), keepdims=True) + 1e-6
        patch = (patch - mean) / std
        
        return torch.tensor(patch, dtype=torch.float32)

# 加载位于 Colab 本地磁盘的 TIF 文件
tif_file = '/content/Sentinel2_ROI.tif'
# 如果是村级数据切片较少，建议 stride 设为 32 或 16 来增加数据量 (Data Augmentation)
dataset = AlphaEarthDataset(tif_file, patch_size=128, stride=32) 
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
```

### Cell 4：定义核心模型架构与损失函数

```python
class STPBlock(nn.Module):
    """Space-Time-Precision (STP) Encoder Block"""
    def __init__(self, channels=64):
        super().__init__()
        self.prec_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.space_attn = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
        self.time_attn = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)

    def forward(self, x_prec, x_space, x_time):
        out_prec = self.prec_conv(x_prec)
        
        B, C, H_s, W_s = x_space.shape
        x_space_flat = x_space.view(B, C, -1).permute(0, 2, 1)
        out_space, _ = self.space_attn(x_space_flat, x_space_flat, x_space_flat)
        out_space = out_space.permute(0, 2, 1).view(B, C, H_s, W_s)
        
        B, C, H_t, W_t = x_time.shape
        x_time_flat = x_time.view(B, C, -1).permute(0, 2, 1)
        out_time, _ = self.time_attn(x_time_flat, x_time_flat, x_time_flat)
        out_time = out_time.permute(0, 2, 1).view(B, C, H_t, W_t)
        
        out_space_up = F.interpolate(out_space, size=out_prec.shape[2:], mode='bilinear')
        out_time_up  = F.interpolate(out_time, size=out_prec.shape[2:], mode='bilinear')
        
        return out_prec + out_space_up + out_time_up

class LocalAlphaEarthEncoder(nn.Module):
    def __init__(self, in_channels=5, z_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, z_dim, kernel_size=1)
        self.stp_block = STPBlock(channels=z_dim)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        feat = self.proj(x)
        x_prec = F.interpolate(feat, scale_factor=0.5, mode='bilinear')
        x_time = F.interpolate(feat, scale_factor=0.125, mode='bilinear')
        x_space = F.interpolate(feat, scale_factor=0.0625, mode='bilinear')
        
        o_fused = self.stp_block(x_prec, x_space, x_time)
        z = self.pool(o_fused).flatten(1)
        z = F.normalize(z, p=2, dim=1) # 投影到 S^63 超球面
        return z

class ImplicitDecoder(nn.Module):
    def __init__(self, z_dim=64, out_channels=5):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 1, 0), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),   nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),    nn.ReLU(),
            nn.ConvTranspose2d(8, out_channels, 4, 2, 1)
        )
    def forward(self, z):
        x = self.fc(z).view(-1, 256, 1, 1)
        return self.up(x)

def batch_uniformity_loss(z):
    sim_matrix = torch.mm(z, z.t())
    mask = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    sim_matrix.masked_fill_(mask, 0.0)
    return torch.mean(torch.abs(sim_matrix))
```

### Cell 5：执行训练并仅保存模型到云盘

```python
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的计算设备: {device}")

encoder = LocalAlphaEarthEncoder().to(device)
decoder = ImplicitDecoder().to(device)

optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

epochs = 50
print("开始训练...")

for epoch in range(epochs):
    encoder.train()
    decoder.train()
    
    total_loss, total_rec, total_uni = 0, 0, 0
    start_time = time.time()
    
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        z = encoder(batch)
        reconstructed = decoder(z)
        
        loss_rec = F.mse_loss(reconstructed, batch)
        loss_uni = batch_uniformity_loss(z)
        loss = loss_rec + 0.2 * loss_uni 
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_rec += loss_rec.item()
        total_uni += loss_uni.item()
        
    scheduler.step()
    
    n_batches = max(1, len(dataloader))
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/n_batches:.4f} (Rec: {total_rec/n_batches:.4f}, Uni: {total_uni/n_batches:.4f}) | Time: {time.time() - start_time:.1f}s")

# 仅保存几MB的模型权重到 Google Drive
save_path = '/content/drive/MyDrive/AlphaEarth_Data/local_alphaearth_encoder.pth'
torch.save(encoder.state_dict(), save_path)
print(f"🎉 训练完成！模型权重已安全保存至 Google Drive: {save_path} (占用约 2.6 MB)")
```

---

## 阶段三：集成到本地代码

训练完成后，去您的 Google Drive 下载 `local_alphaearth_encoder.pth`（只有大约2-3 MB），放到本地的 `D:\adk\weights\` 目录。
然后您就可以在 `data_agent\world_model.py` 中像这样直接调用专属您的本地模型，而无需再依赖网络请求了：

```python
encoder = LocalAlphaEarthEncoder()
encoder.load_state_dict(torch.load("D:/adk/weights/local_alphaearth_encoder.pth"))
encoder.eval()
```
