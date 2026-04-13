# %% [markdown]
# # GeoAdapter Benchmark: EuroSAT (Complete Experiment)
#
# Runs the full 5-method x 5-modality benchmark on EuroSAT.
# Target: Google Colab Pro+ with A100 GPU.
#
# ## Setup (run once)
# ```
# !git clone https://github.com/zhouning/alphaearth-training-system.git /content/AlphaEarth-System
# %cd /content/AlphaEarth-System
# !pip install -e ".[bench]" -q
# ```
#
# ## Prithvi Weights
# Upload `Prithvi_100M.pt` (432 MB) to Google Drive at `MyDrive/weights/Prithvi_100M.pt`,
# or download from HuggingFace:
# ```
# !huggingface-cli download ibm-nasa-geospatial/Prithvi-EO-1.0-100M --local-dir /content/prithvi_weights
# ```

# %% Cell 1: Imports and Config
import torch
import torch.nn as nn
import numpy as np
import json
import time
import os
from pathlib import Path
from torch.utils.data import DataLoader
from collections import defaultdict

from geoadapter.models.prithvi import PrithviBackbone
from geoadapter.adapters.geo_adapter import GeoAdapter
from geoadapter.adapters.zero_pad import ZeroPadAdapter
from geoadapter.adapters.lora import inject_lora
from geoadapter.adapters.bitfit import configure_bitfit
from geoadapter.adapters.houlsby import inject_houlsby_adapters
from geoadapter.models.heads import ClassificationHead
from geoadapter.engine.trainer import PEFTTrainer
from geoadapter.engine.evaluator import compute_classification_metrics
from geoadapter.data.datasets import ModalityConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# --- CONFIGURE THESE ---
PRITHVI_WEIGHTS = "/content/prithvi_weights/Prithvi_100M.pt"  # Adjust path
EUROSAT_ROOT = "/content/data/eurosat"
RESULTS_DIR = "/content/results"
EPOCHS = 50
BATCH_SIZE = 64
SEEDS = [42, 123, 456]
NUM_CLASSES = 10
# --- END CONFIG ---

os.makedirs(RESULTS_DIR, exist_ok=True)

# %% Cell 2: Load EuroSAT Dataset
# torchgeo's EuroSAT: 13 bands, 64x64, 10 classes
from torchgeo.datasets import EuroSAT

print("Downloading EuroSAT (first run only, ~2.5 GB)...")
train_ds_full = EuroSAT(root=EUROSAT_ROOT, split="train", download=True)
test_ds_full = EuroSAT(root=EUROSAT_ROOT, split="test", download=True)
print(f"Train: {len(train_ds_full)}, Test: {len(test_ds_full)}")

# Check a sample
sample = train_ds_full[0]
print(f"Sample image shape: {sample['image'].shape}")  # [13, 64, 64]
print(f"Sample label: {sample['label']}")

# %% Cell 3: Dataset Wrapper with Modality Selection
class EuroSATModality(torch.utils.data.Dataset):
    """Wraps EuroSAT with band selection for different modality configs."""

    def __init__(self, base_ds, band_indices=None, c_in=None, normalize=True):
        self.base = base_ds
        self.band_indices = band_indices
        self.c_in = c_in
        self.normalize = normalize

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        img = sample["image"].float()  # [13, 64, 64]

        # Band selection
        if self.band_indices is not None:
            img = img[self.band_indices]  # [C_selected, 64, 64]

        # For SAR simulation: use bands 0,1 and treat as VV/VH
        if self.c_in is not None and img.shape[0] != self.c_in:
            if self.c_in < img.shape[0]:
                img = img[:self.c_in]
            else:
                pad = torch.zeros(self.c_in - img.shape[0], img.shape[1], img.shape[2])
                img = torch.cat([img, pad], dim=0)

        # Normalize: per-channel z-score
        if self.normalize:
            mean = img.mean(dim=(-2, -1), keepdim=True)
            std = img.std(dim=(-2, -1), keepdim=True) + 1e-6
            img = (img - mean) / std

        return img, sample["label"]


# Define modality configs matching the design spec
MODALITIES = {
    "s2_full": {"indices": list(range(10)), "c_in": 10, "desc": "Sentinel-2 10 bands"},
    "rgb":     {"indices": [3, 2, 1],       "c_in": 3,  "desc": "RGB (B4,B3,B2)"},
    "rgb_sar": {"indices": [3, 2, 1, 0],    "c_in": 4,  "desc": "RGB + simulated SAR"},
    "gf2":     {"indices": [3, 2, 1, 7],    "c_in": 4,  "desc": "GF-2 (B,G,R,NIR)"},
    "sar_only":{"indices": [0, 1],          "c_in": 2,  "desc": "Simulated SAR 2-band"},
}

# %% Cell 4: Experiment Runner
def build_model(method_name, modality_name, seed):
    """Build backbone + adapter + head for one experiment."""
    torch.manual_seed(seed)
    cfg = MODALITIES[modality_name]

    # Fresh backbone each time (frozen weights)
    backbone = PrithviBackbone(pretrained=True, checkpoint_path=PRITHVI_WEIGHTS)

    # Apply PEFT to backbone
    if method_name == "bitfit":
        configure_bitfit(backbone)
    elif method_name == "lora_r8":
        for block in backbone.blocks:
            inject_lora(block, rank=8)
    elif method_name == "houlsby":
        for block in backbone.blocks:
            inject_houlsby_adapters(block, bottleneck_dim=64)

    # Build adapter
    if method_name == "geoadapter":
        adapter = GeoAdapter(in_channels=cfg["c_in"], out_channels=6)
    else:
        adapter = ZeroPadAdapter(in_channels=cfg["c_in"], out_channels=6)

    head = ClassificationHead(in_dim=768, num_classes=NUM_CLASSES)

    # Differential LR: backbone PEFT params get lower LR
    lr_peft = 1e-4 if method_name in ("lora_r8", "houlsby", "bitfit") else None

    trainer = PEFTTrainer(
        backbone, adapter, head,
        lr=1e-3, lr_peft=lr_peft,
        epochs=EPOCHS, device=device,
    )

    # Count trainable params
    n_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    n_params += sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    n_params += sum(p.numel() for p in head.parameters())

    return trainer, n_params


def run_experiment(method_name, modality_name, seed):
    """Run one full experiment: train + evaluate. Returns metrics dict."""
    cfg = MODALITIES[modality_name]
    tag = f"{method_name}|{modality_name}|seed={seed}"

    # Build datasets
    train_ds = EuroSATModality(train_ds_full, cfg["indices"], cfg["c_in"])
    test_ds = EuroSATModality(test_ds_full, cfg["indices"], cfg["c_in"])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)

    # Build model
    trainer, n_params = build_model(method_name, modality_name, seed)
    print(f"\n[{tag}] trainable_params={n_params:,}, train={len(train_ds)}, test={len(test_ds)}")

    # Train
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in train_loader:
            loss = trainer.train_step(batch_x, batch_y)
            epoch_loss += loss
            n_batches += 1
        trainer.step_scheduler()
        avg_loss = epoch_loss / max(n_batches, 1)
        if epoch % 10 == 0 or epoch == EPOCHS:
            print(f"  [{tag}] Epoch {epoch}/{EPOCHS} loss={avg_loss:.4f}")
    train_time = time.time() - t0

    # Evaluate
    all_preds, all_labels = [], []
    for batch_x, batch_y in test_loader:
        logits = trainer.predict(batch_x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_y.numpy())

    metrics = compute_classification_metrics(np.array(all_labels), np.array(all_preds))
    metrics.update({
        "method": method_name,
        "modality": modality_name,
        "seed": seed,
        "trainable_params": n_params,
        "train_time_sec": round(train_time, 1),
        "epochs": EPOCHS,
    })
    print(f"  [{tag}] OA={metrics['overall_accuracy']:.4f} F1={metrics['macro_f1']:.4f} time={train_time:.0f}s")
    return metrics

# %% Cell 5: Run Full Benchmark
METHODS = ["linear_probe", "bitfit", "lora_r8", "houlsby", "geoadapter"]

all_results = []
total = len(METHODS) * len(MODALITIES) * len(SEEDS)
done = 0

print(f"Starting {total} experiments ({len(METHODS)} methods x {len(MODALITIES)} modalities x {len(SEEDS)} seeds)")
print(f"Estimated time: {total * 3:.0f}-{total * 6:.0f} minutes on A100")
print("=" * 70)

for method in METHODS:
    for modality in MODALITIES:
        for seed in SEEDS:
            result = run_experiment(method, modality, seed)
            all_results.append(result)
            done += 1
            print(f"--- Progress: {done}/{total} ({100*done/total:.0f}%) ---")

            # Save intermediate results after each experiment
            with open(f"{RESULTS_DIR}/eurosat_results.json", "w") as f:
                json.dump(all_results, f, indent=2)

print("\n" + "=" * 70)
print(f"DONE! {total} experiments completed. Results saved to {RESULTS_DIR}/eurosat_results.json")

# %% Cell 6: Aggregate Results Table
import pandas as pd

df = pd.DataFrame(all_results)

# Pivot: mean OA per method x modality
pivot_oa = df.pivot_table(values="overall_accuracy", index="method", columns="modality", aggfunc="mean")
pivot_f1 = df.pivot_table(values="macro_f1", index="method", columns="modality", aggfunc="mean")
pivot_std = df.pivot_table(values="overall_accuracy", index="method", columns="modality", aggfunc="std")

print("\n=== Overall Accuracy (mean) ===")
print(pivot_oa.round(4).to_string())
print("\n=== Macro F1 (mean) ===")
print(pivot_f1.round(4).to_string())
print("\n=== OA Std Dev ===")
print(pivot_std.round(4).to_string())

# Params per method
params = df.groupby("method")["trainable_params"].first()
print("\n=== Trainable Parameters ===")
print(params.to_string())

# Save tables
pivot_oa.round(4).to_csv(f"{RESULTS_DIR}/table1_oa.csv")
pivot_f1.round(4).to_csv(f"{RESULTS_DIR}/table1_f1.csv")
print(f"\nTables saved to {RESULTS_DIR}/")

# %% Cell 7: Generate Paper Figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Fig.3: Heatmap (method x modality)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(pivot_oa, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[0], vmin=0.3, vmax=1.0)
axes[0].set_title("Overall Accuracy")
axes[0].set_ylabel("PEFT Method")
axes[0].set_xlabel("Modality Configuration")

sns.heatmap(pivot_f1, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[1], vmin=0.3, vmax=1.0)
axes[1].set_title("Macro F1 Score")
axes[1].set_ylabel("")
axes[1].set_xlabel("Modality Configuration")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/fig3_heatmap.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{RESULTS_DIR}/fig3_heatmap.pdf", bbox_inches="tight")
print("Fig.3 saved")

# Fig.4: Pareto (params vs accuracy)
fig, ax = plt.subplots(figsize=(8, 6))
method_colors = {"linear_probe": "#1f77b4", "bitfit": "#ff7f0e", "lora_r8": "#2ca02c",
                 "houlsby": "#d62728", "geoadapter": "#9467bd"}

for method in METHODS:
    sub = df[df["method"] == method]
    mean_oa = sub["overall_accuracy"].mean()
    n_params = sub["trainable_params"].iloc[0]
    ax.scatter(n_params, mean_oa, s=150, c=method_colors[method], label=method, zorder=5)
    ax.annotate(method, (n_params, mean_oa), textcoords="offset points",
                xytext=(8, 8), fontsize=9)

ax.set_xscale("log")
ax.set_xlabel("Trainable Parameters (log scale)")
ax.set_ylabel("Mean Overall Accuracy")
ax.set_title("Efficiency vs Accuracy Pareto")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/fig4_pareto.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{RESULTS_DIR}/fig4_pareto.pdf", bbox_inches="tight")
print("Fig.4 saved")

# Fig.5: Training time comparison
fig, ax = plt.subplots(figsize=(8, 5))
time_by_method = df.groupby("method")["train_time_sec"].mean().reindex(METHODS)
ax.barh(range(len(METHODS)), time_by_method.values, color=[method_colors[m] for m in METHODS])
ax.set_yticks(range(len(METHODS)))
ax.set_yticklabels(METHODS)
ax.set_xlabel("Mean Training Time (seconds)")
ax.set_title("Training Efficiency")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/fig5_time.png", dpi=300, bbox_inches="tight")
print("Fig.5 saved")

print(f"\nAll figures saved to {RESULTS_DIR}/")

# %% Cell 8: GeoAdapter Ablation Study
# Compare: Full GeoAdapter vs Proj+SE vs Proj-only vs Zero-pad
print("\n=== Ablation Study ===")

class GeoAdapterProjOnly(nn.Module):
    """Ablation: only Channel Projection (1x1 Conv), no SE or DWConv."""
    def __init__(self, c_in, c_out=6):
        super().__init__()
        self.in_channels = c_in
        self.out_channels = c_out
        self.proj = nn.Conv2d(c_in, c_out, kernel_size=1)
    def forward(self, x):
        return self.proj(x)

class GeoAdapterProjSE(nn.Module):
    """Ablation: Projection + SE Attention, no Spatial Refinement."""
    def __init__(self, c_in, c_out=6):
        super().__init__()
        self.in_channels = c_in
        self.out_channels = c_out
        self.proj = nn.Conv2d(c_in, c_out, kernel_size=1)
        mid = max(1, c_out // 2)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c_out, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, c_out), nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.proj(x)
        attn = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * attn

ablation_results = []
ablation_modality = "gf2"  # 4-band, the most interesting case
cfg_abl = MODALITIES[ablation_modality]

for ablation_name, adapter_cls in [
    ("zero_pad", lambda c: ZeroPadAdapter(in_channels=c, out_channels=6)),
    ("proj_only", lambda c: GeoAdapterProjOnly(c, 6)),
    ("proj_se", lambda c: GeoAdapterProjSE(c, 6)),
    ("full_geoadapter", lambda c: GeoAdapter(in_channels=c, out_channels=6)),
]:
    for seed in SEEDS:
        torch.manual_seed(seed)
        backbone = PrithviBackbone(pretrained=True, checkpoint_path=PRITHVI_WEIGHTS)
        adapter = adapter_cls(cfg_abl["c_in"])
        head = ClassificationHead(in_dim=768, num_classes=NUM_CLASSES)
        trainer = PEFTTrainer(backbone, adapter, head, lr=1e-3, epochs=EPOCHS, device=device)

        train_ds = EuroSATModality(train_ds_full, cfg_abl["indices"], cfg_abl["c_in"])
        test_ds = EuroSATModality(test_ds_full, cfg_abl["indices"], cfg_abl["c_in"])
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        n_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        print(f"  Ablation: {ablation_name}, seed={seed}, adapter_params={n_params}")

        for epoch in range(1, EPOCHS + 1):
            for bx, by in train_loader:
                trainer.train_step(bx, by)
            trainer.step_scheduler()

        all_preds, all_labels = [], []
        for bx, by in test_loader:
            logits = trainer.predict(bx)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(by.numpy())

        m = compute_classification_metrics(np.array(all_labels), np.array(all_preds))
        m["ablation"] = ablation_name
        m["seed"] = seed
        m["adapter_params"] = n_params
        ablation_results.append(m)
        print(f"    OA={m['overall_accuracy']:.4f} F1={m['macro_f1']:.4f}")

# Save and plot ablation
with open(f"{RESULTS_DIR}/ablation_results.json", "w") as f:
    json.dump(ablation_results, f, indent=2)

abl_df = pd.DataFrame(ablation_results)
abl_mean = abl_df.groupby("ablation")["overall_accuracy"].mean()
abl_std = abl_df.groupby("ablation")["overall_accuracy"].std()
order = ["zero_pad", "proj_only", "proj_se", "full_geoadapter"]
abl_mean = abl_mean.reindex(order)
abl_std = abl_std.reindex(order)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(range(4), abl_mean.values, yerr=abl_std.values, capsize=5,
              color=["#cccccc", "#88bbdd", "#5599cc", "#9467bd"])
ax.set_xticks(range(4))
ax.set_xticklabels(["Zero-Pad", "Proj Only", "Proj+SE", "Full GeoAdapter"])
ax.set_ylabel("Overall Accuracy")
ax.set_title(f"GeoAdapter Ablation Study (modality: {ablation_modality})")
for i, (v, s) in enumerate(zip(abl_mean.values, abl_std.values)):
    ax.text(i, v + s + 0.005, f"{v:.3f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/fig5_ablation.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{RESULTS_DIR}/fig5_ablation.pdf", bbox_inches="tight")
print("Ablation figure saved")

# %% Cell 9: Channel Attention Visualization
print("\n=== Channel Attention Weights ===")
from geoadapter.viz import extract_channel_attention_weights

for mod_name, cfg in MODALITIES.items():
    adapter = GeoAdapter(in_channels=cfg["c_in"], out_channels=6)
    # Train briefly to get meaningful attention weights
    backbone = PrithviBackbone(pretrained=True, checkpoint_path=PRITHVI_WEIGHTS)
    head = ClassificationHead(in_dim=768, num_classes=NUM_CLASSES)
    trainer = PEFTTrainer(backbone, adapter, head, lr=1e-3, epochs=10, device=device)

    train_ds = EuroSATModality(train_ds_full, cfg["indices"], cfg["c_in"])
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    for epoch in range(10):
        for bx, by in loader:
            trainer.train_step(bx, by)

    weights = extract_channel_attention_weights(adapter.cpu())
    print(f"  {mod_name} ({cfg['desc']}): attention weights = {np.round(weights, 3)}")

print(f"\nExperiment complete! All results in {RESULTS_DIR}/")
