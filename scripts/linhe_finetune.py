"""Linhe PEFT fine-tuning: supervised classification on real labels.

Pipeline:
  RGB patch -> GeoAdapter(3->6) -> frozen Prithvi-100M -> Houlsby adapters -> ClassificationHead

Supports two modes:
  1. Real labels from _labels.parquet (default when --labels is provided or auto-detected)
  2. Pseudo-labels via KMeans (fallback when no label file exists)

Usage:
  python scripts/linhe_finetune.py --epochs 10 --batch-size 16
  python scripts/linhe_finetune.py --labels data/linhe_patches/_labels.parquet --epochs 10

Output:
  results/linhe/linhe_houlsby_rgb.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset, random_split

from geoadapter.adapters.geo_adapter import GeoAdapter
from geoadapter.adapters.houlsby import inject_houlsby_adapters
from geoadapter.adapters.bitfit import configure_bitfit
from geoadapter.adapters.zero_pad import ZeroPadAdapter
from geoadapter.engine.trainer import PEFTTrainer
from geoadapter.models.heads import ClassificationHead
from geoadapter.models.prithvi import PrithviBackbone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "linhe"
OUT.mkdir(parents=True, exist_ok=True)


class LinhePatchDataset(Dataset):
    def __init__(self, patch_paths: list[Path], labels: np.ndarray, modality: str = "rgb"):
        self.patch_paths = patch_paths
        self.labels = labels.astype(np.int64)
        self.modality = modality

    def __len__(self) -> int:
        return len(self.patch_paths)

    def __getitem__(self, idx: int):
        data = np.load(self.patch_paths[idx])
        if self.modality == "s2":
            x = data["s2"].astype(np.float32)
            if x.max() > 1.0:
                x = x / 10000.0
        else:
            x = data["rgb"].astype(np.float32) / 255.0
        y = self.labels[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def build_pseudo_labels(paths: list[Path], n_clusters: int = 8) -> np.ndarray:
    feats = []
    for p in paths:
        x = np.load(p)["rgb"].astype(np.float32) / 255.0
        # 3ch mean + std + simple vegetation-ish proxy from RGB
        mean = x.mean(axis=(1, 2))
        std = x.std(axis=(1, 2))
        r, g, b = mean
        exg = 2 * g - r - b
        feats.append(np.concatenate([mean, std, [exg]], axis=0))
    feats = np.stack(feats)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    return km.fit_predict(feats)


def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)


def find_default_ckpt(cli_value: str | None) -> str | None:
    if cli_value:
        return cli_value
    candidates = [
        ROOT / "data" / "prithvi_100m.pt",
        ROOT / "data" / "models" / "prithvi_100m.pt",
        ROOT / "data" / "weights" / "prithvi_100m.pt",
        ROOT / "data" / "weights" / "Prithvi_100M.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def load_real_labels(index_df: pd.DataFrame, labels_path: Path, modality: str = "rgb") -> tuple[list[Path], np.ndarray, list[str]]:
    labels_df = pd.read_parquet(labels_path)
    merged = index_df.merge(labels_df[["patch_path", "label"]], on="patch_path", how="inner")
    if modality == "s2" and "s2_patch_path" in merged.columns:
        merged = merged[merged["s2_patch_path"].notna()]
        paths = [ROOT / p for p in merged["s2_patch_path"].tolist()]
    else:
        paths = [ROOT / p for p in merged["patch_path"].tolist()]
    class_names = sorted(merged["label"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    labels = np.array([class_to_idx[c] for c in merged["label"].tolist()])
    return paths, labels, class_names


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.from_numpy(weights)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-index", default=str(ROOT / "data" / "linhe_patches" / "_index.parquet"))
    ap.add_argument("--labels", default=None, help="path to _labels.parquet; auto-detected if omitted")
    ap.add_argument("--modality", default="rgb", choices=["rgb", "s2"])
    ap.add_argument("--no-pretrain", action="store_true", help="use random init backbone (ablation)")
    ap.add_argument("--peft-method", default="houlsby", choices=["linear_probe", "bitfit", "houlsby"])
    ap.add_argument("--prithvi-ckpt", default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bottleneck-dim", type=int, default=64)
    ap.add_argument("--num-classes", type=int, default=None)
    ap.add_argument("--max-patches", type=int, default=0)
    args = ap.parse_args()

    idx = pd.read_parquet(args.patch_index)
    idx = idx[idx["modality"] == "rgb"].copy()
    if args.max_patches:
        idx = idx.head(args.max_patches)

    labels_path = Path(args.labels) if args.labels else Path(args.patch_index).parent / "_labels.parquet"
    use_real = labels_path.exists()

    if use_real:
        patch_paths, labels, class_names = load_real_labels(idx, labels_path, modality=args.modality)
        num_classes = len(class_names)
        print(f"[info] real labels: {len(labels)} samples, {num_classes} classes: {class_names}")
        class_weights = compute_class_weights(labels, num_classes)
    else:
        num_classes = args.num_classes or 8
        patch_paths = [ROOT / p for p in idx["patch_path"].tolist()]
        labels = build_pseudo_labels(patch_paths, n_clusters=num_classes)
        class_names = [str(i) for i in range(num_classes)]
        class_weights = None
        print(f"[info] pseudo-labels: {len(labels)} samples, {num_classes} clusters")

    if args.num_classes and args.num_classes != num_classes:
        num_classes = args.num_classes

    ds = LinhePatchDataset(patch_paths, labels, modality=args.modality)
    n_train = int(len(ds) * 0.8)
    n_val = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    if args.no_pretrain:
        backbone = PrithviBackbone(pretrained=False)
        print("[info] using RANDOM INIT backbone (no GeoFM pretrain)")
    else:
        ckpt = find_default_ckpt(args.prithvi_ckpt)
        if ckpt is None:
            raise FileNotFoundError(
                "Prithvi checkpoint not found. Put it at data/prithvi_100m.pt or pass --prithvi-ckpt"
            )
        backbone = PrithviBackbone(pretrained=True, checkpoint_path=ckpt)
    peft = args.peft_method
    if peft == "houlsby":
        for block in backbone.blocks:
            inject_houlsby_adapters(block, bottleneck_dim=args.bottleneck_dim)
    elif peft == "bitfit":
        configure_bitfit(backbone)
    if args.modality == "s2":
        adapter = ZeroPadAdapter(in_channels=6, out_channels=6)
    else:
        adapter = GeoAdapter(in_channels=3, out_channels=6)
    head = ClassificationHead(in_dim=768, num_classes=num_classes)

    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad) + \
                       sum(p.numel() for p in adapter.parameters()) + \
                       sum(p.numel() for p in head.parameters())
    print(f"[info] peft={peft} trainable_params={trainable_params:,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = PEFTTrainer(
        backbone=backbone,
        adapter=adapter,
        head=head,
        lr=args.lr,
        epochs=args.epochs,
        task="classification",
        device=device,
    )
    if class_weights is not None:
        trainer.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_val = 0.0
    save_path = OUT / f"linhe_{peft}_rgb.pt"

    for epoch in range(args.epochs):
        trainer.backbone.train()
        if trainer.adapter:
            trainer.adapter.train()
        trainer.head.train()
        train_loss = 0.0
        n = 0
        for x, y in train_loader:
            loss = trainer.train_step(x, y)
            train_loss += loss * x.size(0)
            n += x.size(0)
        trainer.step_scheduler()

        trainer.backbone.eval()
        if trainer.adapter:
            trainer.adapter.eval()
        trainer.head.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                logits = trainer.predict(x)
                pred = logits.argmax(dim=1).cpu()
                all_preds.extend(pred.tolist())
                all_labels.extend(y.tolist())
        val_acc = sum(p == t for p, t in zip(all_preds, all_labels)) / max(len(all_labels), 1)
        avg_loss = train_loss / max(n, 1)
        print(f"epoch {epoch+1}/{args.epochs} train_loss={avg_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc >= best_val:
            best_val = val_acc
            torch.save({
                "backbone": trainer.backbone.state_dict(),
                "adapter": trainer.adapter.state_dict() if trainer.adapter else None,
                "head": trainer.head.state_dict(),
                "config": vars(args),
                "best_val_acc": best_val,
                "class_names": class_names,
                "label_source": "real" if use_real else "pseudo",
            }, save_path)

    print(f"\n[done] best val_acc={best_val:.4f} → {save_path}")
    cr = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0, output_dict=True)
    if use_real:
        print("\n--- Per-class report (val set) ---")
        print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    report = {
        "peft_method": peft,
        "best_val_acc": round(best_val, 4),
        "weighted_f1": round(cr.get("weighted avg", {}).get("f1-score", 0), 4),
        "macro_f1": round(cr.get("macro avg", {}).get("f1-score", 0), 4),
        "trainable_params": trainable_params,
        "label_source": "real" if use_real else "pseudo",
        "class_names": class_names,
        "per_class_f1": {c: round(cr.get(c, {}).get("f1-score", 0), 4) for c in class_names},
        "n_train": n_train,
        "n_val": n_val,
        "epochs": args.epochs,
    }
    report_path = OUT / f"linhe_{peft}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[info] report → {report_path}")


if __name__ == "__main__":
    main()
