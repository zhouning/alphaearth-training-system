"""CLI entry point: python -m geoadapter.bench.run_benchmark --config path/to/config.yaml"""
import argparse
import yaml
import json
import itertools
import gc
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ckpt_path(ckpt_dir, method_name, modality, seed):
    from pathlib import Path
    return Path(ckpt_dir) / f"{method_name}__{modality}__seed{seed}.pt"


def _save_ckpt(path, trainer, adapter, head, epoch):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "adapter": adapter.state_dict(),
        "head": head.state_dict(),
        "backbone_peft": {k: v for k, v in trainer.backbone.state_dict().items()
                          if any(tag in k for tag in ("lora_", "adapter", "houlsby"))},
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict(),
    }, path)


def _load_ckpt(path, trainer, adapter, head):
    import torch
    ck = torch.load(path, map_location=trainer.device, weights_only=False)
    adapter.load_state_dict(ck["adapter"])
    head.load_state_dict(ck["head"])
    if ck.get("backbone_peft"):
        miss, _ = trainer.backbone.load_state_dict(ck["backbone_peft"], strict=False)
        # only PEFT subkeys were saved, so strict=False and missing keys = frozen backbone tensors
    trainer.optimizer.load_state_dict(ck["optimizer"])
    trainer.scheduler.load_state_dict(ck["scheduler"])
    return ck["epoch"]


def run_single_experiment(method_cfg, modality_cfg, global_cfg, seed,
                          ckpt_dir=None, ckpt_every=2):
    """Run one (method, modality, seed) combination. Returns metrics dict."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from geoadapter.models.prithvi import PrithviBackbone
    from geoadapter.models.heads import ClassificationHead
    from geoadapter.adapters import GeoAdapter, ZeroPadAdapter
    from geoadapter.adapters.lora import inject_lora
    from geoadapter.adapters.bitfit import configure_bitfit
    from geoadapter.adapters.houlsby import inject_houlsby_adapters
    from geoadapter.data.datasets import ModalityConfig
    from geoadapter.engine.trainer import PEFTTrainer
    from geoadapter.engine.evaluator import compute_classification_metrics, compute_multilabel_metrics, compute_segmentation_metrics
    import numpy as np

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_m = ModalityConfig(modality_cfg["preset"])
    epochs = global_cfg["experiment"]["epochs"]
    batch_size = global_cfg["experiment"]["batch_size"]

    backbone = PrithviBackbone(
        pretrained=global_cfg["prithvi"]["pretrained"],
        checkpoint_path=global_cfg["prithvi"].get("checkpoint"),
    )

    peft = method_cfg.get("peft")
    if peft == "lora":
        for block in backbone.blocks:
            inject_lora(block, rank=method_cfg.get("rank", 8))
    elif peft == "bitfit":
        configure_bitfit(backbone)
    elif peft == "houlsby":
        for block in backbone.blocks:
            inject_houlsby_adapters(block, bottleneck_dim=method_cfg.get("bottleneck_dim", 64))
    elif peft == "full_finetune":
        for p in backbone.parameters():
            p.requires_grad_(True)
    elif peft == "lora_split_qkv":
        from geoadapter.adapters.lora import split_qkv_and_inject_lora
        for block in backbone.blocks:
            split_qkv_and_inject_lora(block, rank=method_cfg.get("rank", 8))

    if method_cfg["adapter"] == "geo_adapter":
        adapter = GeoAdapter(in_channels=cfg_m.c_in, out_channels=6)
    else:
        adapter = ZeroPadAdapter(in_channels=cfg_m.c_in, out_channels=6)

    task_type = global_cfg["experiment"].get("task", "classification")
    num_classes = global_cfg["experiment"].get("num_classes", 10)
    if task_type == "multilabel":
        from geoadapter.models.heads import MultiLabelHead
        head = MultiLabelHead(in_dim=768, num_classes=num_classes)
    elif task_type == "segmentation":
        from geoadapter.models.heads import SegmentationHead
        head = SegmentationHead(in_dim=768, num_classes=num_classes, patch_size=16)
    else:
        head = ClassificationHead(in_dim=768, num_classes=num_classes)
    trainer = PEFTTrainer(
        backbone, adapter, head,
        lr=global_cfg["training"]["lr"],
        lr_peft=global_cfg["training"].get("lr_peft"),
        epochs=epochs,
        task=task_type,
        device=device,
        class_weights=global_cfg["training"].get("class_weights"),
        loss=global_cfg["training"].get("loss", "ce"),
        focal_gamma=global_cfg["training"].get("focal_gamma", 2.0),
    )

    n_trainable = sum(p.numel() for p in list(head.parameters()) +
                      list(adapter.parameters()) +
                      [p for p in backbone.parameters() if p.requires_grad])

    tag = f"{method_cfg['name']}|{modality_cfg['preset']}|seed={seed}"
    print(f"  [{tag}] device={device}, trainable_params={n_trainable:,}")

    # Try to load real dataset; fall back to synthetic for smoke test
    train_loader = None
    val_loader = None
    try:
        ds_root = global_cfg["experiment"]["dataset_root"]
        dataset_name = global_cfg["experiment"].get("dataset", "eurosat")
        max_samples = global_cfg["experiment"].get("max_samples")
        val_max_samples = global_cfg["experiment"].get("val_max_samples", max_samples)
        if dataset_name == "bigearthnet":
            from geoadapter.data.datasets import load_bigearthnet
            train_ds = load_bigearthnet(root=ds_root, modality=modality_cfg["preset"], split="train", max_samples=max_samples)
            val_ds = load_bigearthnet(root=ds_root, modality=modality_cfg["preset"], split="test", max_samples=val_max_samples)
        elif dataset_name == "sen1floods11":
            from geoadapter.data.datasets import load_sen1floods11
            train_ds = load_sen1floods11(root=ds_root, modality=modality_cfg["preset"], split="train", max_samples=max_samples)
            val_ds = load_sen1floods11(root=ds_root, modality=modality_cfg["preset"], split="test", max_samples=val_max_samples)
        elif dataset_name == "landcoverai":
            from geoadapter.data.datasets import load_landcoverai
            train_ds = load_landcoverai(root=ds_root, split="train", max_samples=max_samples)
            val_ds = load_landcoverai(root=ds_root, split="val", max_samples=val_max_samples)
        elif dataset_name == "linhe_buildings":
            from geoadapter.data.datasets import load_linhe_buildings
            split_mode = global_cfg["experiment"].get("split_mode", "scene")
            train_ds = load_linhe_buildings(
                root=ds_root, split="train", max_samples=max_samples, seed=seed,
                positive_min_share=global_cfg["experiment"].get("positive_min_share", 0.0),
                split_mode=split_mode,
            )
            val_ds = load_linhe_buildings(
                root=ds_root, split="val", max_samples=val_max_samples, seed=seed,
                positive_min_share=global_cfg["experiment"].get("positive_min_share", 0.0),
                split_mode=split_mode,
            )
        elif dataset_name == "linhe_lulc":
            from geoadapter.data.datasets import load_linhe_lulc
            year = global_cfg["experiment"].get("year", 2022)
            train_ds = load_linhe_lulc(root=ds_root, year=year, split="train", max_samples=max_samples)
            val_ds = load_linhe_lulc(root=ds_root, year=year, split="val", max_samples=val_max_samples)
        else:
            from geoadapter.data.datasets import load_eurosat
            train_ds = load_eurosat(root=ds_root, modality=modality_cfg["preset"], split="train")
            val_ds = load_eurosat(root=ds_root, modality=modality_cfg["preset"], split="test")
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        print(f"  [{tag}] Loaded {dataset_name}: {len(train_ds)} train, {len(val_ds)} val")
    except Exception as e:
        print(f"  [{tag}] Dataset not available ({e}), using synthetic data")
        x_syn = torch.randn(64, cfg_m.c_in, 64, 64)
        if task_type == "multilabel":
            y_syn = torch.randint(0, 2, (64, num_classes)).float()
        elif task_type == "segmentation":
            y_syn = torch.randint(0, num_classes, (64, 64, 64)).long()
        else:
            y_syn = torch.randint(0, num_classes, (64,))
        train_loader = DataLoader(TensorDataset(x_syn, y_syn), batch_size=batch_size)

    # Training loop with optional per-epoch checkpoint / resume
    start_epoch = 1
    ckpt_path = None
    if ckpt_dir is not None:
        ckpt_path = _ckpt_path(ckpt_dir, method_cfg["name"], modality_cfg["preset"], seed)
        if ckpt_path.exists():
            try:
                last = _load_ckpt(ckpt_path, trainer, adapter, head)
                start_epoch = last + 1
                print(f"  [{tag}] Resumed from epoch {last} via {ckpt_path.name}")
            except Exception as e:
                print(f"  [{tag}] ckpt load failed ({e}), starting from epoch 1")

    for epoch in range(start_epoch, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in train_loader:
            loss = trainer.train_step(batch_x, batch_y)
            epoch_loss += loss
            n_batches += 1
        trainer.step_scheduler()
        if epoch % 10 == 0 or epoch == epochs:
            avg = epoch_loss / max(n_batches, 1)
            print(f"  [{tag}] Epoch {epoch}/{epochs} loss={avg:.4f}")
        if ckpt_path is not None and (epoch % ckpt_every == 0 or epoch == epochs):
            _save_ckpt(ckpt_path, trainer, adapter, head, epoch)

    # Evaluation
    metrics = {"method": method_cfg["name"], "modality": modality_cfg["preset"],
               "seed": seed, "trainable_params": n_trainable}
    if val_loader:
        if task_type == "multilabel":
            all_scores, all_labels = [], []
            for batch_x, batch_y in val_loader:
                logits = trainer.predict(batch_x)
                all_scores.append(logits.sigmoid().cpu().numpy())
                all_labels.append(batch_y.cpu().numpy())
            eval_metrics = compute_multilabel_metrics(np.vstack(all_labels), np.vstack(all_scores))
            metrics.update(eval_metrics)
            print(f"  [{tag}] mAP={eval_metrics['mAP']:.4f}")
        elif task_type == "segmentation":
            all_preds, all_labels = [], []
            for batch_x, batch_y in val_loader:
                logits = trainer.predict(batch_x)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(batch_y.numpy())
            eval_metrics = compute_segmentation_metrics(
                np.concatenate(all_labels), np.concatenate(all_preds))
            metrics.update(eval_metrics)
            print(f"  [{tag}] mIoU={eval_metrics.get('mIoU', 0):.4f}")
        else:
            all_preds, all_labels = [], []
            for batch_x, batch_y in val_loader:
                logits = trainer.predict(batch_x)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.numpy())
            eval_metrics = compute_classification_metrics(np.array(all_labels), np.array(all_preds))
            metrics.update(eval_metrics)
            print(f"  [{tag}] OA={eval_metrics['overall_accuracy']:.4f} F1={eval_metrics['macro_f1']:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--dry-run", action="store_true", help="Only print experiment matrix")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Save/resume per-epoch checkpoint per experiment. "
                             "Critical for Colab where sessions die mid-training.")
    parser.add_argument("--checkpoint-every", type=int, default=2,
                        help="Save checkpoint every N epochs (default 2).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["experiment"]["epochs"] = args.epochs

    combos = list(itertools.product(cfg["methods"], cfg["modalities"], cfg["experiment"]["seeds"]))
    print(f"Total experiments: {len(combos)}")

    if args.dry_run:
        for method, modality, seed in combos:
            print(f"  {method['name']} x {modality['preset']} x seed={seed}")
        return

    output_path = Path(args.output)
    results = []
    done_keys = set()
    if output_path.exists():
        try:
            results = json.loads(output_path.read_text())
            done_keys = {(r["method"], r["modality"], r["seed"]) for r in results}
            print(f"Resuming from {output_path}: {len(done_keys)} experiments already completed")
        except Exception as e:
            print(f"WARN: could not parse existing output ({e}), starting fresh")
            results = []
            done_keys = set()

    for method, modality, seed in combos:
        key = (method["name"], modality["preset"], seed)
        if key in done_keys:
            print(f"  SKIP {method['name']}|{modality['preset']}|seed={seed} (already done)")
            continue
        result = run_single_experiment(method, modality, cfg, seed,
                                        ckpt_dir=args.checkpoint_dir,
                                        ckpt_every=args.checkpoint_every)
        results.append(result)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"  -> appended to {output_path} ({len(results)}/{len(combos)})")
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
