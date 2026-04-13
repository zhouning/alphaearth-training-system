"""CLI entry point: python -m geoadapter.bench.run_benchmark --config path/to/config.yaml"""
import argparse
import yaml
import json
import itertools
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_single_experiment(method_cfg, modality_cfg, global_cfg, seed):
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
    from geoadapter.engine.evaluator import compute_classification_metrics
    import numpy as np

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_m = ModalityConfig(modality_cfg["preset"])
    epochs = global_cfg["experiment"]["epochs"]
    batch_size = global_cfg["experiment"]["batch_size"]

    backbone = PrithviBackbone(pretrained=global_cfg["prithvi"]["pretrained"])

    peft = method_cfg.get("peft")
    if peft == "lora":
        for block in backbone.blocks:
            inject_lora(block, rank=method_cfg.get("rank", 8))
    elif peft == "bitfit":
        configure_bitfit(backbone)
    elif peft == "houlsby":
        for block in backbone.blocks:
            inject_houlsby_adapters(block, bottleneck_dim=method_cfg.get("bottleneck_dim", 64))

    if method_cfg["adapter"] == "geo_adapter":
        adapter = GeoAdapter(in_channels=cfg_m.c_in, out_channels=6)
    else:
        adapter = ZeroPadAdapter(in_channels=cfg_m.c_in, out_channels=6)

    head = ClassificationHead(in_dim=768, num_classes=10)
    trainer = PEFTTrainer(
        backbone, adapter, head,
        lr=global_cfg["training"]["lr"],
        lr_peft=global_cfg["training"].get("lr_peft"),
        epochs=epochs,
        device=device,
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
        from geoadapter.data.datasets import load_eurosat
        ds_root = global_cfg["experiment"]["dataset_root"]
        train_ds = load_eurosat(root=ds_root, modality=modality_cfg["preset"], split="train")
        val_ds = load_eurosat(root=ds_root, modality=modality_cfg["preset"], split="test")
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        print(f"  [{tag}] Loaded real dataset: {len(train_ds)} train, {len(val_ds)} val")
    except Exception as e:
        print(f"  [{tag}] Dataset not available ({e}), using synthetic data")
        x_syn = torch.randn(64, cfg_m.c_in, 64, 64)
        y_syn = torch.randint(0, 10, (64,))
        train_loader = DataLoader(TensorDataset(x_syn, y_syn), batch_size=batch_size)

    # Training loop
    for epoch in range(1, epochs + 1):
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

    # Evaluation
    metrics = {"method": method_cfg["name"], "modality": modality_cfg["preset"],
               "seed": seed, "trainable_params": n_trainable}
    if val_loader:
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["experiment"]["epochs"] = args.epochs
    results = []

    combos = list(itertools.product(cfg["methods"], cfg["modalities"], cfg["experiment"]["seeds"]))
    print(f"Total experiments: {len(combos)}")

    if args.dry_run:
        for method, modality, seed in combos:
            print(f"  {method['name']} x {modality['preset']} x seed={seed}")
        return

    for method, modality, seed in combos:
        result = run_single_experiment(method, modality, cfg, seed)
        results.append(result)

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
