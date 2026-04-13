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
    from geoadapter.models.prithvi import PrithviBackbone
    from geoadapter.models.heads import ClassificationHead
    from geoadapter.adapters import GeoAdapter, ZeroPadAdapter
    from geoadapter.adapters.lora import inject_lora
    from geoadapter.adapters.bitfit import configure_bitfit
    from geoadapter.adapters.houlsby import inject_houlsby_adapters
    from geoadapter.data.datasets import ModalityConfig
    from geoadapter.engine.trainer import PEFTTrainer

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_m = ModalityConfig(modality_cfg["preset"])

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
    trainer = PEFTTrainer(backbone, adapter, head, lr=global_cfg["training"]["lr"], device=device)

    print(f"  Running: method={method_cfg['name']}, modality={modality_cfg['preset']}, seed={seed}, device={device}")
    return {"method": method_cfg["name"], "modality": modality_cfg["preset"], "seed": seed, "device": device}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--dry-run", action="store_true", help="Only print experiment matrix")
    args = parser.parse_args()

    cfg = load_config(args.config)
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
