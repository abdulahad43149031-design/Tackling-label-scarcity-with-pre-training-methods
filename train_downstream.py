"""
Phase 2 downstream training — linear probe for all non-baseline configs.

Handles all 7 configurations:
  Standalone (1 encoder, no fusion):
    configs/standalone/{mae,dino,diffusion}_probe.yaml
  Duo fusion (2 encoders + EmbeddingFusion):
    configs/fusion/{mae_dino, mae_diffusion, dino_diffusion}.yaml
  Trio fusion (3 encoders + EmbeddingFusion):
    configs/fusion/mae_dino_diffusion.yaml

Output layout:
  results/runs/<config_name>/<label_fraction>/<seed>/
      config.yaml
      metrics.json     ← smoothed_acc (primary), full history
      checkpoint.pt    ← probe head + fusion module weights

Usage:
  uv run python train_downstream.py --config configs/standalone/mae_probe.yaml
  uv run python train_downstream.py --config configs/fusion/mae_dino_diffusion.yaml
  uv run python train_downstream.py --config configs/fusion/mae_dino.yaml --label_fraction 0.05 --seed 1
"""

import os, argparse, yaml, json
import torch

from eval.linear_probe import (
    load_encoder, _DiffusionCacheEncoder,
    extract_all_embeddings_with_diffusion,
    train_probe,
)


# ─────────────────────────── config parsing ──────────────────────────────────

def parse_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_encoder_cfgs(cfg: dict) -> list[dict]:
    """Normalise single-encoder and multi-encoder config formats to a list."""
    if "encoders" in cfg:
        return cfg["encoders"]                              # multi-encoder format
    else:
        return [{                                           # single-encoder format
            "name":    cfg["encoder"],
            "ckpt":    cfg.get("encoder_ckpt"),
            "variant": cfg.get("variant", "ViT-Small/16"),
            **{k: cfg[k] for k in ("diffusion_model", "timestep", "hook_layer",
                                    "cache_dir")
               if k in cfg},
        }]


def config_name_from_path(config_path: str) -> str:
    """Extract a clean name like 'standalone/mae_probe' from the config path."""
    parts = os.path.normpath(config_path).split(os.sep)
    # Find 'configs' in path and take everything after it
    try:
        idx = next(i for i, p in enumerate(parts) if p == "configs")
        return os.path.join(*parts[idx + 1:]).replace(".yaml", "")
    except StopIteration:
        return os.path.basename(config_path).replace(".yaml", "")


# ─────────────────────────────── main ────────────────────────────────────────

def main(config_path: str, label_fraction: float = None, seed: int = None):
    cfg = parse_config(config_path)

    # CLI overrides for running sweeps programmatically
    if label_fraction is not None:
        cfg["label_fraction"] = label_fraction
    if seed is not None:
        cfg["seed"] = seed

    frac        = cfg["label_fraction"]
    run_seed    = cfg["seed"]
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    config_name = config_name_from_path(config_path)

    run_dir = os.path.join("results", "runs", config_name, str(frac), str(run_seed))
    os.makedirs(run_dir, exist_ok=True)
    yaml.dump(cfg, open(os.path.join(run_dir, "config.yaml"), "w"))

    print(f"\n{'='*60}")
    print(f"[downstream] config : {config_name}")
    print(f"[downstream] frac   : {frac}  seed: {run_seed}  device: {device}")
    print(f"{'='*60}")

    # ── Load encoders ─────────────────────────────────────────────────────────
    enc_cfgs = get_encoder_cfgs(cfg)
    encoders = []
    for ec in enc_cfgs:
        print(f"  Loading encoder: {ec['name']}")
        enc = load_encoder(ec, device)
        encoders.append(enc)

    # ── Pre-extract embeddings ────────────────────────────────────────────────
    print("\n  Pre-extracting embeddings (once)...")
    train_embs, test_embs, train_labels, test_labels = \
        extract_all_embeddings_with_diffusion(encoders, enc_cfgs, cfg, device)

    print(f"  Train embeddings: {[list(e.shape) for e in train_embs]}")
    print(f"  Test  embeddings: {[list(e.shape) for e in test_embs]}")

    # ── Train probe ───────────────────────────────────────────────────────────
    print("\n  Training probe head...")
    metrics = train_probe(
        train_embs_list=train_embs,
        test_embs_list=test_embs,
        train_labels=train_labels,
        test_labels=test_labels,
        cfg=cfg,
        out_dir=run_dir,
        device=device,
    )

    print(f"\n[downstream] smoothed_acc = {metrics['smoothed_acc']:.4f}  "
          f"(mean last {metrics['smoothing_window']} epochs)")
    print(f"[downstream] Results → {run_dir}/")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear probe for SSL configs")
    parser.add_argument("--config", required=True,
                        help="Path to YAML config (standalone or fusion)")
    parser.add_argument("--label_fraction", type=float, default=None,
                        help="Override label_fraction in config")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed in config")
    args = parser.parse_args()
    main(args.config, args.label_fraction, args.seed)
