"""
MAE pretraining on STL-10's 100K unlabeled images — Phase 1a.

Training protocol:
  - Encoder: ViT-Small/16 (identical architecture to scratch baseline)
  - Decoder: lightweight 4-layer transformer (decoder_dim=256); discarded after training
  - Loss: normalised pixel MSE on 75%-masked patches
  - Optimizer: AdamW, base_lr scales linearly with batch size (lr = base_lr * bs/256)
  - Schedule: cosine decay with linear warmup
  - Output: results/checkpoints/mae_encoder.pt (encoder weights only)

Usage:
  uv run python train_mae.py
  uv run python train_mae.py --config configs/pretrain/mae.yaml
"""

import os, argparse, json, time, yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from encoders.mae import MAEEncoder, MAEDecoder, mae_loss
from data.stl10_data import get_unlabeled_loader


# ─────────────────────────────── LR schedule ─────────────────────────────────

def cosine_lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup → cosine decay (returns multiplicative factor)."""
    import math
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ─────────────────────────────── training ────────────────────────────────────

def train(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MAE] device={device}  variant={cfg['variant']}  epochs={cfg['epochs']}")

    torch.manual_seed(cfg.get("seed", 42))

    loader = get_unlabeled_loader(
        root=cfg.get("data_root", "./data"),
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 4),
    )
    print(f"[MAE] unlabeled images: {len(loader.dataset):,}  "
          f"steps/epoch: {len(loader)}")

    encoder = MAEEncoder(
        variant=cfg["variant"],
        img_size=cfg.get("img_size", 96),
        patch_size=cfg.get("patch_size", 16),
        drop=0.0,                       # MAE paper: no dropout in encoder during pretraining
    ).to(device)

    decoder = MAEDecoder(
        encoder_dim=encoder.embed_dim,
        decoder_dim=cfg.get("decoder_dim", 256),
        decoder_depth=cfg.get("decoder_depth", 4),
        decoder_heads=cfg.get("decoder_heads", 8),
        num_patches=encoder.n_patches,
        patch_size=cfg.get("patch_size", 16),
        in_channels=3,
    ).to(device)

    # Linear LR scaling: lr = base_lr * batch_size / 256
    effective_lr = cfg["base_lr"] * cfg["batch_size"] / 256.0
    optimizer = AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=effective_lr,
        betas=(0.9, 0.95),             # MAE paper uses β2=0.95
        weight_decay=cfg.get("weight_decay", 0.05),
    )

    total_steps  = len(loader) * cfg["epochs"]
    warmup_steps = len(loader) * cfg.get("warmup_epochs", 20)
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_lr_lambda(s, warmup_steps, total_steps),
    )

    mask_ratio   = cfg.get("mask_ratio", 0.75)
    patch_size   = cfg.get("patch_size", 16)
    norm_pix     = cfg.get("norm_pix_loss", True)
    ckpt_dir     = cfg.get("ckpt_dir", "results/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    history = []
    best_loss = float("inf")

    for epoch in range(cfg["epochs"]):
        encoder.train(); decoder.train()
        epoch_loss, n_batches = 0.0, 0
        pbar = tqdm(loader, desc=f"MAE epoch {epoch+1:3d}/{cfg['epochs']}", leave=False)

        for imgs in pbar:
            imgs = imgs.to(device, non_blocking=True)

            encoded, mask, ids_restore = encoder.forward_features(imgs, mask_ratio)
            pred  = decoder(encoded, ids_restore)
            loss  = mae_loss(pred, imgs, mask, patch_size, norm_pix)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                max_norm=cfg.get("clip_grad", 3.0),
            )
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item(); n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / n_batches
        history.append({"epoch": epoch + 1, "loss": avg_loss})
        print(f"[MAE] epoch {epoch+1:3d}/{cfg['epochs']} | loss {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), os.path.join(ckpt_dir, "mae_encoder_best.pt"))

    # Save final encoder (and log best loss)
    torch.save(encoder.state_dict(), os.path.join(ckpt_dir, "mae_encoder.pt"))
    with open(os.path.join(ckpt_dir, "mae_training_log.json"), "w") as f:
        json.dump({"history": history, "best_loss": best_loss, "config": cfg}, f, indent=2)

    print(f"\n[MAE] Pretraining complete. Best loss: {best_loss:.4f}")
    print(f"[MAE] Encoder saved → {ckpt_dir}/mae_encoder.pt")
    return encoder


# ──────────────────────────────── main ───────────────────────────────────────

DEFAULT_CFG = {
    "variant":        "ViT-Small/16",
    "img_size":       96,
    "patch_size":     16,
    "mask_ratio":     0.75,
    "epochs":         200,
    "batch_size":     128,
    "base_lr":        1.5e-4,
    "weight_decay":   0.05,
    "warmup_epochs":  20,
    "decoder_dim":    256,
    "decoder_depth":  4,
    "decoder_heads":  8,
    "norm_pix_loss":  True,
    "clip_grad":      3.0,
    "num_workers":    4,
    "data_root":      "./data",
    "ckpt_dir":       "results/checkpoints",
    "seed":           42,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAE pretraining on STL-10 unlabeled")
    parser.add_argument("--config", default=None,
                        help="Path to YAML config (overrides defaults)")
    args = parser.parse_args()

    cfg = DEFAULT_CFG.copy()
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))

    train(cfg)
