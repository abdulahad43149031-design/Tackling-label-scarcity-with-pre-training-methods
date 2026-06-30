"""
Diffusion feature extraction and hyperparameter sweep — Phase 1c.

No training required. This script:
  1. Sweeps (timestep × hook_layer) combinations using a quick linear probe on 10% labels.
  2. Picks the best config by validation accuracy.
  3. Caches embeddings for ALL STL-10 images (labeled train + test + unlabeled) using the best config.
  4. Saves embeddings and best config to disk.

Cached embeddings are used by train_downstream.py instead of re-running the U-Net every epoch.

Usage:
  uv run python extract_features.py
  uv run python extract_features.py --config configs/pretrain/diffusion.yaml
"""

import os, argparse, json, yaml
import numpy as np
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from encoders.diffusion import DiffusionEncoder
from data.stl10_data import get_all_images_loader, get_labeled_loaders


# ─────────────────────────── embedding extraction ────────────────────────────

@torch.no_grad()
def extract_embeddings(encoder: DiffusionEncoder, loader, device: str) -> tuple:
    """Extract embeddings for all batches in a loader."""
    all_embs, all_labels = [], []
    for batch in tqdm(loader, desc="  extracting", leave=False):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            imgs, labels = batch
        else:
            imgs, labels = batch, torch.full((batch.shape[0],), -1)
        imgs = imgs.to(device, non_blocking=True)
        embs = encoder.encode(imgs).cpu()
        all_embs.append(embs)
        all_labels.append(labels if isinstance(labels, torch.Tensor) else torch.tensor(labels))
    return torch.cat(all_embs), torch.cat(all_labels)


# ──────────────────────────── sweep + probe ──────────────────────────────────

def quick_probe_accuracy(
    train_embs: torch.Tensor,
    train_labels: torch.Tensor,
    test_embs: torch.Tensor,
    test_labels: torch.Tensor,
) -> float:
    """
    Fit a simple logistic regression on embeddings and return test accuracy.
    Used for the timestep × hook_layer sweep.
    """
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(train_embs.numpy())
    X_te   = scaler.transform(test_embs.numpy())
    y_tr   = train_labels.numpy()
    y_te   = test_labels.numpy()

    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", multi_class="multinomial")
    clf.fit(X_tr, y_tr)
    return float((clf.predict(X_te) == y_te).mean())


def run_sweep(cfg: dict, device: str):
    """
    Grid search over (timestep × hook_layer) using 10% labels as validation.
    Returns the best (timestep, hook_layer) pair.
    """
    model_id       = cfg["model_id"]
    model_img_size = cfg.get("model_img_size", 32)
    timesteps      = cfg.get("timesteps_to_sweep", [100, 250, 500])
    hook_layers    = cfg.get("hook_layers_to_sweep", ["mid_block", "down_blocks.2"])
    data_root      = cfg.get("data_root", "./data")

    # Load labeled data for quick sweep (10% labels, seed=0)
    train_loader_10pct, test_loader, _ = get_labeled_loaders(
        root=data_root, label_fraction=0.1, seed=0, batch_size=256, num_workers=4
    )

    results = []
    print(f"\n[Diffusion sweep] model={model_id}  "
          f"timesteps={timesteps}  layers={hook_layers}")

    for timestep in timesteps:
        for hook_layer in hook_layers:
            print(f"  sweep: timestep={timestep}, layer={hook_layer}", end="  ")

            enc = DiffusionEncoder(
                model_id=model_id,
                timestep=timestep,
                hook_layer=hook_layer,
                model_img_size=model_img_size,
                device=device,
            )

            tr_embs, tr_labels = extract_embeddings(enc, train_loader_10pct, device)
            te_embs, te_labels = extract_embeddings(enc, test_loader, device)
            acc = quick_probe_accuracy(tr_embs, tr_labels, te_embs, te_labels)

            print(f"→ accuracy: {acc:.4f}")
            results.append({
                "timestep":   timestep,
                "hook_layer": hook_layer,
                "accuracy":   acc,
            })

            # Free GPU memory
            del enc; torch.cuda.empty_cache()

    best = max(results, key=lambda r: r["accuracy"])
    print(f"\n[Diffusion sweep] Best: timestep={best['timestep']}, "
          f"layer={best['hook_layer']}, acc={best['accuracy']:.4f}")
    return best, results


# ──────────────────────────── full extraction ─────────────────────────────────

def extract_and_cache(cfg: dict, best_cfg: dict, device: str):
    """
    Extract embeddings for all STL-10 splits using the best config,
    and save to disk for downstream reuse.
    """
    model_id       = cfg["model_id"]
    model_img_size = cfg.get("model_img_size", 32)
    data_root      = cfg.get("data_root", "./data")
    out_dir        = cfg.get("embedding_cache_dir", "results/checkpoints/diffusion_embeddings")
    os.makedirs(out_dir, exist_ok=True)

    enc = DiffusionEncoder(
        model_id=model_id,
        timestep=best_cfg["timestep"],
        hook_layer=best_cfg["hook_layer"],
        model_img_size=model_img_size,
        device=device,
    )

    splits = {
        "train":     ("train",     1.0, 0),
        "test":      ("test",      1.0, 0),
        "unlabeled": ("unlabeled", 1.0, 0),
    }

    for name, (split, frac, seed) in splits.items():
        print(f"[Diffusion] extracting {name} split...", end=" ")
        loader, labels_arr = get_all_images_loader(
            root=data_root, split=split,
            label_fraction=frac, seed=seed,
            batch_size=256, num_workers=4,
        )
        embs, labels = extract_embeddings(enc, loader, device)
        torch.save({"embeddings": embs, "labels": labels},
                   os.path.join(out_dir, f"{name}.pt"))
        print(f"saved {embs.shape} → {out_dir}/{name}.pt")

    print(f"\n[Diffusion] All embeddings cached to {out_dir}/")


# ──────────────────────────────── main ───────────────────────────────────────

DEFAULT_CFG = {
    "model_id":              "google/ddpm-cifar10-32",
    "model_img_size":        32,
    "timesteps_to_sweep":    [100, 250, 500],
    "hook_layers_to_sweep":  ["mid_block", "down_blocks.2"],
    "data_root":             "./data",
    "ckpt_dir":              "results/checkpoints",
    "embedding_cache_dir":   "results/checkpoints/diffusion_embeddings",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion feature extraction + sweep")
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip_sweep", action="store_true",
                        help="Skip sweep and use timestep/hook_layer from config directly")
    args = parser.parse_args()

    cfg = DEFAULT_CFG.copy()
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Diffusion] device={device}")

    ckpt_dir = cfg.get("ckpt_dir", "results/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.skip_sweep:
        best_cfg = {
            "timestep":   cfg.get("timestep", 250),
            "hook_layer": cfg.get("hook_layer", "mid_block"),
            "accuracy":   None,
        }
        sweep_results = []
    else:
        best_cfg, sweep_results = run_sweep(cfg, device)

    # Save best config
    best_cfg_path = os.path.join(ckpt_dir, "diffusion_best_config.yaml")
    with open(best_cfg_path, "w") as f:
        yaml.dump({**cfg, **best_cfg}, f)
    with open(os.path.join(ckpt_dir, "diffusion_sweep_results.json"), "w") as f:
        json.dump(sweep_results, f, indent=2)

    # Extract and cache embeddings
    extract_and_cache(cfg, best_cfg, device)
    print("[Diffusion] Done.")
