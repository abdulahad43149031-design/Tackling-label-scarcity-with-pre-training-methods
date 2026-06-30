"""
Centered Kernel Alignment (CKA) — go/no-go gate before trusting fusion results.
Reference: Kornblith et al., "Similarity of Neural Network Representations Revisited" (ICML 2019)

Computes linear CKA between every pair of frozen encoder embeddings on the full STL-10
labeled train set (5,000 images).

Interpretation:
  CKA ≈ 0.0 → representations are very different → fusion likely helps
  CKA ≈ 1.0 → representations are redundant    → fusion unlikely to help

Usage:
  uv run python -m eval.cka
  uv run python -m eval.cka --config configs/cka.yaml
"""

import os, argparse, json, yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm


# ─────────────────────────── CKA computation ─────────────────────────────────

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Unbiased estimator of linear CKA.
    X: (n, d1), Y: (n, d2) — row = one sample's embedding.

    CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))
    using the linear kernel: K = X X^T, L = Y Y^T
    """
    n  = X.shape[0]
    X  = X - X.mean(dim=0, keepdim=True)         # centre columns
    Y  = Y - Y.mean(dim=0, keepdim=True)

    # Gram matrices (n × n)
    K  = X @ X.T
    L  = Y @ Y.T

    # Frobenius inner product = sum of element-wise product
    hsic_kl = (K * L).sum()
    hsic_kk = (K * K).sum()
    hsic_ll = (L * L).sum()

    if hsic_kk == 0 or hsic_ll == 0:
        return 0.0
    return float(hsic_kl / (hsic_kk.sqrt() * hsic_ll.sqrt()))


# ─────────────────────── embedding extraction ────────────────────────────────

@torch.no_grad()
def _extract(encoder, loader, device: str) -> torch.Tensor:
    all_embs = []
    for imgs, _ in tqdm(loader, desc="  extract", leave=False):
        embs = encoder.encode(imgs.to(device)).cpu()
        all_embs.append(embs)
    return torch.cat(all_embs)


def extract_all(encoder_map: dict, loader, device: str) -> dict:
    """
    encoder_map: {name: encoder}
    Returns {name: (N, d) embedding tensor}
    """
    embeddings = {}
    for name, enc in encoder_map.items():
        print(f"  Extracting embeddings: {name}")
        enc.eval()
        enc.to(device)
        embeddings[name] = _extract(enc, loader, device)
    return embeddings


# ─────────────────────────── CKA analysis ────────────────────────────────────

def run_cka_analysis(
    encoder_map:   dict,
    loader,
    device:        str,
    out_dir:       str = "results/plots",
    diffusion_cache_paths: dict = None,  # {split: path} for diffusion cached embeddings
) -> dict:
    """
    Compute pairwise CKA, print verdict, save heatmap.

    Args:
        encoder_map          : {name: encoder} for MAE and DINO (live encoders)
        loader               : labeled train loader (5K images)
        device               : "cuda" or "cpu"
        out_dir              : where to save the heatmap
        diffusion_cache_paths: if provided, loads diffusion embeddings from cache
    Returns:
        cka_matrix as dict {(name_i, name_j): cka_value}
    """
    os.makedirs(out_dir, exist_ok=True)

    # Collect embeddings
    embeddings = {}
    for name, enc in encoder_map.items():
        print(f"  [{name}] extracting...")
        enc.eval(); enc.to(device)
        embeddings[name] = _extract(enc, loader, device)

    # Load diffusion from cache if provided
    if diffusion_cache_paths:
        cache_path = diffusion_cache_paths.get("train")
        if cache_path and os.path.exists(cache_path):
            print("  [diffusion] loading from cache...")
            data = torch.load(cache_path)
            embeddings["diffusion"] = data["embeddings"].float()
        else:
            print(f"  [diffusion] cache not found at {cache_path}, skipping")

    names = list(embeddings.keys())
    n     = len(names)

    # Pairwise CKA
    cka_matrix = {}
    matrix_np  = np.zeros((n, n))

    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i == j:
                val = 1.0
            elif (ni, nj) in cka_matrix:
                val = cka_matrix[(ni, nj)]
            else:
                X   = embeddings[ni].float()
                Y   = embeddings[nj].float()
                # Sub-sample to at most 2000 for speed (full 5K is also fine)
                if X.shape[0] > 2000:
                    idx = torch.randperm(X.shape[0])[:2000]
                    X, Y = X[idx], Y[idx]
                val = linear_cka(X, Y)
                cka_matrix[(ni, nj)] = val
                cka_matrix[(nj, ni)] = val
            matrix_np[i, j] = val

    # Print verdict
    print("\n── CKA Go/No-Go Verdicts ──────────────────────────────────")
    for (ni, nj), val in cka_matrix.items():
        if names.index(ni) >= names.index(nj):
            continue
        verdict = "✅ Complement (fusion likely helps)" if val < 0.5 else \
                  "⚠️  Similar (fusion may be redundant)"
        print(f"  {ni:12s} ↔ {nj:12s}  CKA = {val:.4f}  {verdict}")
    print()

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(max(4, n + 1), max(4, n + 1)))
    im = ax.imshow(matrix_np, cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=12)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=12)
    ax.set_title("Pairwise Linear CKA\n(0 = complementary, 1 = redundant)", fontsize=13)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix_np[i, j]:.3f}", ha="center", va="center",
                    fontsize=11, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    heatmap_path = os.path.join(out_dir, "cka_heatmap.png")
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    print(f"[CKA] Heatmap saved → {heatmap_path}")

    # Save raw values
    results = {f"{ni}_vs_{nj}": cka_matrix.get((ni, nj), None)
               for i, ni in enumerate(names) for j, nj in enumerate(names) if i < j}
    with open(os.path.join(out_dir, "cka_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ──────────────────────────────── main ───────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CKA analysis between frozen encoders")
    parser.add_argument("--mae_ckpt",        default="results/checkpoints/mae_encoder.pt")
    parser.add_argument("--dino_ckpt",       default="results/checkpoints/dino_encoder.pt")
    parser.add_argument("--diffusion_cache", default="results/checkpoints/diffusion_embeddings/train.pt")
    parser.add_argument("--variant",         default="ViT-Small/16")
    parser.add_argument("--data_root",       default="./data")
    parser.add_argument("--out_dir",         default="results/plots")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CKA] device={device}")

    from encoders.mae import MAEEncoder
    from encoders.dino import DINOEncoder
    from data.stl10_data import get_labeled_loaders

    encoder_map = {}
    if os.path.exists(args.mae_ckpt):
        enc = MAEEncoder(variant=args.variant)
        enc.load_state_dict(torch.load(args.mae_ckpt, map_location=device))
        enc.freeze()
        encoder_map["mae"] = enc
    else:
        print(f"[CKA] MAE checkpoint not found: {args.mae_ckpt}")

    if os.path.exists(args.dino_ckpt):
        enc = DINOEncoder(variant=args.variant)
        enc.load_state_dict(torch.load(args.dino_ckpt, map_location=device))
        enc.freeze()
        encoder_map["dino"] = enc
    else:
        print(f"[CKA] DINO checkpoint not found: {args.dino_ckpt}")

    if len(encoder_map) == 0:
        print("[CKA] No encoder checkpoints found. Run train_mae.py and train_dino.py first.")
        raise SystemExit(1)

    train_loader, _, _ = get_labeled_loaders(
        root=args.data_root, label_fraction=1.0, seed=0, batch_size=256, num_workers=4
    )
    diff_cache = {"train": args.diffusion_cache} if os.path.exists(args.diffusion_cache) else None

    run_cka_analysis(
        encoder_map=encoder_map,
        loader=train_loader,
        device=device,
        out_dir=args.out_dir,
        diffusion_cache_paths=diff_cache,
    )
