"""
Full experiment orchestrator — loops all configs × label fractions × seeds.

Execution order (matches two-phase pipeline):
  Phase 0  : Scratch baseline (train_baseline.py)
  Phase 1a : MAE pretraining  (train_mae.py)
  Phase 1b : DINO pretraining (train_dino.py)
  Phase 1c : Diffusion feature extraction + sweep (extract_features.py)
  Phase 2  : Linear probe sweep for all 7 SSL configs (train_downstream.py)
  Analysis : CKA between frozen encoders (eval/cka.py)
  Summary  : Aggregate all metrics to results/summary.csv + plots

Usage:
  uv run python run_sweep.py                          # full pipeline
  uv run python run_sweep.py --phase baseline         # only baseline sweep
  uv run python run_sweep.py --phase pretrain         # only Phase 1
  uv run python run_sweep.py --phase downstream       # only Phase 2
  uv run python run_sweep.py --phase cka              # only CKA analysis
  uv run python run_sweep.py --phase summary          # only aggregate results

Skip flags:
  --skip_baseline      skip baseline sweep
  --skip_mae           skip MAE pretraining
  --skip_dino          skip DINO pretraining
  --skip_diffusion     skip diffusion extraction
  --skip_downstream    skip downstream probe sweep
  --skip_cka           skip CKA analysis
"""

import os, argparse, json, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from itertools import product


# ──────────────────────── experiment configuration ───────────────────────────

LABEL_FRACS = [0.01, 0.05, 0.10, 1.0]
SEEDS       = [0, 1, 2]

DOWNSTREAM_CONFIGS = [
    "configs/standalone/mae_probe.yaml",
    "configs/standalone/dino_probe.yaml",
    "configs/standalone/diffusion_probe.yaml",
    "configs/fusion/mae_dino.yaml",
    "configs/fusion/mae_diffusion.yaml",
    "configs/fusion/dino_diffusion.yaml",
    "configs/fusion/mae_dino_diffusion.yaml",
]

CONFIG_DISPLAY_NAMES = {
    "baseline/vit_small_scratch":        "Scratch (ViT-Small)",
    "standalone/mae_probe":              "MAE",
    "standalone/dino_probe":             "DINO",
    "standalone/diffusion_probe":        "Diffusion",
    "fusion/mae_dino":                   "MAE + DINO",
    "fusion/mae_diffusion":              "MAE + Diffusion",
    "fusion/dino_diffusion":             "DINO + Diffusion",
    "fusion/mae_dino_diffusion":         "MAE + DINO + Diffusion ★",
}

COLORS = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D",
    "#3B1F2B", "#44BBA4", "#E94F37", "#393E41",
]


# ─────────────────────────── phase runners ───────────────────────────────────

def run_baseline(skip: bool = False):
    if skip:
        print("[sweep] Skipping baseline.")
        return
    from train_baseline import main as baseline_main
    print("\n" + "═" * 60)
    print("[sweep] Phase 0 — Scratch baseline sweep")
    print("═" * 60)
    for frac, seed in product(LABEL_FRACS, SEEDS):
        run_dir = os.path.join("results", "runs", "baseline", str(frac), str(seed))
        if os.path.exists(os.path.join(run_dir, "metrics.json")):
            print(f"  [skip] baseline frac={frac} seed={seed} — already done")
            continue
        print(f"\n  baseline: frac={frac}  seed={seed}")
        baseline_main("configs/baseline/vit_small_scratch.yaml",
                      label_fraction=frac, seed=seed)


def run_pretrain_mae(skip: bool = False):
    if skip:
        print("[sweep] Skipping MAE pretraining.")
        return
    ckpt = "results/checkpoints/mae_encoder.pt"
    if os.path.exists(ckpt):
        print(f"[sweep] MAE checkpoint found ({ckpt}). Skipping pretraining.")
        return
    print("\n" + "═" * 60)
    print("[sweep] Phase 1a — MAE pretraining")
    print("═" * 60)
    from train_mae import train as mae_train
    mae_train({})          # uses DEFAULT_CFG


def run_pretrain_dino(skip: bool = False):
    if skip:
        print("[sweep] Skipping DINO pretraining.")
        return
    ckpt = "results/checkpoints/dino_encoder.pt"
    if os.path.exists(ckpt):
        print(f"[sweep] DINO checkpoint found ({ckpt}). Skipping pretraining.")
        return
    print("\n" + "═" * 60)
    print("[sweep] Phase 1b — DINO pretraining")
    print("═" * 60)
    from train_dino import train as dino_train
    dino_train({})


def run_extract_diffusion(skip: bool = False):
    if skip:
        print("[sweep] Skipping diffusion extraction.")
        return
    cache_dir = "results/checkpoints/diffusion_embeddings"
    if os.path.exists(os.path.join(cache_dir, "train.pt")):
        print(f"[sweep] Diffusion embeddings found ({cache_dir}). Skipping extraction.")
        return
    print("\n" + "═" * 60)
    print("[sweep] Phase 1c — Diffusion feature extraction")
    print("═" * 60)
    import extract_features as ef
    ef.DEFAULT_CFG["data_root"] = "./data"
    best_cfg, _ = ef.run_sweep(ef.DEFAULT_CFG, "cuda" if __import__("torch").cuda.is_available() else "cpu")
    ef.extract_and_cache(ef.DEFAULT_CFG, best_cfg, "cuda" if __import__("torch").cuda.is_available() else "cpu")


def run_downstream(skip: bool = False):
    if skip:
        print("[sweep] Skipping downstream probe sweep.")
        return
    print("\n" + "═" * 60)
    print("[sweep] Phase 2 — Downstream linear probe sweep")
    print("═" * 60)
    from train_downstream import main as downstream_main, config_name_from_path
    for config_path in DOWNSTREAM_CONFIGS:
        for frac, seed in product(LABEL_FRACS, SEEDS):
            cname   = config_name_from_path(config_path)
            run_dir = os.path.join("results", "runs", cname, str(frac), str(seed))
            if os.path.exists(os.path.join(run_dir, "metrics.json")):
                print(f"  [skip] {cname} frac={frac} seed={seed}")
                continue
            print(f"\n  {cname}: frac={frac}  seed={seed}")
            try:
                downstream_main(config_path, label_fraction=frac, seed=seed)
            except Exception as e:
                print(f"  [ERROR] {e}")


def run_cka(skip: bool = False):
    if skip:
        print("[sweep] Skipping CKA.")
        return
    print("\n" + "═" * 60)
    print("[sweep] CKA analysis")
    print("═" * 60)
    from eval.cka import run_cka_analysis
    from data.stl10_data import get_labeled_loaders
    import torch

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    enc_map = {}

    mae_ckpt  = "results/checkpoints/mae_encoder.pt"
    dino_ckpt = "results/checkpoints/dino_encoder.pt"

    if os.path.exists(mae_ckpt):
        from encoders.mae import MAEEncoder
        enc = MAEEncoder(); enc.load_state_dict(torch.load(mae_ckpt, map_location=device))
        enc.freeze(); enc_map["mae"] = enc
    if os.path.exists(dino_ckpt):
        from encoders.dino import DINOEncoder
        enc = DINOEncoder(); enc.load_state_dict(torch.load(dino_ckpt, map_location=device))
        enc.freeze(); enc_map["dino"] = enc

    if not enc_map:
        print("[sweep] No encoder checkpoints found — skipping CKA.")
        return

    loader, _, _ = get_labeled_loaders(label_fraction=1.0, seed=0, batch_size=256)
    diff_cache   = {"train": "results/checkpoints/diffusion_embeddings/train.pt"}
    run_cka_analysis(enc_map, loader, device, "results/plots", diff_cache)


# ─────────────────────────── aggregation + plots ─────────────────────────────

def aggregate_results():
    """
    Walk results/runs/, collect all metrics.json files, build summary.csv,
    and generate label efficiency curves.
    """
    print("\n" + "═" * 60)
    print("[sweep] Aggregating results → results/summary.csv")
    print("═" * 60)

    rows = []
    runs_dir = os.path.join("results", "runs")
    if not os.path.exists(runs_dir):
        print("[sweep] No results found yet.")
        return

    for root_dir, dirs, files in os.walk(runs_dir):
        if "metrics.json" in files:
            with open(os.path.join(root_dir, "metrics.json")) as f:
                m = json.load(f)
            # Infer config name and run params from path structure
            rel = os.path.relpath(root_dir, runs_dir)
            parts = rel.replace("\\", "/").split("/")
            if len(parts) >= 3:
                config_parts = parts[:-2]
                frac         = float(parts[-2])
                seed         = int(parts[-1])
                config_name  = "/".join(config_parts)
                rows.append({
                    "config":       config_name,
                    "display_name": CONFIG_DISPLAY_NAMES.get(config_name, config_name),
                    "label_fraction": frac,
                    "seed":         seed,
                    "smoothed_acc": m.get("smoothed_acc", m.get("final_acc", None)),
                })

    if not rows:
        print("[sweep] No results collected.")
        return

    df = pd.DataFrame(rows)
    df.to_csv("results/summary.csv", index=False)
    print(f"[sweep] Saved {len(df)} rows → results/summary.csv")

    # Label efficiency curves
    _plot_label_efficiency(df)
    _plot_fusion_bar(df)


def _plot_label_efficiency(df: "pd.DataFrame"):
    """Mean ± std test accuracy vs. label fraction, one line per config."""
    fig, ax = plt.subplots(figsize=(9, 6))
    configs  = df["config"].unique()
    for i, cfg in enumerate(sorted(configs)):
        sub    = df[df["config"] == cfg]
        grp    = sub.groupby("label_fraction")["smoothed_acc"]
        fracs  = sorted(grp.groups.keys())
        means  = [grp.get_group(f).mean() for f in fracs]
        stds   = [grp.get_group(f).std()  for f in fracs]
        label  = CONFIG_DISPLAY_NAMES.get(cfg, cfg)
        color  = COLORS[i % len(COLORS)]
        lw     = 2.5 if "fusion/mae_dino_diffusion" in cfg else 1.5
        ax.errorbar(fracs, means, yerr=stds, label=label, color=color,
                    linewidth=lw, marker="o", capsize=4, markersize=5)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_xticks(LABEL_FRACS)
    ax.set_xlabel("Label Fraction", fontsize=13)
    ax.set_ylabel("Test Accuracy (mean ± std over seeds)", fontsize=13)
    ax.set_title("Label Efficiency Curves — STL-10", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = "results/plots/label_efficiency_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[sweep] Plot saved → {path}")


def _plot_fusion_bar(df: "pd.DataFrame"):
    """Bar chart comparing all configs at each label fraction."""
    fracs   = sorted(df["label_fraction"].unique())
    configs = sorted(df["config"].unique())
    n_cfg   = len(configs)
    x       = np.arange(len(fracs))
    width   = 0.8 / max(n_cfg, 1)

    fig, ax = plt.subplots(figsize=(max(10, len(fracs) * 3), 5))
    for i, cfg in enumerate(configs):
        means = []
        stds  = []
        for f in fracs:
            sub  = df[(df["config"] == cfg) & (df["label_fraction"] == f)]["smoothed_acc"]
            means.append(sub.mean() if len(sub) > 0 else 0)
            stds.append(sub.std()   if len(sub) > 1 else 0)
        label = CONFIG_DISPLAY_NAMES.get(cfg, cfg)
        ax.bar(x + i * width, means, width=width * 0.9, yerr=stds,
               label=label, color=COLORS[i % len(COLORS)], capsize=3)

    ax.set_xticks(x + width * n_cfg / 2)
    ax.set_xticklabels([f"{int(f * 100)}% labels" for f in fracs], fontsize=11)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Fusion vs. Baseline — All Label Fractions", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = "results/plots/fusion_comparison_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[sweep] Plot saved → {path}")


# ──────────────────────────────── main ───────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full experiment sweep orchestrator")
    parser.add_argument("--phase", default="all",
                        choices=["all", "baseline", "pretrain", "downstream", "cka", "summary"],
                        help="Which phase to run")
    parser.add_argument("--skip_baseline",   action="store_true")
    parser.add_argument("--skip_mae",        action="store_true")
    parser.add_argument("--skip_dino",       action="store_true")
    parser.add_argument("--skip_diffusion",  action="store_true")
    parser.add_argument("--skip_downstream", action="store_true")
    parser.add_argument("--skip_cka",        action="store_true")
    args = parser.parse_args()

    os.makedirs("results/plots",       exist_ok=True)
    os.makedirs("results/checkpoints", exist_ok=True)

    p = args.phase
    if p in ("all", "baseline"):
        run_baseline(skip=args.skip_baseline)
    if p in ("all", "pretrain"):
        run_pretrain_mae(skip=args.skip_mae)
        run_pretrain_dino(skip=args.skip_dino)
        run_extract_diffusion(skip=args.skip_diffusion)
    if p in ("all", "downstream"):
        run_downstream(skip=args.skip_downstream)
    if p in ("all", "cka"):
        run_cka(skip=args.skip_cka)
    if p in ("all", "summary"):
        aggregate_results()

    print("\n[sweep] All done.")
