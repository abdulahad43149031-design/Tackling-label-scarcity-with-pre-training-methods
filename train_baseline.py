"""
No-pretraining scratch baseline.

Trains ViT-Small/16 end-to-end from random init on scarce labeled STL-10 data.
This is the floor every SSL config must beat.

Reporting convention:
  reported_accuracy = mean(test_acc[-smoothing_window:])   ← pre-decided, not argmax
  Per config/fraction: run this script across seeds [0,1,2] → report mean ± std.

Output layout:
  results/runs/baseline/<label_fraction>/<seed>/
      config.yaml
      metrics.json     ← includes full history + smoothed_acc
      checkpoint.pt    ← best checkpoint (lowest test loss)
  results/plots/baseline/
      <frac>_<seed>_training_curves.png
      <frac>_<seed>_confusion_matrix.png
      <frac>_<seed>_sample_predictions.png
"""

import os, json, time, argparse, yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from models.vit import VisionTransformer
from data.stl10_data import get_labeled_loaders, CLASS_NAMES, NORM_MEAN, NORM_STD


# ------------------------------------------------------------------- helpers --

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def linear_warmup_decay(step, warmup_steps, total_steps):
    """Same schedule shape as the paper (linear warmup → linear decay),
    scaled to a training length STL-10 can support."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.0, 1.0 - progress)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, total = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        total += y.size(0)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(y.cpu())
    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return (preds == labels).mean(), total_loss / total, preds, labels


# -------------------------------------------------------------------- train --

def train(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg["seed"])

    train_loader, test_loader, n_train = get_labeled_loaders(
        label_fraction=cfg["label_fraction"],
        seed=cfg["seed"],
        batch_size=cfg["batch_size"],
    )
    print(f"Training on {n_train} labeled images "
          f"(label_fraction={cfg['label_fraction']}, "
          f"actual_batch_size={train_loader.batch_size})")

    model     = VisionTransformer(variant=cfg["variant"], drop=cfg["dropout"]).to(device)
    optimizer = Adam(model.parameters(), lr=cfg["base_lr"],
                     betas=(0.9, 0.999), weight_decay=cfg["weight_decay"])

    steps_per_epoch = max(1, len(train_loader))
    total_steps     = steps_per_epoch * cfg["epochs"]
    warmup_steps    = max(1, int(total_steps * cfg["warmup_fraction"]))
    scheduler       = LambdaLR(optimizer,
                                lr_lambda=lambda s: linear_warmup_decay(s, warmup_steps, total_steps))
    criterion = nn.CrossEntropyLoss()

    history    = {"train_loss": [], "test_loss": [], "test_acc": []}
    best_loss  = float("inf")
    best_state = None

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item(); n_batches += 1

        test_acc, test_loss, _, _ = evaluate(model, test_loader, device, criterion)
        history["train_loss"].append(epoch_loss / n_batches)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(float(test_acc))

        # Track best checkpoint by test loss
        if test_loss < best_loss:
            best_loss  = test_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"epoch {epoch+1:3d}/{cfg['epochs']} | "
              f"train_loss {history['train_loss'][-1]:.4f} | "
              f"test_acc {test_acc:.4f}")

    # Smoothed accuracy: mean of last N epochs (pre-decided, not argmax)
    n = cfg.get("smoothing_window", 10)
    smoothed_acc = float(np.mean(history["test_acc"][-n:]))

    # Reload best checkpoint for final eval + confusion matrix
    model.load_state_dict(best_state)
    model.to(device)
    final_acc, _, preds, labels = evaluate(model, test_loader, device, criterion)
    report = classification_report(labels, preds, target_names=CLASS_NAMES, output_dict=True)

    return model, history, smoothed_acc, final_acc, preds, labels, report, test_loader, best_state


# ----------------------------------------------------------------- plotting --

def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train loss")
    axes[0].plot(history["test_loss"],  label="test loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].legend(); axes[0].set_title("Loss")
    axes[1].plot(history["test_acc"], color="green")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("test accuracy")
    axes[1].set_title("Test Accuracy")
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


def plot_confusion_matrix(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


def denormalize(img_tensor):
    mean = torch.tensor(NORM_MEAN).view(3, 1, 1)
    std  = torch.tensor(NORM_STD).view(3, 1, 1)
    return (img_tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def plot_sample_predictions(model, test_loader, class_names, device, save_path, n_samples=12):
    model.eval()
    x, y = next(iter(test_loader))
    idx  = np.random.choice(len(x), size=min(n_samples, len(x)), replace=False)
    x_sample, y_sample = x[idx].to(device), y[idx]
    with torch.no_grad():
        preds = model(x_sample).argmax(dim=1).cpu()

    n_cols = 4
    n_rows = int(np.ceil(len(idx) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.4, n_rows * 2.6))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i >= len(idx):
            ax.axis("off"); continue
        correct = preds[i].item() == y_sample[i].item()
        ax.imshow(denormalize(x_sample[i]))
        ax.set_title(f"pred: {class_names[preds[i]]}\ntrue: {class_names[y_sample[i]]}",
                     color="green" if correct else "red", fontsize=9)
        ax.axis("off")
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------- main --

def main(config_path: str, label_fraction: float = None, seed: int = None):
    cfg  = yaml.safe_load(open(config_path))
    # Allow programmatic overrides (used by run_sweep.py)
    if label_fraction is not None:
        cfg["label_fraction"] = label_fraction
    if seed is not None:
        cfg["seed"] = seed
    frac = cfg["label_fraction"]
    seed = cfg["seed"]

    # Deterministic output paths — no run_name, no date suffix → never silently overwritten
    run_dir  = os.path.join("results", "runs", "baseline", str(frac), str(seed))
    plot_dir = os.path.join("results", "plots", "baseline")
    os.makedirs(run_dir,  exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    yaml.dump(cfg, open(os.path.join(run_dir, "config.yaml"), "w"))

    (model, history, smoothed_acc, final_acc,
     preds, labels, report, test_loader, best_state) = train(cfg)

    torch.save(best_state, os.path.join(run_dir, "checkpoint.pt"))

    n = cfg.get("smoothing_window", 10)
    json.dump(
        {
            "label_fraction":  frac,
            "seed":            seed,
            "smoothed_acc":    smoothed_acc,   # mean(test_acc[-N:]) — primary reported metric
            "smoothing_window": n,
            "final_acc":       final_acc,      # best-ckpt accuracy, kept for reference
            "history":         history,
            "classification_report": report,
        },
        open(os.path.join(run_dir, "metrics.json"), "w"),
        indent=2,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag    = f"{frac}_{seed}"
    plot_training_curves(history,
                         os.path.join(plot_dir, f"{tag}_training_curves.png"))
    plot_confusion_matrix(labels, preds, CLASS_NAMES,
                          os.path.join(plot_dir, f"{tag}_confusion_matrix.png"))
    plot_sample_predictions(model, test_loader, CLASS_NAMES, device,
                            os.path.join(plot_dir, f"{tag}_sample_predictions.png"))

    print(f"\nSmoothed acc (mean last {n} epochs): {smoothed_acc:.4f}")
    print(f"Best-ckpt acc:                        {final_acc:.4f}")
    print(f"Results → {run_dir}/")
    print(f"Plots   → {plot_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline/vit_small_scratch.yaml")
    args = parser.parse_args()
    main(args.config)
