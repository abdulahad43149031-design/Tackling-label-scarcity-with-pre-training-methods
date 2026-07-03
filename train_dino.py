"""
DINO pretraining on STL-10's 100K unlabeled images — Phase 1b.

Resumable training: trains for `resume_interval` epochs per run (default 20),
saves a full resume checkpoint, and stops. Next run auto-detects and resumes
from where it left off. When all `epochs` are complete, intermediate/resume
checkpoints are deleted — only the final dino_encoder.pt is kept.

Training protocol:
  - Student: ViT-Small/16 (trainable)
  - Teacher: EMA copy of student (no gradient); momentum starts at 0.996 → 1.0 (cosine)
  - Multi-crop: 2 global crops (scale 0.4–1.0) + 6 local crops (scale 0.05–0.4), all 96×96
  - Loss: DINOLoss (cross-entropy + centering to prevent collapse)
  - LR: cosine schedule with linear warmup; WD: cosine warmup 0.04 → 0.4
  - Freeze last layer for first freeze_last_layer_epochs to stabilise training
  - Output: results/checkpoints/dino_encoder.pt (student encoder weights only)

Usage:
  uv run python train_dino.py                                    # trains epochs 1-20, stops
  uv run python train_dino.py                                    # resumes epochs 21-40, stops
  uv run python train_dino.py                                    # ... until 100 epochs done
  uv run python train_dino.py --config configs/pretrain/dino.yaml
"""

import os, argparse, json, copy, yaml, glob
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from encoders.dino import (
    DINOEncoder, DINOHead, DINOLoss,
    get_multicrop_transform, update_teacher, cosine_scheduler,
)
from data.stl10_data import get_dino_unlabeled_loader


# ──────────────────────────── resume helpers ─────────────────────────────────

RESUME_CKPT_NAME = "dino_resume.pt"


def _save_resume_checkpoint(
    path: str,
    epoch: int,
    global_step: int,
    best_loss: float,
    history: list,
    student, teacher, student_head, teacher_head,
    optimizer, criterion,
):
    """Save everything needed to resume training exactly where we left off."""
    torch.save({
        "epoch":              epoch,
        "global_step":        global_step,
        "best_loss":          best_loss,
        "history":            history,
        "student":            student.state_dict(),
        "teacher":            teacher.state_dict(),
        "student_head":       student_head.state_dict(),
        "teacher_head":       teacher_head.state_dict(),
        "optimizer":          optimizer.state_dict(),
        "criterion_center":   criterion.center.clone(),
    }, path)


def _load_resume_checkpoint(
    path: str,
    device: str,
    student, teacher, student_head, teacher_head,
    optimizer, criterion,
) -> tuple:
    """Load resume checkpoint and return (start_epoch, global_step, best_loss, history)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    student.load_state_dict(ckpt["student"])
    teacher.load_state_dict(ckpt["teacher"])
    student_head.load_state_dict(ckpt["student_head"])
    teacher_head.load_state_dict(ckpt["teacher_head"])
    optimizer.load_state_dict(ckpt["optimizer"])
    criterion.center.copy_(ckpt["criterion_center"])
    return (
        ckpt["epoch"],        # last completed epoch (0-indexed)
        ckpt["global_step"],
        ckpt["best_loss"],
        ckpt["history"],
    )


def _cleanup_intermediate_checkpoints(ckpt_dir: str):
    """Delete resume and best-loss checkpoints, keeping only the final encoder."""
    for fname in [RESUME_CKPT_NAME, "dino_encoder_best.pt"]:
        fpath = os.path.join(ckpt_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            print(f"[DINO] Deleted intermediate checkpoint: {fname}")


# ──────────────────────────────── training ───────────────────────────────────

def train(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_epochs = cfg["epochs"]
    resume_interval = cfg.get("resume_interval", 20)

    print(f"[DINO] device={device}  variant={cfg['variant']}  "
          f"total_epochs={n_epochs}  resume_interval={resume_interval}")

    torch.manual_seed(cfg.get("seed", 42))

    n_local_crops = cfg.get("n_local_crops", 6)
    n_crops       = 2 + n_local_crops                         # total student views

    mc_transform = get_multicrop_transform(
        img_size=cfg.get("img_size", 96),
        n_local_crops=n_local_crops,
    )
    loader = get_dino_unlabeled_loader(
        multicrop_transform=mc_transform,
        root=cfg.get("data_root", "./data"),
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 4),
    )
    n_steps = len(loader)
    print(f"[DINO] unlabeled images: {len(loader.dataset):,}  "
          f"steps/epoch: {n_steps}  n_crops: {n_crops}")

    # Student and teacher (teacher = deep copy, frozen)
    student = DINOEncoder(
        variant=cfg["variant"],
        img_size=cfg.get("img_size", 96),
        patch_size=cfg.get("patch_size", 16),
    ).to(device)
    teacher = copy.deepcopy(student).to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    in_dim  = student.embed_dim
    out_dim = cfg.get("out_dim", 4096)

    student_head = DINOHead(
        in_dim=in_dim, out_dim=out_dim,
        hidden_dim=cfg.get("hidden_dim", 2048),
        bottleneck_dim=cfg.get("bottleneck_dim", 256),
        norm_last_layer=True,
    ).to(device)
    teacher_head = copy.deepcopy(student_head).to(device)
    for p in teacher_head.parameters():
        p.requires_grad = False

    # Loss
    criterion = DINOLoss(
        out_dim=out_dim,
        n_crops=n_crops,
        warmup_teacher_temp=cfg.get("warmup_teacher_temp", 0.04),
        teacher_temp=cfg.get("teacher_temp", 0.07),
        warmup_teacher_temp_epochs=cfg.get("warmup_teacher_temp_epochs", 30),
        n_epochs=n_epochs,
        student_temp=cfg.get("student_temp", 0.1),
    ).to(device)

    # Optimizer
    params_groups = [
        {"params": student.parameters(),      "lr": cfg["base_lr"]},
        {"params": student_head.parameters(), "lr": cfg["base_lr"]},
    ]
    optimizer = AdamW(params_groups, weight_decay=cfg.get("weight_decay", 0.04))

    # LR and WD schedules (cosine) — computed over ALL epochs so schedule is
    # consistent regardless of which chunk we're in
    total_steps = n_steps * n_epochs
    lr_schedule = cosine_scheduler(
        base=cfg["base_lr"], end=cfg.get("min_lr", 1e-6),
        n_epochs=n_epochs, n_steps_per_epoch=n_steps,
    )
    wd_schedule = cosine_scheduler(
        base=cfg.get("weight_decay", 0.04), end=cfg.get("weight_decay_end", 0.4),
        n_epochs=n_epochs, n_steps_per_epoch=n_steps,
    )
    # Teacher momentum schedule (cosine from start_mom to 1.0)
    mom_schedule = cosine_scheduler(
        base=cfg.get("teacher_momentum_start", 0.996),
        end=1.0,
        n_epochs=n_epochs, n_steps_per_epoch=n_steps,
    )

    freeze_last_layer = cfg.get("freeze_last_layer_epochs", 1)
    ckpt_dir  = cfg.get("ckpt_dir", "results/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Resume from checkpoint if one exists ─────────────────────────────────
    resume_path = os.path.join(ckpt_dir, RESUME_CKPT_NAME)
    start_epoch = 0
    global_step = 0
    best_loss   = float("inf")
    history     = []

    if os.path.exists(resume_path):
        print(f"[DINO] Found resume checkpoint → loading {resume_path}")
        start_epoch, global_step, best_loss, history = _load_resume_checkpoint(
            resume_path, device,
            student, teacher, student_head, teacher_head,
            optimizer, criterion,
        )
        print(f"[DINO] Resuming from epoch {start_epoch + 1}  "
              f"(global_step={global_step}, best_loss={best_loss:.4f})")
    else:
        print(f"[DINO] No resume checkpoint found — starting from scratch")

    # Check if training is already complete
    if start_epoch >= n_epochs:
        print(f"[DINO] Training already complete ({start_epoch}/{n_epochs} epochs).")
        return student

    # Compute stop epoch for this run
    stop_epoch = min(start_epoch + resume_interval, n_epochs)
    print(f"[DINO] This run: epochs {start_epoch + 1} → {stop_epoch}  "
          f"({stop_epoch - start_epoch} epochs)")

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, stop_epoch):
        student.train(); student_head.train()
        epoch_loss, n_batches = 0.0, 0
        pbar = tqdm(loader, desc=f"DINO epoch {epoch+1:3d}/{n_epochs}", leave=False)

        for crops in pbar:
            # Apply per-step LR and WD
            lr = lr_schedule[min(global_step, total_steps - 1)].item()
            wd = wd_schedule[min(global_step, total_steps - 1)].item()
            for pg in optimizer.param_groups:
                pg["lr"] = lr
                pg["weight_decay"] = wd

            # All crops → device
            crops = [c.to(device, non_blocking=True) for c in crops]

            # Student processes ALL crops
            all_crops_cat = torch.cat(crops, dim=0)        # (B * n_crops, C, H, W)
            student_out   = student_head(student.encode(all_crops_cat))

            # Teacher processes only 2 global crops (no gradient)
            with torch.no_grad():
                teacher_out = teacher_head(teacher.encode(torch.cat(crops[:2], dim=0)))

            loss = criterion(student_out, teacher_out, epoch)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clip
            nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(student_head.parameters()),
                max_norm=cfg.get("clip_grad", 3.0),
            )
            # Freeze last layer of head for the first N epochs (stabilises training)
            if epoch < freeze_last_layer:
                student_head.last_layer.parametrizations.weight.original0.grad = None

            optimizer.step()

            # EMA teacher update
            momentum = mom_schedule[min(global_step, total_steps - 1)].item()
            update_teacher(student, teacher, momentum)
            update_teacher(student_head, teacher_head, momentum)

            epoch_loss += loss.item(); n_batches += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        avg_loss = epoch_loss / n_batches
        history.append({"epoch": epoch + 1, "loss": avg_loss})
        print(f"[DINO] epoch {epoch+1:3d}/{n_epochs} | loss {avg_loss:.4f} | lr {lr:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(student.state_dict(), os.path.join(ckpt_dir, "dino_encoder_best.pt"))

    # ── Post-chunk: save or finalize ─────────────────────────────────────────
    training_complete = (stop_epoch >= n_epochs)

    if training_complete:
        # Save final student encoder
        torch.save(student.state_dict(), os.path.join(ckpt_dir, "dino_encoder.pt"))
        with open(os.path.join(ckpt_dir, "dino_training_log.json"), "w") as f:
            json.dump({"history": history, "best_loss": best_loss, "config": cfg}, f, indent=2)

        # Clean up all intermediate checkpoints
        _cleanup_intermediate_checkpoints(ckpt_dir)

        print(f"\n[DINO] ✓ Pretraining COMPLETE ({n_epochs}/{n_epochs} epochs). "
              f"Best loss: {best_loss:.4f}")
        print(f"[DINO] Final encoder saved → {ckpt_dir}/dino_encoder.pt")
    else:
        # Save resume checkpoint and stop
        _save_resume_checkpoint(
            resume_path, stop_epoch, global_step, best_loss, history,
            student, teacher, student_head, teacher_head,
            optimizer, criterion,
        )
        # Also save a partial training log
        with open(os.path.join(ckpt_dir, "dino_training_log.json"), "w") as f:
            json.dump({"history": history, "best_loss": best_loss,
                        "epochs_completed": stop_epoch, "epochs_total": n_epochs,
                        "config": cfg}, f, indent=2)

        print(f"\n[DINO] ⏸ Paused at epoch {stop_epoch}/{n_epochs}. "
              f"Best loss so far: {best_loss:.4f}")
        print(f"[DINO] Resume checkpoint saved → {resume_path}")
        print(f"[DINO] Run the same command again to continue training.")

    return student


# ──────────────────────────────── main ───────────────────────────────────────

DEFAULT_CFG = {
    "variant":                    "ViT-Small/16",
    "img_size":                   96,
    "patch_size":                 16,
    "epochs":                     100,
    "resume_interval":            20,
    "batch_size":                 64,
    "base_lr":                    5e-4,
    "min_lr":                     1e-6,
    "weight_decay":               0.04,
    "weight_decay_end":           0.4,
    "teacher_momentum_start":     0.996,
    "out_dim":                    4096,
    "hidden_dim":                 2048,
    "bottleneck_dim":             256,
    "n_local_crops":              6,
    "student_temp":               0.1,
    "warmup_teacher_temp":        0.04,
    "teacher_temp":               0.07,
    "warmup_teacher_temp_epochs": 30,
    "freeze_last_layer_epochs":   1,
    "clip_grad":                  3.0,
    "num_workers":                4,
    "data_root":                  "./data",
    "ckpt_dir":                   "results/checkpoints",
    "seed":                       42,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINO pretraining on STL-10 unlabeled")
    parser.add_argument("--config", default=None,
                        help="Path to YAML config (overrides defaults)")
    args = parser.parse_args()

    cfg = DEFAULT_CFG.copy()
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))

    train(cfg)

