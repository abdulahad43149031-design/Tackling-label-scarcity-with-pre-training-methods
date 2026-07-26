# Project Context: SSL Pretraining Fusion for Label-Scarce Image Classification

**Researcher:** Abdul Ahad, AI/ML B.Tech student, IIIT Nagpur (with a senior collaborator)
**Status as of this summary:** Repo restructured; scratch baseline script updated and ready to run. Encoder pretraining (MAE/DINO/Diffusion) not yet built.

---

## 1. Research Question

Does self-supervised pretraining (MAE, DINO, diffusion-based feature extraction) on a **fixed, small unlabeled image pool** produce embeddings that require fewer labels downstream than training a classifier from scratch on the same labeled budget? And specifically: do MAE, DINO, and diffusion learn **complementary** representations — such that fusing them outperforms any single method or pairwise combination?

**Important framing constraint:** This is *not* a claim that our pretraining (on ~100K unlabeled images) will rival ImageNet-scale (14M images) pretrained models in absolute accuracy. The research question is **relative and within-budget**: given a fixed small unlabeled pool, which pretraining strategy (or fusion of strategies) is most label-efficient? Limitations section must explicitly state we are not claiming ImageNet-pretrained-model-level absolute accuracy.

---

## 2. Dataset Decision: STL-10 (locked in)

| Reason | Detail |
|---|---|
| Unlabeled pool | 100,000 images, **disjoint** from the labeled set |
| Labeled train/test | 5,000 / 8,000 images, 10 classes (balanced) |
| Why not CIFAR-100 | No unlabeled split — pretraining would reuse the same images used for scarcity evaluation, contaminating the comparison |
| Resolution | 96×96 (native; patch_size=16 → 36 patches, not upsampled to 224×224) |
| Classes (order matters, used in code) | `airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck` |

---

## 3. Two-Phase Pipeline

**Phase 1 — Pretraining (unlabeled data, once per encoder, then frozen forever):**
- **MAE:** ViT-Small/16 encoder + lightweight decoder, trained via masked patch reconstruction (~75% masking). Decoder discarded; only encoder kept → `results/checkpoints/mae_encoder.pt`
- **DINO:** Student + teacher ViT-Small/16, self-distillation across augmented crops, teacher = EMA of student. Teacher discarded; only student kept → `results/checkpoints/dino_encoder.pt`
- **Diffusion:** No training — use a pretrained diffusion U-Net, add noise at fixed timestep `t`, hook an intermediate block's activation, spatially pool → embedding. Timestep and hooked layer are hyperparameters requiring a small sweep before locking in.

**Phase 2 — Downstream eval (scarce labeled data, every config):**
- Frozen encoder(s) → embedding(s) → [EmbeddingFusion if >1 encoder] → linear head → class prediction.
- **Linear probe protocol** (primary): encoders fully frozen, only head (+ fusion module) trains.

---

## 4. Experiment Matrix (8 configs total)

| # | Config | Frozen | Trainable | Script |
|---|---|---|---|---|
| 0 | **ViT-Small scratch baseline** | none | full ViT + head | `train_baseline.py` |
| 1 | MAE standalone | MAE encoder | linear head | `train_downstream.py` |
| 2 | DINO standalone | DINO encoder | linear head | `train_downstream.py` |
| 3 | Diffusion standalone | Diffusion U-Net | linear head | `train_downstream.py` |
| 4 | MAE + DINO | both | fusion module + head | `train_downstream.py` |
| 5 | MAE + Diffusion | both | fusion module + head | `train_downstream.py` |
| 6 | DINO + Diffusion | both | fusion module + head | `train_downstream.py` |
| 7 | **MAE + DINO + Diffusion (trio)** | all three | fusion module + head | `train_downstream.py` |

**Pairwise configs (4, 5, 6) are essential** — without them, can't attribute why trio wins/loses.

**Label fractions:** `[0.01, 0.05, 0.10, 1.0]` × seeds `[0, 1, 2]` for every config.

---

## 5. Reporting Convention (locked in)

- **Per run:** `smoothed_acc = mean(test_acc[-10:])` — last 10 epochs, pre-decided, not argmax. Kills oscillation noise without touching the test-set-selection problem.
- **Per config/fraction:** `mean ± std` across 3 seeds. Standard practice in DINO/SimCLR/MoCo papers.
- Both `smoothed_acc` and `final_acc` (from best-loss checkpoint) saved in `metrics.json`; `smoothed_acc` is the primary reported metric.

---

## 6. Fusion Module Design

`EmbeddingFusion` class (`fusion/fusion_module.py`), two modes:
- **`concat_proj`** (default/primary): concatenate frozen embeddings → `Linear → LayerNorm → GELU` → `fused_dim` vector. Preferred because MAE/DINO/Diffusion embeddings come from different objectives and don't share a common space.
- **`cross_attn`** (fallback): multi-head attention across per-encoder embeddings, lets the model weight encoders per-image.

Only the fusion module + head train in duo/trio configs; encoders never unfreeze.

---

## 7. CKA — the "go/no-go gate"

Centered Kernel Alignment between every pair of frozen encoders' embeddings, computed **before** trusting fusion results. Low similarity → encoders are learning different things (fusion has a real chance). High similarity → fusion likely won't help (valid negative result).

---

## 8. ViT Architecture

Implementing **"An Image is Worth 16x16 Words" (Dosovitskiy et al. 2021)** from scratch (not via timm). Used identically as the backbone for the scratch baseline AND as the encoder architecture for MAE/DINO (architecture-controlled comparisons).

**Active variant: ViT-Small/16** (~22M params) — appropriate for STL-10's 5K labeled images. ViT-Base (86M) was considered but is oversized for this data regime.

| Variant | Patch | Embed dim | Depth | Heads | MLP dim | Params |
|---|---|---|---|---|---|---|
| **ViT-Small/16** ← active | 16 | 512 | 8 | 8 | 2048 | ~22M |
| ViT-Base/16 | 16 | 768 | 12 | 12 | 3072 | ~86M |
| ViT-Large/16 | 16 | 1024 | 24 | 16 | 4096 | ~307M |

**Hyperparameters (locked-in):**

| Hyperparameter | Paper value | Our value | Why |
|---|---|---|---|
| Optimizer | Adam, β1=0.9, β2=0.999 | same | not dataset-size dependent |
| Weight decay | 0.1 | same | not dataset-size dependent |
| Gradient clipping | global norm 1.0 | same | not dataset-size dependent |
| Dropout | 0.1 | same | not dataset-size dependent |
| LR | — | 0.0001 | tuned for STL-10 scale |
| LR schedule shape | linear warmup → linear decay | same shape | just shorter |
| Batch size | 4096 | 64, capped at `min(64, len(train_set))` | 4096 > total images at low fractions |
| Warmup | 10,000 steps | `warmup_fraction=0.1` of total steps | 10K would exceed total training |
| Epochs | — | 350 (baseline) | tuned |
| Smoothing window | — | last 10 epochs | pre-decided reporting convention |

---

## 9. Repo Structure

```
research/
├── train_baseline.py          # Phase 0: ViT-Small scratch, 4 fracs × 3 seeds
├── train_mae.py               # Phase 1a: pretrain MAE on 100K unlabeled (stub)
├── train_dino.py              # Phase 1b: pretrain DINO on 100K unlabeled (stub)
├── extract_features.py        # Phase 1c: diffusion U-Net feature extraction (stub)
├── train_downstream.py        # Phase 2: linear probe for all non-baseline configs (stub)
├── run_sweep.py               # orchestrator: loops all configs × fracs × seeds (stub)
│
├── data/
│   ├── stl10_data.py          # stratified label-fraction loader, shared across all configs
│   └── stl10_binary/          # raw dataset files (git-ignored)
│
├── models/
│   └── vit.py                 # ViT-Small/16 + Base/Large (paper-faithful, from scratch)
│
├── encoders/
│   ├── base.py                # BaseEncoder ABC: encode(x) → (B, embed_dim)
│   ├── mae.py                 # MAE encoder wrapper (stub)
│   ├── dino.py                # DINO student encoder wrapper (stub)
│   └── diffusion.py           # pretrained U-Net hook → pooled embedding (stub)
│
├── fusion/
│   └── fusion_module.py       # EmbeddingFusion: concat_proj + cross_attn
│
├── eval/
│   ├── linear_probe.py        # linear probe harness (stub)
│   └── cka.py                 # CKA go/no-go gate (stub)
│
├── configs/
│   ├── baseline/
│   │   └── vit_small_scratch.yaml
│   ├── standalone/
│   │   ├── mae_probe.yaml
│   │   ├── dino_probe.yaml
│   │   └── diffusion_probe.yaml
│   └── fusion/
│       ├── mae_dino.yaml
│       ├── mae_diffusion.yaml
│       ├── dino_diffusion.yaml
│       └── mae_dino_diffusion.yaml   # trio — the full fusion
│
├── results/
│   ├── runs/                  # results/runs/<config>/<frac>/<seed>/{config.yaml, metrics.json, checkpoint.pt}
│   ├── plots/                 # label_efficiency_curves, cka_heatmap, fusion_bar, per-run plots
│   ├── checkpoints/           # mae_encoder.pt, dino_encoder.pt (saved after Phase 1)
│   └── summary.csv            # aggregated mean ± std across seeds
│
├── app/                       # Streamlit demo (built last)
│
├── pyproject.toml             # uv-managed; torch/torchvision pinned to cu128 index
├── uv.lock
└── .gitignore                 # git-ignores: data/stl10_binary/, results/runs/, .venv/
```

**Key gotcha fixed:** Output paths are now `results/runs/<config>/<frac>/<seed>/` — deterministic, never silently overwritten. Old `run_name`-based naming is gone.

---

## 10. Environment Setup

- Package manager: **uv**, not pip.
- Windows machine, RTX 5050 (Blackwell, sm_120).
- `pyproject.toml` pins `torch`/`torchvision` to the `cu128` PyTorch index (CUDA 12.8, covers Blackwell).
- Setup: `git clone` → `uv sync` → `uv run python train_baseline.py`

---

## 11. Current Status

| Item | Status |
|---|---|
| Repo restructured | ✅ Done |
| `train_baseline.py` (ViT-Small, 4 fracs × 3 seeds) | ✅ Ready to run |
| `models/vit.py` (ViT-Small/16 architecture) | ✅ Done |
| `fusion/fusion_module.py` (EmbeddingFusion) | ✅ Done |
| `encoders/base.py` (BaseEncoder ABC) | ✅ Done |
| MAE pretraining (`train_mae.py`, `encoders/mae.py`) | 🔲 Stub — next |
| DINO pretraining (`train_dino.py`, `encoders/dino.py`) | 🔲 Stub |
| Diffusion extraction (`extract_features.py`, `encoders/diffusion.py`) | 🔲 Stub |
| `eval/linear_probe.py` | 🔲 Stub |
| `eval/cka.py` | 🔲 Stub |
| `train_downstream.py` | 🔲 Stub |
| Baseline 4-fraction sweep results | 🔲 Not yet run |

---

## 12. Next Steps (in order)

1. Run `train_baseline.py` across all 4 fractions × 3 seeds → first scarcity curve (the floor).
2. Build `encoders/mae.py` + `train_mae.py` → pretrain MAE on 100K unlabeled STL-10.
3. Build `encoders/dino.py` + `train_dino.py` → pretrain DINO similarly.
4. Build `encoders/diffusion.py` + `extract_features.py` → sweep timestep/layer, cache embeddings.
5. Implement `eval/linear_probe.py` + `train_downstream.py` → run same 4-frac × 3-seed sweep for all 7 non-baseline configs.
6. Run `eval/cka.py` between all three frozen encoders (go/no-go check).
7. Aggregate into `results/summary.csv` + plots.
8. Build Streamlit demo (`app/`).
