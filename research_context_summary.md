# Project Context: SSL Pretraining Fusion for Label-Scarce Image Classification

**Researcher:** Abdul Ahad, AI/ML B.Tech student, IIIT Nagpur (with a senior collaborator)
**Status as of this summary:** Scratch baseline implemented and running; pretraining phase (MAE/DINO/Diffusion) not yet built.

---

## 1. Research Question

Does self-supervised pretraining (MAE, DINO, diffusion-based feature extraction) on a **fixed, small unlabeled image pool** produce embeddings that require fewer labels downstream than training a classifier from scratch on the same labeled budget? And specifically: do MAE, DINO, and diffusion learn **complementary** representations — such that fusing them outperforms any single method or pairwise combination?

**Important framing constraint (raised by senior, resolved):** This is *not* a claim that our pretraining (on ~100K unlabeled images) will rival ImageNet-scale (14M images) pretrained models in absolute accuracy. It can't, and we don't claim it does. The research question is **relative and within-budget**: given a fixed small unlabeled pool, which pretraining strategy (or fusion of strategies) is most label-efficient? This mirrors the framing used in SimCLR/MoCo/DINO papers, which report linear-probe accuracy *as a function of label fraction*, not absolute SOTA. Limitations section must explicitly state we are not claiming ImageNet-pretrained-model-level absolute accuracy.

---

## 2. Dataset Decision: STL-10 (locked in, not CIFAR-100)

| Reason | Detail |
|---|---|
| Unlabeled pool | 100,000 images, **disjoint** from the labeled set |
| Labeled train/test | 5,000 / 8,000 images, 10 classes (balanced) |
| Why not CIFAR-100 | No unlabeled split exists — pretraining would have to reuse the same images later used for scarcity evaluation, contaminating the comparison |
| Resolution | 96×96 (native; not upsampled to 224×224 to match the original ViT paper — patch16 gives 36 patches instead of 196, faster training, deliberate choice) |
| Classes (order matters, used in code) | `airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck` |

---

## 3. Two-Phase Pipeline (applies to every pretraining method)

**Phase 1 — Pretraining (unlabeled data, once per encoder, then frozen forever):**
- **MAE:** ViT encoder + lightweight decoder, trained via masked patch reconstruction (~75% masking). Decoder discarded after training; only encoder kept.
- **DINO:** Student + teacher ViT (identical architecture), self-distillation across augmented crops, teacher = EMA of student. Teacher discarded; only student kept.
- **Diffusion:** **No training** — use a pretrained diffusion U-Net, add noise to an image at a fixed timestep `t`, hook an intermediate block's activation, spatially pool → embedding. Timestep and hooked layer are unconstrained hyperparameters requiring a small sweep before locking in.

**Phase 2 — Downstream eval (scarce labeled data, every config):**
- Frozen encoder(s) → embedding(s) → [fusion module, if >1 encoder] → linear head → class prediction.
- **Linear probe protocol** (primary): encoder(s) fully frozen, only head (+ fusion module if present) trains.
- Fine-tuning (secondary/optional): unfreeze encoder at low LR, gives a ceiling number — less clean for comparing representation quality.

---

## 4. Experiment Matrix (8 configs total)

| Config | Frozen | Trainable | Pretraining needed |
|---|---|---|---|
| **ViT scratch baseline** (no pretraining) | none | full ViT (encoder + head) | No — trained end-to-end from random init directly on scarce labels. This is the **floor** every other config must beat. |
| MAE standalone | MAE encoder | linear head | Yes, once |
| DINO standalone | DINO encoder | linear head | Yes, once |
| Diffusion standalone | Diffusion U-Net | linear head | No (pretrained ckpt, feature extraction only) |
| MAE + DINO | both | fusion module + head | No (reuse frozen encoders) |
| MAE + Diffusion | both | fusion module + head | No |
| DINO + Diffusion | both | fusion module + head | No |
| MAE + DINO + Diffusion (trio) | all three | fusion module + head | No |

**Pairwise configs are NOT redundant** — without them, can't tell *why* trio fusion wins/loses (which pair drives the effect, whether one encoder is dead weight). They're cheap: no new pretraining, just a small fusion module + head on top of already-frozen encoders.

**Label fractions used across every config:** `[0.01, 0.05, 0.10, 1.0]` (1%, 5%, 10%, 100% of the 5,000-image labeled train set), each run across multiple seeds (e.g. `[0, 1, 2]`).

---

## 5. Fusion Module Design

`EmbeddingFusion` class, two modes:
- **`concat_proj`** (default/primary): concatenate frozen embeddings → `Linear → LayerNorm → GELU` → fixed-size fused vector. Preferred over naive addition (addition forces same dimensionality + comparable embedding *spaces*, which MAE/DINO embeddings don't share — they come from different objectives).
- **`cross_attn`** (fallback if concat underperforms): multi-head attention across the set of per-encoder embeddings, lets the model weight encoders per-image rather than fixed 50/50.

Only the fusion module + head train in pairwise/trio configs; encoders never unfreeze.

---

## 6. CKA — the "go/no-go gate"

Centered Kernel Alignment between every pair of frozen encoders' embeddings, computed **before** trusting fusion results. Tells you whether MAE/DINO/Diffusion are actually learning different things. Low similarity → fusion has a real chance to help; high similarity → fusion likely won't help, and that's a valid negative result, not a failure.

---

## 7. ViT Architecture — Faithfulness to Original Paper

Implementing **"An Image is Worth 16x16 Words" (Dosovitskiy et al. 2021)** architecture from scratch (not via timm), used identically as the backbone for the scratch baseline AND as the encoder architecture for MAE/DINO later (so comparisons are architecture-controlled).

**Variant table (`VIT_CONFIGS` in `models/vit_paper.py`):**

| Variant | Patch | Embed dim | Depth | Heads | MLP ratio | Params |
|---|---|---|---|---|---|---|
| ViT-Base/16 | 16 | 768 | 12 | 12 | 4.0 (3072 hidden) | 86M |
| ViT-Large/16 | 16 | 1024 | 24 | 16 | 4.0 (4096 hidden) | 307M |
| ViT-Huge/14 | 14 | 1280 | 32 | 16 | 4.0 (5120 hidden) | 632M |

Currently using **ViT-Base/16** for the scratch baseline. Open question (unresolved): ViT-Base (86M params) is vastly oversized relative to STL-10's 5,000-image labeled set — may need to size down to a Small/Tiny-scale variant if overfitting dominates every fraction including 100%.

**Hyperparameters — literal vs. adapted (locked-in table, important for methods section):**

| Hyperparameter | Paper value | Our value | Why |
|---|---|---|---|
| Optimizer | Adam, β1=0.9, β2=0.999 | same | not dataset-size dependent |
| Weight decay | 0.1 | same | not dataset-size dependent |
| Gradient clipping | global norm 1.0 | same | not dataset-size dependent |
| Dropout | 0.1 | same | not dataset-size dependent |
| LR schedule shape | linear warmup → linear decay | same shape | just shorter |
| Batch size | 4096 | adapted, capped at `min(batch_size, len(train_set))` | 4096 > total images at 1%/5%/10% fractions — mathematically impossible otherwise |
| Warmup steps | 10,000 | `warmup_fraction: 0.1` of total steps | 10K steps would exceed total training length on this dataset |

**Justification for adapting rather than copying literally:** the paper's own headline finding is that ViT underperforms CNNs without massive (JFT-300M-scale) pretraining data — copying the literal recipe onto a tiny dataset isn't more faithful, it's missing the paper's point. Defensible position: keep everything that defines the *method* (architecture, optimizer identity, weight decay, dropout, clip norm, schedule shape); only adapt what's mathematically tied to JFT-scale data (batch size, warmup length, total steps).

---

## 8. Repo Structure (established, full annotated version exists as a PDF: `research_project_structure.pdf`)

```
research/
├── encoders/          # mae_vit.py, dino_vit.py, diffusion_features.py, base_encoder.py (shared interface)
├── fusion/            # fusion_module.py (EmbeddingFusion)
├── eval/              # linear_probe.py, cka.py
├── configs/           # one YAML per experiment (8 configs)
├── results/
│   ├── runs/          # results/runs/<run_name>_<date>/{config.yaml, metrics.json, checkpoint.pt}
│   ├── plots/         # label_efficiency_curves.png, cka_heatmap.png, fusion_comparison_bar.png, per-run training curves/confusion matrices/sample predictions
│   ├── summary.csv
│   └── aggregate_results.py
├── app/               # streamlit_app.py (interactive demo + static results viewer), utils.py
├── data/              # stl10_data.py (stratified label-fraction loader, shared across all configs)
├── models/            # vit_paper.py (ViT-Base/16/Large/Huge per original paper)
├── train.py           # unified entrypoint, branches on len(encoders) in config
├── train_vit_scratch.py   # the no-pretraining baseline script (currently running)
└── run_eval_sweep.py  # loops all configs/*.yaml
```

**Critical gotcha already hit:** `train_vit_scratch.py` names output files from `cfg['run_name']`, NOT from `label_fraction`. Forgetting to change `run_name` when changing `label_fraction` silently overwrites the previous run's results with no warning.

---

## 9. Environment Setup (working, resolved)

- Package manager: **uv**, not pip.
- Windows machine, RTX 5050 (Blackwell, sm_120).
- `pyproject.toml` uses `[[tool.uv.index]]` + `[tool.uv.sources]` to pin `torch`/`torchvision` to the `cu128` PyTorch index (`https://download.pytorch.org/whl/cu128`) — CUDA 12.8 covers Ampere (30-series)/Ada Lovelace (40-series)/Blackwell (50-series) in one wheel.
- Other deps (`matplotlib`, `pyyaml`, `scikit-learn`) resolve from normal PyPI, no special index needed.
- Setup flow for anyone cloning: `git clone` → `uv sync` → `uv run python train_vit_scratch.py` (no manual venv activation required, though optional).
- **Resolved issue:** `PermissionError` during STL-10 extraction on Windows — caused by a partially-extracted folder from a previous failed download (likely OneDrive-related path). Fixed by deleting `data/stl10_binary/` and `data/stl10_binary.tar.gz` and re-running.

---

## 10. Current Status / Open Threads

- **Running now:** `train_vit_scratch.py`, ViT-Base/16, STL-10, `label_fraction: 1.0`, batch_size 64, 50 epochs. Early epochs (1-5) showed noisy accuracy (~20-29%) — flagged as plausibly still within LR warmup (~5 epochs at this batch size/dataset size), not necessarily broken. Decision pending: let it finish, then judge; if still flat/noisy past epoch 20, lower `base_lr` to `0.0001` or downsize the ViT variant.
- **Reporting convention clarified:** the script currently reports **final-epoch accuracy**, not best-epoch — `evaluate()` is called once after the training loop ends using whatever weights exist at the last epoch. A best-checkpoint-tracking patch was drafted (save `best_state` whenever `test_acc` improves, reload before final eval) but not yet applied — **decide and apply before treating any results as final.**
- **Batch size silently adapts per fraction:** `actual_batch_size = min(batch_size, len(train_set))` — e.g. at 1% (50 images) batch size silently becomes 50, not 64. Not a bug, but worth disclosing in methods (different fractions train with different effective batch sizes, not just different data amounts).
- **LR should stay fixed across the 4 label-fraction runs within this baseline** (so the sweep isolates data-quantity effects, not retuned optimization). LR does *not* need to match across different *configs* later (e.g. scratch ViT vs. MAE linear probe) — different optimization problems, each gets appropriately tuned within the shared protocol (same fractions/seeds/splits/metric).
- **Not yet built:** MAE pretraining script, DINO pretraining script, diffusion feature extractor, fusion training script, CKA script, results aggregation script, Streamlit demo app. All were designed/discussed conceptually and as code sketches, but not yet implemented as the final on-disk versions tied to this STL-10 setup.
- **This sandbox cannot execute the actual training** (no GPU, network locked to package indexes only, disk too small for full CUDA torch install) — all code so far has been reviewed by reasoning through it, not run end-to-end in chat. All real runs happen on Abdul's local RTX 5050 machine.

---

## 11. Next Steps (in order)

1. Resolve the scratch-baseline training instability (or confirm it's just warmup) and finalize best-vs-final-epoch reporting choice.
2. Complete the 4-fraction sweep (`0.01/0.05/0.10/1.0`) for the scratch baseline with a fixed LR → first scarcity curve.
3. Build `encoders/mae_vit.py` and pretrain MAE on STL-10's 100K unlabeled images.
4. Build `encoders/dino_vit.py`, pretrain DINO similarly.
5. Build linear-probe harness, rerun the same 4-fraction sweep on top of each frozen encoder → compare against scratch baseline.
6. Build `diffusion_features.py` (pretrained U-Net feature hook), sweep timestep/layer choice.
7. Run CKA between all three frozen encoders (go/no-go check).
8. Build fusion configs (3 pairwise + 1 trio), rerun sweep.
9. Aggregate everything into `summary.csv` + plots; build Streamlit demo.
