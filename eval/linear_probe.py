"""
Linear probe evaluation harness — Phase 2 downstream evaluation.

Supports all 7 non-baseline configs:
  - Standalone (1 encoder, no fusion): standalone/{mae,dino,diffusion}_probe.yaml
  - Duo fusion (2 encoders): fusion/{mae_dino, mae_diffusion, dino_diffusion}.yaml
  - Trio fusion (3 encoders): fusion/mae_dino_diffusion.yaml

Reporting convention:
  smoothed_acc = mean(test_acc[-smoothing_window:])   ← primary metric
  mean ± std across seeds (computed by run_sweep.py)

Pre-extracts embeddings once per encoder before training the head → fast epoch iterations.
"""

import os, json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from data.stl10_data import get_labeled_loaders, get_all_images_loader, CLASS_NAMES
from fusion.fusion_module import EmbeddingFusion


# ─────────────────────────────── encoder loading ─────────────────────────────

def load_encoder(enc_cfg: dict, device: str):
    """
    Load a single frozen encoder from config dict.
    enc_cfg keys: name, ckpt (optional for diffusion), variant, + diffusion-specific fields.
    Returns: encoder (frozen BaseEncoder subclass)
    """
    name = enc_cfg["name"]

    if name == "mae":
        from encoders.mae import MAEEncoder
        enc = MAEEncoder(variant=enc_cfg.get("variant", "ViT-Small/16"))
        enc.load_state_dict(torch.load(enc_cfg["ckpt"], map_location=device))
        enc.freeze()
        enc.to(device)
        return enc

    elif name == "dino":
        from encoders.dino import DINOEncoder
        enc = DINOEncoder(variant=enc_cfg.get("variant", "ViT-Small/16"))
        enc.load_state_dict(torch.load(enc_cfg["ckpt"], map_location=device))
        enc.freeze()
        enc.to(device)
        return enc

    elif name == "diffusion":
        # Diffusion embeddings are pre-cached to disk — return a wrapper that loads from cache
        return _DiffusionCacheEncoder(enc_cfg)

    else:
        raise ValueError(f"Unknown encoder name: '{name}'")


class _DiffusionCacheEncoder:
    """
    Thin wrapper that serves pre-cached diffusion embeddings from disk.
    Not a real nn.Module — used only inside extract_all_embeddings().
    """
    def __init__(self, enc_cfg: dict):
        cache_dir = enc_cfg.get(
            "cache_dir", "results/checkpoints/diffusion_embeddings"
        )
        self._cache = {
            split: torch.load(os.path.join(cache_dir, f"{split}.pt"))
            for split in ("train", "test")
            if os.path.exists(os.path.join(cache_dir, f"{split}.pt"))
        }
        # embed_dim inferred from cached tensors
        sample_emb    = next(iter(self._cache.values()))["embeddings"]
        self._embed_dim = sample_emb.shape[-1]

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def get_cached(self, split: str):
        """Returns (embeddings, labels) tensors for a given split."""
        d = self._cache[split]
        return d["embeddings"], d["labels"]


# ──────────────────────────── embedding extraction ───────────────────────────

@torch.no_grad()
def _extract_from_loader(encoder, loader, device: str):
    """Run encoder over all images in a loader → (N, embed_dim), (N,) labels."""
    all_embs, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="  pre-extracting", leave=False):
        imgs   = imgs.to(device, non_blocking=True)
        embs   = encoder.encode(imgs).cpu()
        all_embs.append(embs)
        all_labels.append(labels)
    return torch.cat(all_embs), torch.cat(all_labels)


def extract_all_embeddings(encoders: list, cfg: dict, device: str):
    """
    Pre-extract embeddings from all encoders for the labeled train and test sets.
    Returns:
      train_embs: list of (N_train, embed_dim_i) tensors, one per encoder
      test_embs : list of (N_test,  embed_dim_i) tensors, one per encoder
      train_labels, test_labels: (N,) int tensors
    """
    train_loader, test_loader, _ = get_labeled_loaders(
        root=cfg.get("data_root", "./data"),
        label_fraction=cfg["label_fraction"],
        seed=cfg["seed"],
        batch_size=256,
        num_workers=cfg.get("num_workers", 4),
    )

    train_embs_list, test_embs_list = [], []
    train_labels = test_labels = None

    for enc in encoders:
        if isinstance(enc, _DiffusionCacheEncoder):
            # Load from cached full-split embeddings; subset to the stratified indices
            # For simplicity, re-extract via the labeled loader using a DiffusionEncoder instance
            # (only runs if cache is available; otherwise raises a clear error)
            raise RuntimeError(
                "Diffusion encoder requires pre-cached embeddings.\n"
                "Run: uv run python extract_features.py"
            )
        else:
            tr_embs, tr_labels = _extract_from_loader(enc, train_loader, device)
            te_embs, te_labels = _extract_from_loader(enc, test_loader,  device)
            train_embs_list.append(tr_embs)
            test_embs_list.append(te_embs)
            if train_labels is None:
                train_labels = tr_labels
                test_labels  = te_labels

    return train_embs_list, test_embs_list, train_labels, test_labels


def extract_all_embeddings_with_diffusion(encoders: list, enc_cfgs: list, cfg: dict, device: str):
    """
    Version that handles diffusion cache alongside live MAE/DINO encoders.
    enc_cfgs is the list of per-encoder config dicts.
    """
    train_loader, test_loader, _ = get_labeled_loaders(
        root=cfg.get("data_root", "./data"),
        label_fraction=cfg["label_fraction"],
        seed=cfg["seed"],
        batch_size=256,
        num_workers=cfg.get("num_workers", 4),
    )

    train_embs_list, test_embs_list = [], []
    train_labels = test_labels = None

    for enc, ec in zip(encoders, enc_cfgs):
        if isinstance(enc, _DiffusionCacheEncoder):
            # Load stratified subset from cached embeddings
            cache_dir    = ec.get("cache_dir", "results/checkpoints/diffusion_embeddings")
            tr_cache_path = os.path.join(cache_dir, "train.pt")
            te_cache_path = os.path.join(cache_dir, "test.pt")
            if not os.path.exists(tr_cache_path):
                raise RuntimeError(f"Diffusion cache not found: {tr_cache_path}\n"
                                   "Run: uv run python extract_features.py")
            tr_data = torch.load(tr_cache_path)
            te_data = torch.load(te_cache_path)
            # Align to stratified indices used by train_loader
            from data.stl10_data import stratified_subset_indices
            from torchvision import datasets
            import numpy as np
            full_train = datasets.STL10(root=cfg.get("data_root", "./data"),
                                        split="train", download=False)
            if cfg["label_fraction"] < 1.0:
                idx = stratified_subset_indices(
                    full_train.labels, cfg["label_fraction"], cfg["seed"]
                )
                tr_embs   = tr_data["embeddings"][idx]
                tr_labels = tr_data["labels"][idx]
            else:
                tr_embs   = tr_data["embeddings"]
                tr_labels = tr_data["labels"]
            te_embs   = te_data["embeddings"]
            te_labels = te_data["labels"]
        else:
            tr_embs, tr_labels = _extract_from_loader(enc, train_loader, device)
            te_embs, te_labels = _extract_from_loader(enc, test_loader,  device)

        train_embs_list.append(tr_embs.float())
        test_embs_list.append(te_embs.float())
        if train_labels is None:
            train_labels = tr_labels
            test_labels  = te_labels

    return train_embs_list, test_embs_list, train_labels, test_labels


# ──────────────────────────── in-memory dataset ───────────────────────────────

class EmbeddingDataset(torch.utils.data.Dataset):
    """Simple dataset wrapping pre-extracted embedding tensors."""
    def __init__(self, embs: torch.Tensor, labels: torch.Tensor):
        self.embs   = embs
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.embs[idx], self.labels[idx]


# ──────────────────────── LR schedule (matches baseline) ──────────────────────

def linear_warmup_decay(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.0, 1.0 - progress)


# ─────────────────────────────── probe head ──────────────────────────────────

class ProbeHead(nn.Module):
    """
    Trainable part for downstream evaluation.
    For standalone: nn.Linear(embed_dim, n_classes)
    For multi-encoder: EmbeddingFusion → nn.Linear(fused_dim, n_classes)
    """
    def __init__(
        self,
        embed_dims: list[int],
        n_classes:  int   = 10,
        fused_dim:  int   = 256,
        fusion_mode: str  = "concat_proj",
        num_heads:  int   = 4,
    ):
        super().__init__()
        if len(embed_dims) == 1:
            self.fusion = None
            self.head   = nn.Linear(embed_dims[0], n_classes)
        else:
            self.fusion = EmbeddingFusion(
                embed_dims=embed_dims,
                fused_dim=fused_dim,
                mode=fusion_mode,
                num_heads=num_heads,
            )
            self.head = nn.Linear(fused_dim, n_classes)

    def forward(self, embs: list[torch.Tensor]) -> torch.Tensor:
        if self.fusion is not None:
            x = self.fusion(embs)
        else:
            x = embs[0]
        return self.head(x)


# ──────────────────────────── main probe function ─────────────────────────────

@torch.no_grad()
def evaluate_probe(head: ProbeHead, loader, device: str, criterion):
    head.eval()
    all_preds, all_labels = [], []
    total_loss, total     = 0.0, 0
    for emb_batch, labels in loader:
        if isinstance(emb_batch, torch.Tensor):
            emb_batch = [emb_batch]
        emb_batch = [e.to(device) for e in emb_batch]
        labels    = labels.to(device)
        logits    = head(emb_batch)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        total      += labels.size(0)
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(labels.cpu())
    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return float((preds == labels).mean()), total_loss / total, preds, labels


def train_probe(
    train_embs_list:  list,
    test_embs_list:   list,
    train_labels:     torch.Tensor,
    test_labels:      torch.Tensor,
    cfg:              dict,
    out_dir:          str,
    device:           str,
) -> dict:
    """
    Train the linear probe head (+ optional fusion module) on pre-extracted embeddings.

    Returns a metrics dict with smoothed_acc (primary) and full history.
    """
    os.makedirs(out_dir, exist_ok=True)

    embed_dims  = [e.shape[-1] for e in train_embs_list]
    n_classes   = cfg.get("n_classes", 10)
    fused_dim   = cfg.get("fused_dim", 256)
    fusion_mode = cfg.get("fusion_mode", "concat_proj")
    epochs      = cfg.get("epochs", 100)
    batch_size  = cfg.get("batch_size", 256)
    base_lr     = cfg.get("base_lr", 1e-3)
    weight_decay = cfg.get("weight_decay", 0.0)
    warmup_frac = cfg.get("warmup_fraction", 0.1)
    window      = cfg.get("smoothing_window", 10)

    head = ProbeHead(embed_dims, n_classes, fused_dim, fusion_mode).to(device)
    optimizer = Adam(head.parameters(), lr=base_lr, weight_decay=weight_decay)

    # Build in-memory loaders
    if len(train_embs_list) == 1:
        # Standalone: single tensor per sample
        tr_ds = EmbeddingDataset(train_embs_list[0], train_labels)
        te_ds = EmbeddingDataset(test_embs_list[0],  test_labels)
    else:
        # Multi-encoder: concatenate for storage, split later in the head
        # Store as concatenated; head will receive a list (handled by MultiEmbeddingDataset)
        class MultiEmbeddingDataset(torch.utils.data.Dataset):
            def __init__(self, embs_list, labels):
                self.embs_list = embs_list
                self.labels    = labels
            def __len__(self):
                return len(self.labels)
            def __getitem__(self, idx):
                return [e[idx] for e in self.embs_list], self.labels[idx]

        def multi_collate(batch):
            n_enc    = len(batch[0][0])
            embs_out = [torch.stack([b[0][i] for b in batch]) for i in range(n_enc)]
            labels   = torch.tensor([b[1] for b in batch])
            return embs_out, labels

        tr_ds   = MultiEmbeddingDataset(train_embs_list, train_labels)
        te_ds   = MultiEmbeddingDataset(test_embs_list,  test_labels)

    actual_bs  = min(batch_size, len(tr_ds))
    tr_loader  = torch.utils.data.DataLoader(
        tr_ds, batch_size=actual_bs, shuffle=True,
        collate_fn=multi_collate if len(train_embs_list) > 1 else None,
    )
    te_loader  = torch.utils.data.DataLoader(
        te_ds, batch_size=256, shuffle=False,
        collate_fn=multi_collate if len(train_embs_list) > 1 else None,
    )

    total_steps  = len(tr_loader) * epochs
    warmup_steps = max(1, int(total_steps * warmup_frac))
    scheduler    = LambdaLR(optimizer,
                            lr_lambda=lambda s: linear_warmup_decay(s, warmup_steps, total_steps))
    criterion    = nn.CrossEntropyLoss()
    history      = {"train_loss": [], "test_loss": [], "test_acc": []}

    for epoch in range(epochs):
        head.train()
        ep_loss, n_b = 0.0, 0
        for emb_batch, labels in tr_loader:
            if isinstance(emb_batch, torch.Tensor):
                emb_batch = [emb_batch]
            emb_batch = [e.to(device) for e in emb_batch]
            labels    = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(head(emb_batch), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            ep_loss += loss.item(); n_b += 1

        test_acc, test_loss, _, _ = evaluate_probe(head, te_loader, device, criterion)
        history["train_loss"].append(ep_loss / n_b)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

    smoothed_acc = float(np.mean(history["test_acc"][-window:]))
    _, _, preds, labels_np = evaluate_probe(head, te_loader, device, criterion)

    # Save checkpoint and metrics
    torch.save(head.state_dict(), os.path.join(out_dir, "checkpoint.pt"))
    metrics = {
        "label_fraction":   cfg["label_fraction"],
        "seed":             cfg["seed"],
        "smoothed_acc":     smoothed_acc,
        "smoothing_window": window,
        "history":          history,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
