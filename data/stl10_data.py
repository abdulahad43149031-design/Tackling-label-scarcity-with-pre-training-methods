"""
STL-10 data loading utilities — shared across all configs.

Loaders:
  get_labeled_loaders()   — stratified subset of the 5K labeled train set + full test set.
                            Used by: train_baseline.py, train_downstream.py, eval/linear_probe.py
  get_unlabeled_loader()  — full 100K unlabeled split with a single augmentation transform.
                            Used by: train_mae.py
  get_dino_unlabeled_loader() — 100K unlabeled with multi-crop transform (returns crop lists).
                            Used by: train_dino.py
  get_all_images_loader() — labeled train + unlabeled (no labels) for embedding caching.
                            Used by: extract_features.py, eval/cka.py
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# ─────────────────────────────── dataset constants ───────────────────────────

IMG_SIZE   = 96
NORM_MEAN  = (0.4467, 0.4398, 0.4066)
NORM_STD   = (0.2603, 0.2566, 0.2713)
NUM_CLASSES = 10

CLASS_NAMES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]

# ──────────────────────────── standard transforms ────────────────────────────

train_transform = transforms.Compose([
    transforms.RandomCrop(IMG_SIZE, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

# MAE pretraining augmentation (stronger crop, no color jitter — MAE paper §A.2)
mae_pretrain_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.2, 1.0),
                                 interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

# ─────────────────────── dataset wrappers (module-level for pickling) ────────

class _UnlabeledDataset(Dataset):
    """Wraps STL-10 unlabeled split, returning images only (no labels)."""
    def __init__(self, root, transform):
        self.ds = datasets.STL10(root=root, split="unlabeled",
                                 download=True, transform=transform)
    def __getitem__(self, idx):
        img, _ = self.ds[idx]
        return img
    def __len__(self):
        return len(self.ds)


class _DINODataset(Dataset):
    """Wraps STL-10 unlabeled split with a multi-crop transform for DINO."""
    def __init__(self, root, transform):
        self.ds        = datasets.STL10(root=root, split="unlabeled",
                                         download=True)
        self.transform = transform
    def __getitem__(self, idx):
        img, _ = self.ds[idx]
        return self.transform(img)     # list of crop tensors
    def __len__(self):
        return len(self.ds)


class _NoLabelDataset(Dataset):
    """Wraps STL-10 unlabeled split, returning (image, -1) pairs."""
    def __init__(self, root, transform):
        self.ds = datasets.STL10(root=root, split="unlabeled",
                                 download=True, transform=transform)
    def __getitem__(self, idx):
        img, _ = self.ds[idx]
        return img, -1
    def __len__(self):
        return len(self.ds)


# ────────────────────────────── stratified split ─────────────────────────────

def stratified_subset_indices(labels: np.ndarray, fraction: float, seed: int) -> list:
    """
    Stratified subset of a dataset: samples the same fraction from every class.
    Ensures class balance is preserved even at very low label fractions.
    """
    rng     = np.random.RandomState(seed)
    indices = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_keep  = max(1, int(len(cls_idx) * fraction))
        indices.extend(cls_idx[:n_keep])
    rng.shuffle(indices)
    return indices

# ──────────────────────────────── public API ─────────────────────────────────

def get_labeled_loaders(
    root:           str   = "./data",
    label_fraction: float = 1.0,
    seed:           int   = 0,
    batch_size:     int   = 64,
    num_workers:    int   = 4,
):
    """
    Labeled train loader (stratified subset) + full test loader.
    The batch_size is silently capped at len(train_set) for tiny fractions.

    Returns: (train_loader, test_loader, n_train)
    """
    train_full = datasets.STL10(root=root, split="train",
                                download=True, transform=train_transform)
    test_set   = datasets.STL10(root=root, split="test",
                                download=True, transform=eval_transform)

    if label_fraction < 1.0:
        idx       = stratified_subset_indices(train_full.labels, label_fraction, seed)
        train_set = Subset(train_full, idx)
    else:
        train_set = train_full

    actual_bs = min(batch_size, len(train_set))

    train_loader = DataLoader(train_set, batch_size=actual_bs, shuffle=True,
                              num_workers=num_workers, drop_last=True,
                              pin_memory=True, persistent_workers=num_workers > 0)
    test_loader  = DataLoader(test_set,  batch_size=128, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)

    return train_loader, test_loader, len(train_set)


def get_unlabeled_loader(
    root:        str = "./data",
    batch_size:  int = 128,
    num_workers: int = 4,
):
    """
    Full 100K unlabeled STL-10 split with MAE pretraining augmentation.
    Used by train_mae.py.

    Returns: DataLoader (images only, no labels)
    """
    ds = _UnlabeledDataset(root, mae_pretrain_transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, drop_last=True,
                      pin_memory=True, persistent_workers=num_workers > 0)


def _dino_collate(batch):
    """Collate multi-crop samples: stack per crop index across the batch."""
    n_crops = len(batch[0])
    return [torch.stack([sample[i] for sample in batch]) for i in range(n_crops)]


def get_dino_unlabeled_loader(
    multicrop_transform,
    root:        str = "./data",
    batch_size:  int = 64,
    num_workers: int = 4,
):
    """
    100K unlabeled STL-10 split with DINO multi-crop augmentation.
    Each sample is a list of N_crops tensors; collation stacks them per crop.

    Args:
        multicrop_transform: callable returning a list of crop tensors per image
    Returns: DataLoader yielding list[Tensor(B,C,H,W)] of length n_crops
    """
    ds = _DINODataset(root, multicrop_transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, drop_last=True,
                      pin_memory=True, collate_fn=_dino_collate,
                      persistent_workers=num_workers > 0)


def get_all_images_loader(
    root:          str   = "./data",
    split:         str   = "train",
    label_fraction: float = 1.0,
    seed:          int   = 0,
    batch_size:    int   = 128,
    num_workers:   int   = 4,
):
    """
    Loader for embedding pre-extraction. Uses eval_transform (no augmentation).
    Returns: (DataLoader, labels_array) — labels_array is None for unlabeled split.
    """
    if split == "unlabeled":
        ds = _NoLabelDataset(root, eval_transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=num_workers > 0)
        return loader, None

    full = datasets.STL10(root=root, split=split, download=True, transform=eval_transform)
    if label_fraction < 1.0:
        idx  = stratified_subset_indices(full.labels, label_fraction, seed)
        sub  = Subset(full, idx)
        labels = full.labels[idx]
    else:
        sub    = full
        labels = full.labels

    loader = DataLoader(sub, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=num_workers > 0)
    return loader, labels
