"""
    STL-10 loader with stratified sub-sampling.

    Shared config among all the self-supervised models so the train/test split
    logic is identical across all the models.
"""

import numpy as np
import torch 
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMG_SIZE = 96
NORM_MEAN = (0.4467, 0.4398, 0.4066)
NORM_STD = (0.2603, 0.2566, 0.2713)

CLASS_NAMES = ["airplane", "bird", "car", "cat", "deer",
               "dog", "horse", "monkey", "ship", "truck"]

# Augmentation

train_transform = transforms.Compose([
    transforms.RandomCrop(IMG_SIZE, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD)
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD)
])

def stratified_subset_indices(labels: np.ndarray, fraction: float, seed: int):
    """
    creates a stratified subset of a dataset, meaning it samples the same fraction of examples from every class instead of randomly sampling from the entire dataset.
    """
    rng = np.random.RandomState(seed)
    indices = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_keep = max(1, int(len(cls_idx) * fraction))
        indices.extend(cls_idx[:n_keep])
    rng.shuffle(indices)
    return indices

def get_labeled_loaders(root = "./data", label_fraction = 1.0, seed = 0, batch_size = 64, num_workers = 4):
    """
    prepares the training and testing dataloaders for your experiment. Its main purpose is to let you train on either the full training set or a class-balanced subset while always evaluating on the complete test set.
    """

    train_full = datasets.STL10(root=root, split="train", download = True, transform = train_transform)

    test_set = datasets.STL10(root = root, split="test", download = True, transform = eval_transform)

    if label_fraction < 1.0:
        idx = stratified_subset_indices(train_full.labels, label_fraction, seed)
        train_set = Subset(train_full, idx)
    else:
        train_set = train_full

    actual_batch_size = min(batch_size, len(train_set))

    train_loader = DataLoader(train_set, batch_size = actual_batch_size, shuffle = True, num_workers = num_workers, drop_last = True)

    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, len(train_set)


    


