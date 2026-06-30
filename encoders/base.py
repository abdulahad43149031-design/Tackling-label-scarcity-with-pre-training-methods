"""
Abstract base class for all encoders (MAE, DINO, Diffusion).
Every encoder must implement:
  - encode(x) → embedding tensor of shape (B, embed_dim), frozen after pretraining
  - embed_dim property
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseEncoder(ABC, nn.Module):
    """Shared interface for all frozen encoders used in Phase 2 (linear probe / fusion)."""

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Dimensionality of the output embedding vector."""
        ...

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing a flat embedding.

        Args:
            x: (B, C, H, W) image tensor, normalised.
        Returns:
            (B, embed_dim) embedding tensor.
        """
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def freeze(self):
        """Freeze all parameters. Call after pretraining is complete."""
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
