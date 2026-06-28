"""
Vision Transformer (ViT) — paper-faithful implementation.
Reference: Dosovitskiy et al., "An Image is Worth 16x16 Words" (ICLR 2021).

Supported variants (matching the paper's Table 1):
  ViT-Small/16  : depth=8,  heads=8,  d_model=512,  mlp_dim=2048
  ViT-Base/16   : depth=12, heads=12, d_model=768,  mlp_dim=3072
  ViT-Large/16  : depth=24, heads=16, d_model=1024, mlp_dim=4096

Designed for STL-10 (96×96 RGB, 10 classes).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ configs --

VARIANTS = {
    "ViT-Small/16": dict(depth=8,  num_heads=8,  d_model=512,  mlp_dim=2048),
    "ViT-Base/16":  dict(depth=12, num_heads=12, d_model=768,  mlp_dim=3072),
    "ViT-Large/16": dict(depth=24, num_heads=16, d_model=1024, mlp_dim=4096),
}


# --------------------------------------------------------------- components --

class PatchEmbedding(nn.Module):
    """Split image into non-overlapping patches and linearly project each one."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int, d_model: int):
        super().__init__()
        assert img_size % patch_size == 0, \
            f"Image size {img_size} must be divisible by patch size {patch_size}."
        self.n_patches = (img_size // patch_size) ** 2
        # A single Conv2d with stride=patch_size does the flatten+project in one step.
        self.proj = nn.Conv2d(in_channels, d_model,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, d_model, H/P, W/P) → (B, N, d_model)
        x = self.proj(x)                          # (B, d_model, h, w)
        x = x.flatten(2)                          # (B, d_model, N)
        x = x.transpose(1, 2)                     # (B, N, d_model)
        return x


class MHSelfAttention(nn.Module):
    """Multi-head self-attention (pre-norm variant used in the ViT paper)."""

    def __init__(self, d_model: int, num_heads: int, drop: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.scale     = math.sqrt(self.head_dim)

        self.qkv  = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(drop)
        self.out_drop  = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)         # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)                   # each: (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) / self.scale   # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)   # (B, N, D)
        return self.out_drop(self.proj(out))


class MLP(nn.Module):
    """Position-wise feed-forward network (GELU, paper §3.1)."""

    def __init__(self, d_model: int, mlp_dim: int, drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """One encoder block: LayerNorm → Attention → residual, LayerNorm → MLP → residual."""

    def __init__(self, d_model: int, num_heads: int, mlp_dim: int, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MHSelfAttention(d_model, num_heads, drop)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = MLP(d_model, mlp_dim, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------------------- main model --

class VisionTransformer(nn.Module):
    """
    Vision Transformer for image classification.

    Args:
        variant    : One of the keys in VARIANTS dict (e.g. "ViT-Base/16").
        img_size   : Input image resolution (default 96 for STL-10).
        patch_size : Patch size in pixels (default 16).
        in_channels: Number of input channels (default 3 for RGB).
        num_classes: Number of output classes (default 10 for STL-10).
        drop       : Dropout probability applied throughout the network.
    """

    def __init__(
        self,
        variant:     str   = "ViT-Base/16",
        img_size:    int   = 96,
        patch_size:  int   = 16,
        in_channels: int   = 3,
        num_classes: int   = 10,
        drop:        float = 0.1,
    ):
        super().__init__()

        if variant not in VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Choose from: {list(VARIANTS.keys())}"
            )
        cfg = VARIANTS[variant]
        d_model  = cfg["d_model"]
        depth    = cfg["depth"]
        num_heads = cfg["num_heads"]
        mlp_dim  = cfg["mlp_dim"]

        # --- Patch embedding + positional encoding ---
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        n_patches = self.patch_embed.n_patches

        # Learnable [CLS] token (paper §3.1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Learnable 1-D positional embeddings for (CLS + patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))

        self.pos_drop = nn.Dropout(drop)

        # --- Transformer encoder ---
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, num_heads, mlp_dim, drop)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(d_model)

        # --- Classification head ---
        # Paper uses a single linear layer after pre-training fine-tune stage;
        # for scratch training a simple linear head is standard.
        self.head = nn.Linear(d_model, num_classes)

        # --- Weight init (paper uses truncated normal, std=0.02) ---
        self._init_weights()

    # ---------------------------------------------------------------------- #

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # 1. Patch embedding: (B, C, H, W) → (B, N, d_model)
        x = self.patch_embed(x)

        # 2. Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)          # (B, N+1, d_model)

        # 3. Add positional embedding + dropout
        x = self.pos_drop(x + self.pos_embed)

        # 4. Transformer encoder
        x = self.blocks(x)
        x = self.norm(x)

        # 5. Classify from CLS token
        return self.head(x[:, 0])
