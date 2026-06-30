"""
MAE encoder and pretraining components.
Reference: He et al., "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022)

STL-10 specifics (96×96, patch_size=16 → 36 patches):
  - mask_ratio=0.75 → 9 visible patches, 27 masked per image during pretraining
  - Encoder: processes only visible patches (efficient) → CLS embedding (B, d_model)
  - Decoder: 4-layer lightweight transformer (dim=256); discarded after pretraining
  - Loss: per-patch MSE on masked tokens (with optional pixel normalisation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vit import PatchEmbedding, TransformerBlock, VARIANTS
from encoders.base import BaseEncoder


# ─────────────────────────────── masking utils ───────────────────────────────

def random_masking(x: torch.Tensor, mask_ratio: float = 0.75):
    """
    Random patch masking via uniform noise sorting.

    Args:
        x         : (B, N, D) patch embeddings (no CLS)
        mask_ratio: fraction to mask (default 0.75)
    Returns:
        x_vis     : (B, N_keep, D) visible patches
        mask      : (B, N) bool — True = masked (loss computed here)
        ids_restore: (B, N) inverse permutation to restore original order in decoder
    """
    B, N, D = x.shape
    N_keep  = max(1, int(N * (1.0 - mask_ratio)))

    noise       = torch.rand(B, N, device=x.device)
    ids_shuffle = noise.argsort(dim=1)           # ascending → first N_keep are "visible"
    ids_restore = ids_shuffle.argsort(dim=1)     # inverse

    ids_keep = ids_shuffle[:, :N_keep]
    x_vis    = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

    mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
    mask.scatter_(1, ids_keep, False)            # False = visible

    return x_vis, mask, ids_restore


# ───────────────────────────────── encoder ───────────────────────────────────

class MAEEncoder(BaseEncoder):
    """
    ViT-Small/16 encoder for MAE.

    Pretraining : forward_features(x, mask_ratio) — processes only visible patches.
    Inference   : encode(x) — processes all patches, returns CLS token (B, d_model).
    """

    def __init__(
        self,
        variant:    str   = "ViT-Small/16",
        img_size:   int   = 96,
        patch_size: int   = 16,
        drop:       float = 0.0,
    ):
        super().__init__()
        cfg          = VARIANTS[variant]
        self.d_model = cfg["d_model"]
        self._depth  = cfg["depth"]
        self._heads  = cfg["num_heads"]
        self._mlp    = cfg["mlp_dim"]

        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, self.d_model)
        self.n_patches   = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        # Full positional embedding (CLS + all patches); visible subset indexed from it
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, self.d_model))

        self.blocks = nn.ModuleList([
            TransformerBlock(self.d_model, self._heads, self._mlp, drop)
            for _ in range(self._depth)
        ])
        self.norm = nn.LayerNorm(self.d_model)
        self._init_weights()

    @property
    def embed_dim(self) -> int:
        return self.d_model

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

    def forward_features(self, x: torch.Tensor, mask_ratio: float = 0.75):
        """
        MAE pretraining forward — only processes visible patches.

        Returns:
            encoded    : (B, 1+N_keep, d_model)
            mask       : (B, N) bool — True = masked
            ids_restore: (B, N)
        """
        x   = self.patch_embed(x)                           # (B, N, D)
        x   = x + self.pos_embed[:, 1:, :]                  # patch positional embeddings
        x, mask, ids_restore = random_masking(x, mask_ratio)

        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x   = torch.cat([cls, x], dim=1)                    # (B, 1+N_keep, D)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x), mask, ids_restore

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward on all patches — used after pretraining (frozen encoder).
        Returns CLS token embedding (B, d_model).
        """
        B   = x.shape[0]
        x   = self.patch_embed(x)
        x   = x + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)[:, 0]                           # CLS → (B, d_model)


# ───────────────────────────────── decoder ───────────────────────────────────

class MAEDecoder(nn.Module):
    """
    Lightweight MAE decoder (discarded after pretraining).
    Projects encoder tokens to decoder_dim, fills in mask tokens, reconstructs patches.
    """

    def __init__(
        self,
        encoder_dim:   int = 512,
        decoder_dim:   int = 256,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
        num_patches:   int = 36,
        patch_size:    int = 16,
        in_channels:   int = 3,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.out_dim     = patch_size * patch_size * in_channels

        self.proj       = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.pos_embed  = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, decoder_heads, decoder_dim * 4, drop=0.0)
            for _ in range(decoder_depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, self.out_dim, bias=True)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x_enc: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_enc      : (B, 1+N_keep, encoder_dim) — CLS + visible tokens from encoder
            ids_restore: (B, N) — permutation to restore original patch order
        Returns:
            pred       : (B, N, patch_size^2 * in_channels) — predicted patch pixels
        """
        B     = x_enc.shape[0]
        x     = self.proj(x_enc)                   # (B, 1+N_keep, decoder_dim)
        cls   = x[:, :1, :]
        vis   = x[:, 1:, :]                        # (B, N_keep, D)

        N_keep = vis.shape[1]
        N      = self.num_patches
        mask_tokens = self.mask_token.expand(B, N - N_keep, -1)
        all_tokens  = torch.cat([vis, mask_tokens], dim=1)  # (B, N, D)

        # Unshuffle → restore original patch order
        all_tokens = torch.gather(
            all_tokens, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, all_tokens.shape[-1]),
        )

        x = torch.cat([cls, all_tokens], dim=1)   # (B, N+1, D)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.pred(x[:, 1:, :])              # skip CLS → (B, N, out_dim)


# ─────────────────────────────── loss / utils ────────────────────────────────

def patchify(imgs: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """(B, C, H, W) → (B, N, patch_size^2 * C) pixel targets."""
    B, C, H, W = imgs.shape
    ph = pw = patch_size
    h, w = H // ph, W // pw
    x = imgs.reshape(B, C, h, ph, w, pw)
    x = x.permute(0, 2, 4, 1, 3, 5)              # (B, h, w, C, ph, pw)
    return x.reshape(B, h * w, C * ph * pw)


def mae_loss(
    pred:       torch.Tensor,
    imgs:       torch.Tensor,
    mask:       torch.Tensor,
    patch_size: int  = 16,
    norm_pix:   bool = True,
) -> torch.Tensor:
    """
    MSE reconstruction loss on masked patches only.

    Args:
        pred      : (B, N, patch_size^2*C) decoder predictions
        imgs      : (B, C, H, W) original images
        mask      : (B, N) bool — True = masked (loss computed here)
        norm_pix  : normalise each target patch to zero mean / unit std (MAE paper default)
    """
    target = patchify(imgs, patch_size)
    if norm_pix:
        mean   = target.mean(dim=-1, keepdim=True)
        var    = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()

    per_patch = ((pred - target) ** 2).mean(dim=-1)    # (B, N)
    return (per_patch * mask.float()).sum() / mask.float().sum()
