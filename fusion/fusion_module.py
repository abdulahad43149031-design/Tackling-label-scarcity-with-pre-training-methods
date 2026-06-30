"""
EmbeddingFusion module — combines frozen encoder embeddings for multi-encoder configs.

Two modes:
  concat_proj (default/primary):
    Concatenate all frozen embeddings → Linear → LayerNorm → GELU → fused_dim vector.
    Preferred over naive addition: addition forces same dimensionality + comparable
    embedding spaces, which MAE/DINO/Diffusion don't share (different objectives).

  cross_attn (fallback):
    Multi-head attention across the set of per-encoder embeddings.
    Lets the model weight encoders differently per-image rather than a fixed blend.
    Use if concat_proj shows a plateau or the encoders have very different embedding norms.

Only the fusion module + downstream head train; encoders are always frozen.
"""

import torch
import torch.nn as nn


class EmbeddingFusion(nn.Module):
    """
    Args:
        embed_dims : List of embedding dimensions from each encoder.
                     e.g. [512, 512] for two ViT-Small encoders,
                          [512, 512, C_diffusion] for trio.
        fused_dim  : Output dimensionality of the fused embedding.
        mode       : "concat_proj" (default) or "cross_attn".
        num_heads  : Number of attention heads (only used in cross_attn mode).
    """

    def __init__(
        self,
        embed_dims: list[int],
        fused_dim:  int  = 256,
        mode:       str  = "concat_proj",
        num_heads:  int  = 4,
    ):
        super().__init__()
        self.mode       = mode
        self.embed_dims = embed_dims
        self.fused_dim  = fused_dim

        if mode == "concat_proj":
            total_dim = sum(embed_dims)
            self.proj = nn.Sequential(
                nn.Linear(total_dim, fused_dim),
                nn.LayerNorm(fused_dim),
                nn.GELU(),
            )

        elif mode == "cross_attn":
            # Project all encoders to the same dimensionality first
            self.input_projs = nn.ModuleList([
                nn.Linear(d, fused_dim) for d in embed_dims
            ])
            self.attn = nn.MultiheadAttention(
                embed_dim=fused_dim, num_heads=num_heads, batch_first=True
            )
            self.norm = nn.LayerNorm(fused_dim)

        else:
            raise ValueError(f"Unknown fusion mode '{mode}'. Choose 'concat_proj' or 'cross_attn'.")

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            embeddings: List of tensors, each (B, embed_dim_i) — one per frozen encoder.
        Returns:
            (B, fused_dim) fused embedding.
        """
        if self.mode == "concat_proj":
            x = torch.cat(embeddings, dim=-1)   # (B, sum(embed_dims))
            return self.proj(x)                  # (B, fused_dim)

        else:  # cross_attn
            # Project each embedding to fused_dim, stack as a sequence
            tokens = torch.stack(
                [proj(e) for proj, e in zip(self.input_projs, embeddings)],
                dim=1,                            # (B, n_encoders, fused_dim)
            )
            attn_out, _ = self.attn(tokens, tokens, tokens)
            attn_out = self.norm(attn_out + tokens)
            return attn_out.mean(dim=1)           # (B, fused_dim) — mean-pool over encoders
