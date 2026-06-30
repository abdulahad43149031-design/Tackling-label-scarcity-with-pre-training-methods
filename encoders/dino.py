"""
DINO encoder and pretraining components.
Reference: Caron et al., "Emerging Properties in Self-Supervised Vision Transformers" (ICCV 2021)

STL-10 specifics (96×96):
  - Student and teacher: identical ViT-Small/16 backbones
  - Multi-crop: 2 global crops (scale 0.4–1.0) + n_local crops (scale 0.05–0.4),
    all resized to 96×96 so ViT's fixed positional embeddings remain valid
  - Teacher: EMA of student weights; teacher is discarded after pretraining
  - After pretraining: only student encoder kept → encode(x) → CLS token (B, d_model)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from models.vit import PatchEmbedding, TransformerBlock, VARIANTS
from encoders.base import BaseEncoder


# ──────────────────────────────── encoder ────────────────────────────────────

class DINOEncoder(BaseEncoder):
    """
    ViT-Small/16 used as DINO student (and, by copy, teacher).
    encode(x) → CLS token embedding (B, d_model).
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
        self._heads  = cfg["num_heads"]
        self._mlp    = cfg["mlp_dim"]
        self._depth  = cfg["depth"]

        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, self.d_model)
        n_patches        = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, self.d_model))
        self.pos_drop  = nn.Dropout(drop)

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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, d_model) CLS token embedding."""
        B   = x.shape[0]
        x   = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)[:, 0]


# ──────────────────────────────── DINO head ──────────────────────────────────

class DINOHead(nn.Module):
    """
    Projection MLP + weight-normed prototype layer (student and teacher share this arch).
    in_dim → hidden_dim (×n_layers-1) → bottleneck_dim → L2-norm → out_dim (prototypes)
    """

    def __init__(
        self,
        in_dim:          int  = 512,
        out_dim:         int  = 4096,
        hidden_dim:      int  = 2048,
        bottleneck_dim:  int  = 256,
        n_layers:        int  = 3,
        norm_last_layer: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden_dim), nn.GELU()]
            d = hidden_dim
        layers += [nn.Linear(d, bottleneck_dim)]
        self.mlp = nn.Sequential(*layers)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


# ───────────────────────────────── DINO loss ─────────────────────────────────

class DINOLoss(nn.Module):
    """
    Self-distillation loss with centering (prevents collapse without batch norm).

    Teacher logits are centred by a running mean and sharpened.
    Student logits use a fixed (higher) temperature.
    Loss = cross-entropy between teacher distribution and student log-probs,
           averaged over all (teacher_view, student_view) pairs excluding same-index global pairs.
    """

    def __init__(
        self,
        out_dim:                    int   = 4096,
        n_crops:                    int   = 8,    # 2 global + n_local
        warmup_teacher_temp:        float = 0.04,
        teacher_temp:               float = 0.07,
        warmup_teacher_temp_epochs: int   = 30,
        n_epochs:                   int   = 100,
        student_temp:               float = 0.1,
        center_momentum:            float = 0.9,
    ):
        super().__init__()
        self.student_temp    = student_temp
        self.center_momentum = center_momentum
        self.n_crops         = n_crops
        self.register_buffer("center", torch.zeros(1, out_dim))

        self.teacher_temp_schedule = torch.cat([
            torch.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            torch.full((max(1, n_epochs - warmup_teacher_temp_epochs),), teacher_temp),
        ])

    def forward(
        self,
        student_out: torch.Tensor,   # (B * n_crops, out_dim) — all student crops
        teacher_out: torch.Tensor,   # (B * 2, out_dim)       — 2 global crops only
        epoch:       int,
    ) -> torch.Tensor:
        t_temp    = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)].item()
        s_chunks  = student_out.chunk(self.n_crops)           # n_crops × (B, out_dim)
        t_soft    = F.softmax((teacher_out - self.center) / t_temp, dim=-1).detach()
        t_chunks  = t_soft.chunk(2)                           # 2 global crops

        loss, n = 0.0, 0
        for ti, t in enumerate(t_chunks):
            for si, s in enumerate(s_chunks):
                if ti == si:
                    continue                                   # skip same-view pairs
                loss += torch.sum(
                    -t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1
                ).mean()
                n += 1
        loss /= n
        self._update_center(teacher_out)
        return loss

    @torch.no_grad()
    def _update_center(self, teacher_out: torch.Tensor):
        batch_center = teacher_out.mean(dim=0, keepdim=True)
        self.center  = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


# ──────────────────────────── multi-crop augmentation ────────────────────────

def _odd_ks(size: int, div: int = 10) -> int:
    """Compute an odd kernel size suitable for GaussianBlur."""
    ks = max(3, size // div)
    return ks if ks % 2 == 1 else ks + 1


def get_multicrop_transform(img_size: int = 96, n_local_crops: int = 6):
    """
    DINO multi-crop augmentation.
    Both global and local crops are resized to img_size so ViT's fixed pos embeddings are valid.

    Global crops (×2): RandomResizedCrop(img_size, scale=(0.4, 1.0)) + strong augmentation
    Local crops  (×n): RandomResizedCrop(img_size, scale=(0.05, 0.4)) + moderate augmentation
    """
    normalize = transforms.Normalize(
        mean=(0.4467, 0.4398, 0.4066),
        std=(0.2603, 0.2566, 0.2713),
    )
    color_jitter = transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
    )
    ks = _odd_ks(img_size)

    global_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            img_size, scale=(0.4, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=ks, sigma=(0.1, 2.0))], p=1.0
        ),
        transforms.ToTensor(),
        normalize,
    ])

    local_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            img_size, scale=(0.05, 0.4),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=ks, sigma=(0.1, 2.0))], p=0.5
        ),
        transforms.ToTensor(),
        normalize,
    ])

    class MultiCropTransform:
        def __init__(self, g_t, l_t, n_loc):
            self.g_t   = g_t
            self.l_t   = l_t
            self.n_loc = n_loc
        def __call__(self, img):
            return (
                [self.g_t(img), self.g_t(img)]
                + [self.l_t(img) for _ in range(self.n_loc)]
            )

    return MultiCropTransform(global_transform, local_transform, n_local_crops)


# ──────────────────────────────── EMA update ─────────────────────────────────

@torch.no_grad()
def update_teacher(student: nn.Module, teacher: nn.Module, momentum: float):
    """Exponential Moving Average: teacher ← momentum·teacher + (1−momentum)·student."""
    for p_s, p_t in zip(student.parameters(), teacher.parameters()):
        p_t.data.mul_(momentum).add_((1.0 - momentum) * p_s.data)


def cosine_scheduler(base: float, end: float, n_epochs: int, n_steps_per_epoch: int) -> torch.Tensor:
    """Returns a 1-D tensor of length n_epochs * n_steps_per_epoch with cosine schedule."""
    total = n_epochs * n_steps_per_epoch
    t     = torch.arange(total, dtype=torch.float64)
    return end + 0.5 * (base - end) * (1 + torch.cos(torch.pi * t / total))
