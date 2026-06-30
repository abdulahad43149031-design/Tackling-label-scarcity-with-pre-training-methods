"""
Diffusion feature extractor — no training required.
Reference: Baranchuk et al., "Label-Efficient Semantic Segmentation with Diffusion Models" (ICLR 2022)
           Kwon et al., "Diffusion Models Already Have a Semantic Latent Space" (ICLR 2023)

Strategy:
  1. Load a pretrained DDPM U-Net (HuggingFace diffusers).
  2. Resize STL-10 images to the model's expected resolution.
  3. Add Gaussian noise at a fixed timestep t (forward diffusion — no sampling needed).
  4. Hook an intermediate U-Net block; spatially average-pool its activation → embedding.
  5. embed_dim is determined dynamically from the actual hooked activation shape.

Default model: "google/ddpm-cifar10-32" (32×32)
  - Shares 10 overlapping classes with STL-10
  - Fast: 32×32 forward pass even at batch_size=256
  - mid_block activation: (B, 256, 4, 4) → pool → (B, 256)

Alternative: "google/ddpm-celebahq-256" (256×256) for richer features at the cost of speed.

Timestep and hook_layer are swept in extract_features.py before being locked in.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from encoders.base import BaseEncoder


class DiffusionEncoder(BaseEncoder):
    """
    Pretrained DDPM U-Net used as a frozen feature extractor.

    Args:
        model_id   : HuggingFace model ID (e.g. "google/ddpm-cifar10-32")
        timestep   : noise level to inject (int ∈ [0, T)); more noise = coarser features
        hook_layer : attribute path on the UNet model to hook
                     e.g. "mid_block", "down_blocks.2", "up_blocks.0"
        model_img_size: resolution the UNet expects (images are resized to this)
        device     : "cuda" or "cpu"
    """

    def __init__(
        self,
        model_id:       str = "google/ddpm-cifar10-32",
        timestep:       int = 250,
        hook_layer:     str = "mid_block",
        model_img_size: int = 32,
        device:         str = "cpu",
    ):
        super().__init__()
        self.timestep       = timestep
        self.hook_layer     = hook_layer
        self.model_img_size = model_img_size
        self._device        = device
        self._embed_dim     = None      # set after calibration
        self._activation    = {}

        # Load pretrained U-Net and scheduler
        from diffusers import UNet2DModel, DDPMScheduler
        self.unet      = UNet2DModel.from_pretrained(model_id).to(device)
        self.scheduler = DDPMScheduler.from_pretrained(model_id)
        self.unet.eval()
        for p in self.unet.parameters():
            p.requires_grad = False

        # Register the forward hook
        self._handle = self._register_hook(hook_layer)

        # Resize transform: STL-10 images → model resolution
        self.resize = transforms.Resize(
            (model_img_size, model_img_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )

        # Calibrate embed_dim with a dummy forward pass
        self._calibrate()

    # ─────────────────────── hook registration ───────────────────────────────

    def _get_submodule(self, path: str) -> nn.Module:
        """Traverse dotted attribute path on self.unet (e.g. 'down_blocks.2')."""
        obj = self.unet
        for part in path.split("."):
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        return obj

    def _register_hook(self, hook_layer: str):
        act = self._activation

        def hook_fn(module, input, output):
            # output may be a tuple (some diffusers blocks return (hidden, res_samples))
            feat = output[0] if isinstance(output, tuple) else output
            act["feat"] = feat.detach()

        module = self._get_submodule(hook_layer)
        return module.register_forward_hook(hook_fn)

    # ─────────────────────── calibration ─────────────────────────────────────

    @torch.no_grad()
    def _calibrate(self):
        """Run a single dummy forward pass to record embed_dim."""
        dummy = torch.zeros(1, 3, self.model_img_size, self.model_img_size,
                            device=self._device)
        ts    = torch.tensor([self.timestep], device=self._device)
        self.unet(dummy, ts)
        feat  = self._activation["feat"]                  # (1, C, H, W)
        self._embed_dim = feat.shape[1]                   # channel dim after spatial pool
        print(f"[DiffusionEncoder] hook='{self.hook_layer}' | "
              f"feat shape: {tuple(feat.shape)} | embed_dim: {self._embed_dim}")

    # ─────────────────────── BaseEncoder interface ────────────────────────────

    @property
    def embed_dim(self) -> int:
        if self._embed_dim is None:
            raise RuntimeError("DiffusionEncoder not calibrated — call _calibrate() first.")
        return self._embed_dim

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract diffusion features for a batch of STL-10 images.

        Args:
            x: (B, C, H, W) normalised images in STL-10 colour space
        Returns:
            (B, embed_dim) embedding
        """
        # Undo STL-10 normalisation → [0, 1] → [−1, 1] for diffusion model
        mean = torch.tensor([0.4467, 0.4398, 0.4066], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.2603, 0.2566, 0.2713], device=x.device).view(1, 3, 1, 1)
        x_01 = x * std + mean                              # [0, 1]
        x_11 = x_01 * 2.0 - 1.0                           # [−1, 1] (diffusion convention)

        # Resize to model resolution
        x_model = self.resize(x_11)

        # Add noise at fixed timestep (forward diffusion process)
        noise     = torch.randn_like(x_model)
        timesteps = torch.full((x_model.shape[0],), self.timestep,
                               device=x_model.device, dtype=torch.long)
        x_noisy   = self.scheduler.add_noise(x_model, noise, timesteps)

        # Forward through U-Net (hook fires internally)
        self.unet(x_noisy, timesteps)

        # Spatial average pool hooked activation → flat embedding
        feat = self._activation["feat"]                    # (B, C, H', W')
        return feat.mean(dim=[2, 3])                       # (B, embed_dim)

    def freeze(self):
        """No-op — diffusion encoder is always frozen."""
        for p in self.unet.parameters():
            p.requires_grad = False
        self.unet.eval()

    def __del__(self):
        if hasattr(self, "_handle") and self._handle is not None:
            self._handle.remove()
