
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

from .fractal import flatten_by_curve, unflatten_by_curve


def _groups(channels: int, max_groups: int = 8) -> int:
    for value in range(min(max_groups, channels), 0, -1):
        if channels % value == 0:
            return value
    return 1


class AnatomyPriorBranch(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1) -> None:
        super().__init__()

        hidden_channels = max(1, in_channels // 4)
        groups = _groups(hidden_channels)

        self.branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)


class AnatomyGuidedHilbertMamba(nn.Module):
    def __init__(
        self,
        channels: int,
        curve: str = "hilbert",
        use_forward_mamba: bool = True,
        use_backward_mamba: bool = True,
        use_anatomy_gating: bool = True,
        d_state: int = 8,
        d_conv: int = 2,
        expand: int = 2,
    ) -> None:
        super().__init__()

        self.curve = str(curve)
        self.use_forward_mamba = bool(use_forward_mamba)
        self.use_backward_mamba = bool(use_backward_mamba)
        self.use_anatomy_gating = bool(use_anatomy_gating)

        if self.use_backward_mamba and not self.use_forward_mamba:
            raise ValueError("Backward Mamba cannot be enabled without forward Mamba.")

        use_any_mamba = self.use_forward_mamba or self.use_backward_mamba

        if use_any_mamba and Mamba is None:
            raise ImportError("mamba_ssm is not installed. Install it to enable Mamba.")

        if self.use_anatomy_gating:
            self.anatomy_prior_branch = AnatomyPriorBranch(channels, out_channels=1)
            self.alpha = nn.Parameter(torch.tensor(1.0))
        else:
            self.anatomy_prior_branch = None
            self.register_parameter("alpha", None)

        if self.use_forward_mamba:
            self.mamba_f = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.mamba_f = None

        if self.use_backward_mamba:
            self.mamba_b = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.mamba_b = None

        if use_any_mamba:
            self.norm = nn.LayerNorm(channels)
            self.proj = nn.Linear(channels, channels)
        else:
            self.norm = None
            self.proj = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        height, width = x.shape[-2:]

        if self.use_anatomy_gating:
            anatomy_prior = self.anatomy_prior_branch(x)
            gated = x * (1.0 + self.alpha * anatomy_prior)
        else:
            anatomy_prior = None
            gated = x

        if not self.use_forward_mamba:
            return gated, anatomy_prior

        seq, idx = flatten_by_curve(gated, self.curve)
        seq_f = self.mamba_f(seq)

        if self.use_backward_mamba:
            seq_b = torch.flip(seq, dims=[1])
            seq_b = self.mamba_b(seq_b)
            seq_b = torch.flip(seq_b, dims=[1])
            sequence_update = seq_f + seq_b
        else:
            sequence_update = seq_f

        sequence_update = self.proj(self.norm(sequence_update))
        mixed = unflatten_by_curve(sequence_update, idx, height, width)

        return x + mixed, anatomy_prior


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        groups = _groups(out_channels)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HSMv12D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        base_dim: int = 12,
        depth: int = 3,
        variant: str = "binary_aghm",
        curve: str = "hilbert",
        use_forward_mamba: bool = True,
        use_backward_mamba: bool = True,
        use_anatomy_gating: bool = True,
        d_state: int = 8,
        d_conv: int = 2,
        expand: int = 2,
    ) -> None:
        super().__init__()

        if str(variant).lower() != "binary_aghm":
            raise ValueError("This project only supports model.variant: binary_aghm")

        if depth not in {3, 4}:
            raise ValueError("depth must be 3 or 4")

        self.depth = int(depth)
        self.variant = "binary_aghm"
        self.curve = str(curve)
        self.use_forward_mamba = bool(use_forward_mamba)
        self.use_backward_mamba = bool(use_backward_mamba)
        self.use_anatomy_gating = bool(use_anatomy_gating)

        c = int(base_dim)

        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if self.depth == 4:
            self.enc4 = ConvBlock(c * 4, c * 8)
            self.bottleneck = ConvBlock(c * 8, c * 16)
            bottleneck_channels = c * 16

            self.up3 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
            self.dec3 = ConvBlock(c * 16, c * 8)
            decoder_input_channels = c * 8
        else:
            self.enc4 = None
            self.bottleneck = ConvBlock(c * 4, c * 8)
            bottleneck_channels = c * 8

            self.up3 = None
            self.dec3 = None
            decoder_input_channels = c * 8

        self.bottleneck_mixer = AnatomyGuidedHilbertMamba(
            channels=bottleneck_channels,
            curve=self.curve,
            use_forward_mamba=use_forward_mamba,
            use_backward_mamba=use_backward_mamba,
            use_anatomy_gating=use_anatomy_gating,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.up2 = nn.ConvTranspose2d(decoder_input_channels, c * 4, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c * 8, c * 4)

        self.up1 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c * 4, c * 2)

        self.up0 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec0 = ConvBlock(c * 2, c)

        self.head = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        divisor = 16 if self.depth == 4 else 8
        height, width = x.shape[-2:]

        if height % divisor != 0 or width % divisor != 0:
            raise ValueError(f"Input height and width must be divisible by {divisor}.")

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        if self.depth == 4:
            e4 = self.enc4(self.pool(e3))
            feat = self.bottleneck(self.pool(e4))
        else:
            e4 = None
            feat = self.bottleneck(self.pool(e3))

        feat_mixed, anatomy_prior = self.bottleneck_mixer(feat)

        if self.depth == 4:
            x_dec = self.dec3(torch.cat([self.up3(feat_mixed), e4], dim=1))
        else:
            x_dec = feat_mixed

        x_dec = self.dec2(torch.cat([self.up2(x_dec), e3], dim=1))
        x_dec = self.dec1(torch.cat([self.up1(x_dec), e2], dim=1))
        x_dec = self.dec0(torch.cat([self.up0(x_dec), e1], dim=1))

        logits = self.head(x_dec)

        if return_aux:
            return {"logits": logits, "anatomy_prior": anatomy_prior}

        return logits


def build_model(config: dict) -> HSMv12D:
    model_cfg = config.get("model", config) if isinstance(config, dict) else config
    variant = str(model_cfg.get("variant", "binary_aghm")).lower()

    if variant != "binary_aghm":
        raise ValueError("This project only supports model.variant: binary_aghm")

    return HSMv12D(
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=int(model_cfg.get("out_channels", 4)),
        base_dim=int(model_cfg.get("base_dim", 12)),
        depth=int(model_cfg.get("depth", 3)),
        variant="binary_aghm",
        curve=str(model_cfg.get("curve", "hilbert")),
        use_forward_mamba=bool(model_cfg.get("use_forward_mamba", True)),
        use_backward_mamba=bool(model_cfg.get("use_backward_mamba", True)),
        use_anatomy_gating=bool(model_cfg.get("use_anatomy_gating", True)),
        d_state=int(model_cfg.get("d_state", 8)),
        d_conv=int(model_cfg.get("d_conv", 2)),
        expand=int(model_cfg.get("expand", 2)),
    )
