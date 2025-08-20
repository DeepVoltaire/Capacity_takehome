"""
Siamese U-Net change segmentation using **Segmentation Models PyTorch (SMP)** with a
**pretrained timm-efficientnet-b1** encoder.

Includes two models:
  1) `SiameseUNetSMPShared` — single-modality, shared encoder across time (t0/t1),
     per-scale time fusion, UNet decoder.
  2) `SiameseUNetSMPDualStream` — dual-stream (Optical + SAR) encoders shared over time,
     time fusion within each modality then cross-modal fusion, UNet decoder.

Key points:
- Uses `smp.encoders.get_encoder` with `encoder_name="timm-efficientnet-b1"` (hyphen spelling).
- Accepts arbitrary input channels (e.g., 4 for S2, 2 for SAR). SMP re-inits the first conv weight
  when `in_channels != 3`, so ImageNet weights still help.
- Per-scale fusion choices: "concat", "diff", "absdiff" (default), or "concat_diff".
- Version-friendly construction of `UnetDecoder` supporting SMP ≥0.5 (use_norm) and older (use_batchnorm).

You can train end-to-end (both encoder & decoder) or freeze the encoder for a few epochs then unfreeze.
"""
from __future__ import annotations
from typing import List, Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationHead

try:
    # SMP ≥ 0.5.x
    from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder  # type: ignore
    _SMP_NEW = True
except Exception:  # pragma: no cover
    # legacy path
    from segmentation_models_pytorch.unet.decoder import UnetDecoder  # type: ignore
    _SMP_NEW = False


# ------------------------
# Helpers & Fusion blocks
# ------------------------

def _normalize_encoder_name(name: str) -> str:
    """SMP uses hyphens for timm names. Accept underscore alias(es) and normalize."""
    return name.replace("_", "-")


class _TimeFusion(nn.Module):
    """Fuse features from t0 and t1 at a given scale.

    mode ∈ {"concat", "diff", "absdiff", "concat_diff"}
    Output channels == in_ch (keeps decoder channel math intact).
    """
    def __init__(self, in_ch: int, mode: str = "absdiff"):
        super().__init__()
        self.mode = mode
        if mode == "concat":
            self.proj = nn.Conv2d(in_ch * 2, in_ch, kernel_size=1, bias=False)
        elif mode == "concat_diff":
            self.proj = nn.Conv2d(in_ch * 3, in_ch, kernel_size=1, bias=False)
        elif mode in {"diff", "absdiff"}:
            self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False)
        else:
            raise ValueError(f"Unsupported fusion mode: {mode}")
        self.bn = nn.BatchNorm2d(in_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
        if self.mode == "concat":
            x = torch.cat([f0, f1], dim=1)
        elif self.mode == "diff":
            x = f1 - f0
        elif self.mode == "absdiff":
            x = torch.abs(f1 - f0)
        else:  # concat_diff
            x = torch.cat([f0, f1, torch.abs(f1 - f0)], dim=1)
        x = self.proj(x)
        return self.act(self.bn(x))


class _ModalFusion(nn.Module):
    """Fuse two modality features (e.g., optical & SAR) at same scale.

    mode ∈ {"concat", "se"}
      - concat: [opt, sar] → 1x1 → C
      - se: squeeze-excitation gates per modality, then weighted sum
    Assumes both inputs have C channels.
    """
    def __init__(self, in_ch: int, mode: str = "concat"):
        super().__init__()
        self.mode = mode
        if mode == "concat":
            self.proj = nn.Conv2d(in_ch * 2, in_ch, kernel_size=1, bias=False)
            self.bn = nn.BatchNorm2d(in_ch)
            self.act = nn.ReLU(inplace=True)
        elif mode == "se":
            r = max(8, in_ch // 8)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_ch * 2, r, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(r, 2, 1),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Unsupported modal fusion: {mode}")

    def forward(self, f_opt: torch.Tensor, f_sar: torch.Tensor) -> torch.Tensor:
        if self.mode == "concat":
            x = self.proj(torch.cat([f_opt, f_sar], dim=1))
            return self.act(self.bn(x))
        w = self.se(torch.cat([f_opt, f_sar], dim=1))
        return w[:, :1] * f_opt + w[:, 1:] * f_sar


# --------------------------------------------------
# 1) Siamese U-Net (shared weights) — single modality
# --------------------------------------------------

class SiameseUNetSMPShared(nn.Module):
    """Siamese UNet using an SMP pretrained encoder + SMP Unet decoder.

    Args:
        in_channels: input channels (e.g., 4 for S2 RGB+NIR)
        classes: output channels (1 for binary change)
        encoder_name: e.g. "timm_efficientnet_b1" or "timm-efficientnet-b1"
        encoder_weights: usually "imagenet" (works with non-3ch via weight remapping)
        encoder_depth: number of downsampling blocks to use (3..5)
        decoder_channels: channel plan of Unet decoder blocks (len == encoder_depth)
        time_fusion_mode: per-scale fusion of t0/t1 features
    """
    def __init__(
        self,
        in_channels: int,
        classes: int = 1,
        encoder_name: str = "timm_efficientnet_b1",
        encoder_weights: Optional[str] = "imagenet",
        encoder_depth: int = 5,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        time_fusion_mode: str = "absdiff",
        decoder_interpolation: str = "nearest",  # SMP ≥0.5
    ):
        super().__init__()
        enc_name = _normalize_encoder_name(encoder_name)

        # Shared encoder across time
        self.encoder = smp.encoders.get_encoder(
            enc_name, in_channels=in_channels, depth=encoder_depth, weights=encoder_weights
        )
        self._enc_channels = list(self.encoder.out_channels)  # [in, c1, c2, c3, c4, c5]

        # Per-stage time fusion blocks (same channel dims as encoder features)
        self.time_fuse = nn.ModuleList([_TimeFusion(c, time_fusion_mode) for c in self._enc_channels])

        # UNet decoder built directly from SMP
        self.decoder = UnetDecoder(
            encoder_channels=self._enc_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_norm="batchnorm",
            add_center_block=enc_name.startswith("vgg"),
            attention_type=None,
            interpolation_mode=decoder_interpolation,
        )

        self.seg_head = SegmentationHead(
            in_channels=decoder_channels[-1], out_channels=classes, activation=None, kernel_size=3
        )

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        # Encoder features at each stage
        f0: List[torch.Tensor] = self.encoder(x0)  # len = encoder_depth+1
        f1: List[torch.Tensor] = self.encoder(x1)

        # Per-stage time fusion (keeps original channel dims)
        fused: List[torch.Tensor] = [self.time_fuse[i](f0[i], f1[i]) for i in range(len(f0))]

        # SMP UnetDecoder expects *features (varargs)
        dec = self.decoder(fused)
        return self.seg_head(dec)


# -----------------------------------------------------------------
# 2) Dual-stream Siamese U-Net (SAR + Optical) — shared per modality
# -----------------------------------------------------------------

class SiameseUNetSMPDualStream(nn.Module):
    """Dual-stream (Optical+SAR) Siamese UNet on top of SMP pretrained encoders.

    Pipeline:
      - Optical encoder shared over time, SAR encoder shared over time.
      - Time fusion within each modality per scale.
      - Cross-modal fusion per scale → single fused pyramid.
      - SMP Unet decoder on the fused pyramid.

    Inputs:
      x0_opt, x1_opt: optical t0/t1 (B, C_opt, H, W)
      x0_sar, x1_sar: sar     t0/t1 (B, C_sar, H, W)
    """
    def __init__(
        self,
        optical_in_channels: int,
        sar_in_channels: int,
        classes: int = 1,
        encoder_name: str = "timm_efficientnet_b1",
        encoder_weights: Optional[str] = "noisy-student",
        encoder_depth: int = 5,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        time_fusion_mode: str = "absdiff",
        modal_fusion_mode: str = "concat",  # or "se"
        decoder_interpolation: str = "nearest",
    ):
        super().__init__()
        enc_name = _normalize_encoder_name(encoder_name)

        # Encoders per modality (weights shared over time within each modality)
        self.enc_opt = smp.encoders.get_encoder(
            enc_name, in_channels=optical_in_channels, depth=encoder_depth, weights=encoder_weights
        )
        self.enc_sar = smp.encoders.get_encoder(
            enc_name, in_channels=sar_in_channels, depth=encoder_depth, weights=encoder_weights
        )
        self._enc_channels = list(self.enc_opt.out_channels)  # same layout for both

        # Time fusion inside each modality
        self.tf_opt = nn.ModuleList([_TimeFusion(c, time_fusion_mode) for c in self._enc_channels])
        self.tf_sar = nn.ModuleList([_TimeFusion(c, time_fusion_mode) for c in self._enc_channels])

        # Cross-modal fusion per scale
        self.mf = nn.ModuleList([_ModalFusion(c, modal_fusion_mode) for c in self._enc_channels])

        # Decoder & head
        try:
            self.decoder = UnetDecoder(
                encoder_channels=self._enc_channels,
                decoder_channels=decoder_channels,
                n_blocks=encoder_depth,
                use_norm="batchnorm",
                add_center_block=enc_name.startswith("vgg"),
                attention_type=None,
                interpolation_mode=decoder_interpolation,
            )
        except TypeError:
            self.decoder = UnetDecoder(
                encoder_channels=self._enc_channels,
                decoder_channels=list(decoder_channels),
                n_blocks=encoder_depth,
                use_batchnorm=True,
                center=enc_name.startswith("vgg"),
                attention_type=None,
            )
        self.seg_head = SegmentationHead(
            in_channels=decoder_channels[-1], out_channels=classes, activation=None, kernel_size=3
        )

    def forward(
        self,
        x0_opt: torch.Tensor,
        x1_opt: torch.Tensor,
        x0_sar: torch.Tensor,
        x1_sar: torch.Tensor,
    ) -> torch.Tensor:
        # Encode per modality & time
        f0o: List[torch.Tensor] = self.enc_opt(x0_opt)
        f1o: List[torch.Tensor] = self.enc_opt(x1_opt)
        f0s: List[torch.Tensor] = self.enc_sar(x0_sar)
        f1s: List[torch.Tensor] = self.enc_sar(x1_sar)

        # Time fusion inside each modality
        fo: List[torch.Tensor] = [self.tf_opt[i](f0o[i], f1o[i]) for i in range(len(f0o))]
        fs: List[torch.Tensor] = [self.tf_sar[i](f0s[i], f1s[i]) for i in range(len(f0s))]

        # Cross-modal fusion per scale (keeps encoder channel dims)
        fused: List[torch.Tensor] = [self.mf[i](fo[i], fs[i]) for i in range(len(fo))]

        dec = self.decoder(*fused)
        return self.seg_head(dec)


# ------------------------
# Sanity checks
# ------------------------

def _sanity_single():
    B, H, W = 2, 256, 256
    x0 = torch.randn(B, 4, H, W)
    x1 = torch.randn(B, 4, H, W)
    model = SiameseUNetSMPShared(
        in_channels=4,
        classes=1,
        encoder_name="timm_efficientnet_b1",  # underscore or hyphen accepted
        encoder_weights="imagenet",
        encoder_depth=5,
        decoder_channels=(256, 128, 64, 32, 16),
        time_fusion_mode="concat_diff",
    )
    with torch.inference_mode():
        y = model(x0, x1)
    print("SiameseUNetSMPShared:", y.shape)


def _sanity_dual():
    B, H, W = 2, 256, 256
    x0_opt = torch.randn(B, 4, H, W)
    x1_opt = torch.randn(B, 4, H, W)
    x0_sar = torch.randn(B, 2, H, W)
    x1_sar = torch.randn(B, 2, H, W)
    model = SiameseUNetSMPDualStream(
        optical_in_channels=4,
        sar_in_channels=2,
        classes=1,
        encoder_name="timm-efficientnet-b1",
        encoder_weights="imagenet",
        time_fusion_mode="absdiff",
        modal_fusion_mode="se",
    )
    with torch.inference_mode():
        y = model(x0_opt, x1_opt, x0_sar, x1_sar)
    print("SiameseUNetSMPDualStream:", y.shape)


if __name__ == "__main__":
    _sanity_single()
    _sanity_dual()