"""
Two architectures for change segmentation:

1) Siamese U-Net (shared weights, single-modality)
   - One encoder shared across t0 ("before") and t1 ("after").
   - Per-scale fusion of features via concatenation, (abs)difference, or both.
   - Standard U-Net decoder predicts the change mask.

Notes
-----
- Designed to be minimal-yet-strong baselines you can extend (deeper encoders, more scales, normalization blocks, etc.).
- The code is pure PyTorch with clear shapes; no external deps beyond torch.
- You can plug in pretrained backbones by replacing `UNetEncoder` with timm/segformer encoders, as long as you adapt channel dims.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------
# U-Net building blocks
# ------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if needed (to handle odd sizes)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ------------------------
# Encoders / Decoders
# ------------------------

class UNetEncoder(nn.Module):
    """Classic 4-level U-Net encoder, returns skip features and bottleneck.

    Channels: C1, C2, C3, C4, C5 = base, 2*base, 4*base, 8*base, 16*base.
    Returns (skips, bottleneck) where skips = [e1, e2, e3, e4].
    """
    def __init__(self, in_ch: int, base_ch: int = 64):
        super().__init__()
        C1, C2, C3, C4, C5 = base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16
        self.inc = DoubleConv(in_ch, C1)
        self.down1 = Down(C1, C2)
        self.down2 = Down(C2, C3)
        self.down3 = Down(C3, C4)
        self.down4 = Down(C4, C5)
        self.channels = (C1, C2, C3, C4, C5)

    def forward(self, x: torch.Tensor):
        e1 = self.inc(x)      # (B, C1, H,   W)
        e2 = self.down1(e1)   # (B, C2, H/2, W/2)
        e3 = self.down2(e2)   # (B, C3, H/4, W/4)
        e4 = self.down3(e3)   # (B, C4, H/8, W/8)
        b  = self.down4(e4)   # (B, C5, H/16,W/16)
        return [e1, e2, e3, e4], b


class UNetDecoder(nn.Module):
    """Standard 4-stage decoder that consumes fused skips and fused bottleneck.

    Assumes skip channel sizes follow the encoder base scheme (C1..C4) and bottleneck C5.
    """
    def __init__(self, base_ch: int = 64, out_ch: int = 1, bilinear: bool = True):
        super().__init__()
        C1, C2, C3, C4, C5 = base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16
        self.up1 = Up(C5 + C4, C4, bilinear)
        self.up2 = Up(C4 + C3, C3, bilinear)
        self.up3 = Up(C3 + C2, C2, bilinear)
        self.up4 = Up(C2 + C1, C1, bilinear)
        self.outc = OutConv(C1, out_ch)

    def forward(self, b: torch.Tensor, f_skips) -> torch.Tensor:
        f1, f2, f3, f4 = f_skips  # C1..C4
        x = self.up1(b, f4)
        x = self.up2(x, f3)
        x = self.up3(x, f2)
        x = self.up4(x, f1)
        return self.outc(x)


# ------------------------
# Fusion blocks
# ------------------------

class TimeFusion(nn.Module):
    """Fuse features from two times t0, t1 for a given scale.

    fusion_mode ∈ {"concat", "diff", "absdiff", "concat_diff"}
      - "concat": cat along C then reduce via 1x1 conv to original C
      - "diff":   (t1 - t0) then 3x3 conv
      - "absdiff":|t1 - t0| then 3x3 conv
      - "concat_diff": [t0, t1, |t1-t0|] then 1x1 reduce
    Output channels == in_ch (so decoder channel math stays simple).
    """
    def __init__(self, in_ch: int, fusion_mode: str = "absdiff"):
        super().__init__()
        self.mode = fusion_mode
        if fusion_mode == "concat":
            self.proj = nn.Conv2d(in_ch * 2, in_ch, kernel_size=1, bias=False)
        elif fusion_mode == "concat_diff":
            self.proj = nn.Conv2d(in_ch * 3, in_ch, kernel_size=1, bias=False)
        elif fusion_mode in {"diff", "absdiff"}:
            self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False)
        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        self.bn = nn.BatchNorm2d(in_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
        if self.mode == "concat":
            x = torch.cat([f0, f1], dim=1)
        elif self.mode == "diff":
            x = f1 - f0
        elif self.mode == "absdiff":
            x = torch.abs(f1 - f0)
        elif self.mode == "concat_diff":
            x = torch.cat([f0, f1, torch.abs(f1 - f0)], dim=1)
        else:
            raise RuntimeError
        x = self.proj(x)
        return self.act(self.bn(x))


class ModalFusion(nn.Module):
    """Fuse two modality features (e.g., Optical and SAR) at the same scale.

    mode ∈ {"concat", "se"}
      - concat: [opt, sar] → 1x1 → C
      - se:     squeeze-and-excitation style gating per modality then sum (channels remain C)
    Assumes both inputs have the same channel count C.
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
                nn.Conv2d(r, 2, 1),  # logits per modality
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Unsupported modal fusion: {mode}")

    def forward(self, f_opt: torch.Tensor, f_sar: torch.Tensor) -> torch.Tensor:
        if self.mode == "concat":
            x = self.proj(torch.cat([f_opt, f_sar], dim=1))
            return self.act(self.bn(x))
        else:  # SE gating
            x = torch.cat([f_opt, f_sar], dim=1)
            w = self.se(x)  # (B, 2, 1, 1)
            w_opt = w[:, 0:1]
            w_sar = w[:, 1:2]
            return w_opt * f_opt + w_sar * f_sar


# ------------------------
# 1) Siamese U-Net (shared weights)
# ------------------------

class SiameseUNetShared(nn.Module):
    """Siamese U-Net with a single encoder shared across times and per-scale fusion.

    Args:
        in_ch: input channels (e.g., 4 for S2 RGB+NIR)
        base_ch: U-Net base channels
        fusion_mode: how to fuse t0 vs t1 features per scale (see TimeFusion)
        out_ch: number of output channels (1 for binary change)
        bilinear: use bilinear upsampling in decoder
    """
    def __init__(self, in_ch: int, base_ch: int = 64, fusion_mode: str = "absdiff", out_ch: int = 1, bilinear: bool = True):
        super().__init__()
        self.encoder = UNetEncoder(in_ch, base_ch)
        C1, C2, C3, C4, C5 = self.encoder.channels
        # Per-scale time fusion blocks (skips + bottleneck)
        self.tf1 = TimeFusion(C1, fusion_mode)
        self.tf2 = TimeFusion(C2, fusion_mode)
        self.tf3 = TimeFusion(C3, fusion_mode)
        self.tf4 = TimeFusion(C4, fusion_mode)
        self.tfb = TimeFusion(C5, fusion_mode)
        # Decoder consumes fused features of original dims
        self.decoder = UNetDecoder(base_ch=base_ch, out_ch=out_ch, bilinear=bilinear)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        # Encode both times with shared weights
        s0, b0 = self.encoder(x0)
        s1, b1 = self.encoder(x1)
        # Fuse per scale
        f1 = self.tf1(s0[0], s1[0])
        f2 = self.tf2(s0[1], s1[1])
        f3 = self.tf3(s0[2], s1[2])
        f4 = self.tf4(s0[3], s1[3])
        fb = self.tfb(b0, b1)
        # Decode
        logits = self.decoder(fb, [f1, f2, f3, f4])
        return logits

# ------------------------
# Simple sanity tests
# ------------------------

# def _sanity_single():
#     B, H, W = 2, 256, 256
#     x0 = torch.randn(B, 4, H, W)  # e.g., S2 RGB+NIR
#     x1 = torch.randn(B, 4, H, W)
#     model = SiameseUNetShared(in_ch=4, base_ch=32, fusion_mode="concat_diff", out_ch=1)
#     y = model(x0, x1)
#     print("SiameseUNetShared", y.shape)