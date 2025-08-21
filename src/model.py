from __future__ import annotations
from typing import List, Sequence, Optional

import torch
import torch.nn as nn

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


def build_model(hps):
    """Builds a PyTorch segmentation model.
    Builds a PyTorch segmentation model according to the type of architecture, encoder backbone,
    number of input channel and classes specified.
    Args:
        hps (Hyperparams): Hyperparameters of the model to be built.
    Raises:
        NotImplementedError: The model architecture is not implemented.
    Returns:
        nn.Module: PyTorch segmentation model ready for inference
    """
    num_classes = hps.num_classes + 1
    backbone = hps.backbone if hps.backbone[:4] != "timm" else hps.backbone.replace("_", "-")
    encoder_weights = "imagenet" if hps.backbone[:4] != "timm" else "noisy-student"
    decoder_channels = [int(x * hps.smp_decoder_channels_mult) for x in [256, 128, 64, 32, 16]]
    decoder_attention_type = None if hps.smp_decoder_use_attention == 0 else "scse"

    if hps.model == "siamese_unet":
        model = SiameseUNetSMPShared(
            encoder_name=backbone,
            encoder_depth=5,
            encoder_weights=hps.pretrained,
            decoder_channels=decoder_channels,
            time_fusion_mode="concat_diff",
            in_channels=hps.input_channel,
            classes=num_classes,
        )
    elif hps.model == "concat_unet":
        model = smp.Unet(
            encoder_name=backbone,
            encoder_depth=5,
            encoder_weights=hps.pretrained,
            decoder_use_batchnorm=hps.smp_decoder_use_batchnorm,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            in_channels=hps.input_channel,
            classes=num_classes,
        )
    elif hps.model == "concat_unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name=backbone,
            encoder_depth=5,
            encoder_weights=hps.pretrained,
            decoder_use_batchnorm=hps.smp_decoder_use_batchnorm,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            in_channels=hps.input_channel,
            classes=num_classes,
        )
    else:
        raise NotImplementedError(hps.model)
    print(
        hps.model,
        backbone,
        encoder_weights,
        f"BatchNorm: {hps.smp_decoder_use_batchnorm==1}",
        f"Decoder Channel: {decoder_channels}, Decoder Attention Type: {decoder_attention_type}",
    )
    return model

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


# --------------------------------------------------
# Siamese U-Net (shared weights) — single modality
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



def _sanity():
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


if __name__ == "__main__":
    _sanity()