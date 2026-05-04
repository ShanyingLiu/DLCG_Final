"""Shared CNN decoder: ResNet50 spatial feats (B, C, 4, 4) -> HDR envmap (B, 3, 64, 128)."""

import torch.nn as nn


class FiLMUpBlock(nn.Module):
    """ConvTranspose -> BN -> FiLM (scale/shift) -> ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4,
                                         stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.out_ch = out_ch

    def forward(self, x, gamma, beta):
        x = self.bn(self.deconv(x))
        # gamma/beta: (B, out_ch). FiLM as (1+gamma)*x + beta so zero-init
        # leaves the block as identity scale on the BN output.
        x = x * (1.0 + gamma.view(-1, self.out_ch, 1, 1)) \
            + beta.view(-1, self.out_ch, 1, 1)
        return self.act(x)


class FiLMEnvmapDecoder(nn.Module):
    """FiLM-modulated variant of EnvmapDecoder.

    Each upsample block receives a per-channel (gamma, beta) pair that scales
    and shifts the post-BN feature map. The FiLM tensor is supplied by the
    caller (typically a small MLP from predicted material parameters), which
    lets the decoder be material-*guided* rather than material-*aware*.
    """

    DECODER_CHANNELS = (128, 64, 32, 16)

    def __init__(self, in_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.up1 = FiLMUpBlock(256, 128)
        self.up2 = FiLMUpBlock(128, 64)
        self.up3 = FiLMUpBlock(64, 32)
        self.up4 = FiLMUpBlock(32, 16)

        self.up_w = nn.ConvTranspose2d(
            16, 16, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
        )
        self.bn_w = nn.BatchNorm2d(16)
        self.act_w = nn.ReLU(inplace=True)

        self.head = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.out_act = nn.Softplus()

    @staticmethod
    def total_film_dims():
        return 2 * sum(FiLMEnvmapDecoder.DECODER_CHANNELS)  # gamma + beta

    def _split_film(self, film):
        chans = self.DECODER_CHANNELS
        out = []
        offset = 0
        for c in chans:
            g = film[:, offset:offset + c]
            b = film[:, offset + c:offset + 2 * c]
            out.append((g, b))
            offset += 2 * c
        return out

    def forward(self, x, film):
        gb = self._split_film(film)
        x = self.proj(x)
        x = self.up1(x, *gb[0])
        x = self.up2(x, *gb[1])
        x = self.up3(x, *gb[2])
        x = self.up4(x, *gb[3])
        x = self.act_w(self.bn_w(self.up_w(x)))
        return self.out_act(self.head(x))


class EnvmapDecoder(nn.Module):
    """Upsample (B, in_channels, 4, 4) -> (B, 3, 64, 128) HDR envmap.

    Channel ladder:  in_channels -> 256 (1x1) -> 128 -> 64 -> 32 -> 16 -> 3.
    Spatial ladder:  4 -> 8 -> 16 -> 32 -> 64 -> (64, 128) via final stride (1,2).
    Output activation: Softplus, ensures non-negative HDR with smooth gradient.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, 256, kernel_size=1)

        self.up1 = self._block(256, 128)   #  4 -> 8
        self.up2 = self._block(128, 64)    #  8 -> 16
        self.up3 = self._block(64, 32)     # 16 -> 32
        self.up4 = self._block(32, 16)     # 32 -> 64

        # Asymmetric upsample: H stays 64, W goes 64 -> 128.
        self.up_w = nn.ConvTranspose2d(
            16, 16, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
        )
        self.bn_w = nn.BatchNorm2d(16)
        self.act_w = nn.ReLU(inplace=True)

        self.head = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.out_act = nn.Softplus()

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.proj(x)        # (B, 256, 4, 4)
        x = self.up1(x)         # (B, 128, 8, 8)
        x = self.up2(x)         # (B, 64, 16, 16)
        x = self.up3(x)         # (B, 32, 32, 32)
        x = self.up4(x)         # (B, 16, 64, 64)
        x = self.act_w(self.bn_w(self.up_w(x)))  # (B, 16, 64, 128)
        x = self.head(x)        # (B, 3, 64, 128)
        return self.out_act(x)
