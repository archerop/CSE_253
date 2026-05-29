from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelSummary:
    name: str
    num_parameters: int
    num_trainable_parameters: int


def count_parameters(model: nn.Module) -> ModelSummary:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return ModelSummary(
        name=model.__class__.__name__,
        num_parameters=total,
        num_trainable_parameters=trainable,
    )


class ConvBlock(nn.Module):
    """
    Small Conv-BN-ReLU block for the SimpleCNN baseline.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleMidiToLogMelCNN(nn.Module):
    """
    Lightweight learned baseline for Option 4.

    Task:
        MIDI-derived piano-roll features -> log-mel spectrogram

    Input:
        piano_roll: [B, 3, T, 88]

    Output:
        pred_log_mel: [B, 80, T]

    Design:
    - Treat piano-roll as an image with height=pitch and width=time.
    - Use a few local 2D convolution blocks.
    - Preserve time resolution.
    - Resize vertical axis from 88 piano keys to 80 mel bins.
    - Use a linear output head for log-mel regression.

    This is intentionally not a U-Net. It is a learned baseline used before
    training the stronger U-Net main model.
    """

    def __init__(
        self,
        input_channels: int = 3,
        n_mels: int = 80,
        base_channels: int = 32,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.n_mels = n_mels
        self.base_channels = base_channels

        self.frontend = nn.Sequential(
            ConvBlock(input_channels, base_channels, dropout=dropout),
            ConvBlock(base_channels, base_channels * 2, dropout=dropout),
            ConvBlock(base_channels * 2, base_channels * 2, dropout=dropout),
        )

        self.refine = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels * 2, dropout=dropout),
            ConvBlock(base_channels * 2, base_channels, dropout=dropout),
        )

        self.output = nn.Conv2d(
            base_channels,
            1,
            kernel_size=1,
            padding=0,
        )

    def forward(self, piano_roll: torch.Tensor) -> torch.Tensor:
        """
        piano_roll:
            [B, 3, T, 88]

        returns:
            [B, 80, T]
        """
        if piano_roll.ndim != 4:
            raise ValueError(
                f"Expected piano_roll shape [B, C, T, 88], got {piano_roll.shape}"
            )

        if piano_roll.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got {piano_roll.shape[1]}"
            )

        # [B, C, T, 88] -> [B, C, 88, T]
        x = piano_roll.permute(0, 1, 3, 2).contiguous()

        x = self.frontend(x)

        # Resize pitch axis 88 -> mel axis 80 while preserving time frames.
        target_time_frames = x.shape[-1]
        x = F.interpolate(
            x,
            size=(self.n_mels, target_time_frames),
            mode="bilinear",
            align_corners=False,
        )

        x = self.refine(x)
        x = self.output(x)

        # [B, 1, 80, T] -> [B, 80, T]
        x = x.squeeze(1)

        return x



class ResidualDilatedBlock(nn.Module):
    """
    Residual convolution block with temporal dilation.

    Input/output shape:
        [B, C, 80, T]

    Dilation is applied mainly along the time axis, so the model can capture
    attack/decay patterns over a longer temporal range without becoming a U-Net.
    """

    def __init__(
        self,
        channels: int,
        dilation_time: int,
        dropout: float = 0.05,
        groups: int = 8,
    ) -> None:
        super().__init__()

        if channels % groups != 0:
            groups = 1

        self.net = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=(1, dilation_time),
                dilation=(1, dilation_time),
                bias=False,
            ),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, channels),
        )

        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class ResidualDilatedMidiToLogMelCNN(nn.Module):
    """
    Formal CNN learned baseline for Option 4.

    Task:
        MIDI-derived piano-roll features -> log-mel spectrogram

    Input:
        piano_roll: [B, 3, T, 88]

    Output:
        pred_log_mel: [B, 80, T]

    Design:
    - Treat piano roll as an image: height=pitch, width=time.
    - Extract local pitch/time features with convolution.
    - Resize pitch axis 88 -> mel axis 80.
    - Apply residual dilated CNN blocks on the 80 x T representation.
    - No encoder-decoder downsampling and no U-Net skip connections.

    This makes it a stronger CNN baseline while still being architecturally
    distinct from the later U-Net main model.
    """

    def __init__(
        self,
        input_channels: int = 3,
        n_mels: int = 80,
        hidden_channels: int = 64,
        num_blocks: int = 8,
        dropout: float = 0.05,
        groups: int = 8,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.n_mels = n_mels
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks

        if hidden_channels % groups != 0:
            groups = 1

        self.input_projection = nn.Sequential(
            nn.Conv2d(
                input_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
        )

        dilation_pattern = [1, 2, 4, 8]
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                ResidualDilatedBlock(
                    channels=hidden_channels,
                    dilation_time=dilation_pattern[i % len(dilation_pattern)],
                    dropout=dropout,
                    groups=groups,
                )
            )
        self.residual_blocks = nn.Sequential(*blocks)

        self.output_head = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels // 2,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(max(1, groups // 2), hidden_channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels // 2,
                1,
                kernel_size=1,
                padding=0,
            ),
        )

    def forward(self, piano_roll: torch.Tensor) -> torch.Tensor:
        """
        piano_roll:
            [B, 3, T, 88]

        returns:
            [B, 80, T]
        """
        if piano_roll.ndim != 4:
            raise ValueError(
                f"Expected piano_roll shape [B, C, T, 88], got {piano_roll.shape}"
            )

        if piano_roll.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got {piano_roll.shape[1]}"
            )

        # [B, C, T, 88] -> [B, C, 88, T]
        x = piano_roll.permute(0, 1, 3, 2).contiguous()

        x = self.input_projection(x)

        # Resize vertical axis from piano pitch bins to mel bins.
        target_time_frames = x.shape[-1]
        x = F.interpolate(
            x,
            size=(self.n_mels, target_time_frames),
            mode="bilinear",
            align_corners=False,
        )

        x = self.residual_blocks(x)
        x = self.output_head(x)

        # [B, 1, 80, T] -> [B, 80, T]
        return x.squeeze(1)



class UNetConvBlock(nn.Module):
    """
    Two-layer convolution block for ContourNet-lite U-Net.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        groups: int = 8,
    ) -> None:
        super().__init__()

        if out_channels % groups != 0:
            groups = 1

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetUpBlock(nn.Module):
    """
    Upsample + skip connection + convolution block.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.conv = UNetConvBlock(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
            dropout=dropout,
            groups=groups,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ContourNetLiteUNet(nn.Module):
    """
    PerformanceNet-inspired ContourNet-lite model.

    Task:
        MIDI-derived piano-roll features -> log-mel spectrogram

    Input:
        piano_roll: [B, 3, T, 88]

    Internal condition:
        fixed harmonic pitch-to-mel projection:
        [B, 3, T, 88] -> [B, 3, 80, T]

    Output:
        pred_log_mel: [B, 80, T]

    This model is a lightweight U-Net-style encoder-decoder. It is not a full
    PerformanceNet reproduction, but it follows the same main idea that
    piano-roll-to-spectrogram mapping benefits from a structured convolutional
    encoder-decoder with skip connections.
    """

    def __init__(
        self,
        input_channels: int = 3,
        n_mels: int = 80,
        midi_low: int = 21,
        fmin: float = 30.0,
        fmax: float = 11025.0,
        base_channels: int = 32,
        dropout: float = 0.05,
        groups: int = 8,
        condition_strength: float = 1.0,
    ) -> None:
        super().__init__()

        from app.option4.mel_conditioning import PianoRollToMelCondition

        self.conditioner = PianoRollToMelCondition(
            n_pitches=88,
            midi_low=midi_low,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            sigma_bins=1.25,
            n_harmonics=8,
            harmonic_decay=1.0,
            condition_strength=condition_strength,
            log_scale=True,
        )

        c = base_channels

        self.enc1 = UNetConvBlock(input_channels, c, dropout=dropout, groups=groups)
        self.enc2 = UNetConvBlock(c, c * 2, dropout=dropout, groups=groups)
        self.enc3 = UNetConvBlock(c * 2, c * 4, dropout=dropout, groups=groups)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNetConvBlock(c * 4, c * 8, dropout=dropout, groups=groups)

        self.up3 = UNetUpBlock(c * 8, c * 4, c * 4, dropout=dropout, groups=groups)
        self.up2 = UNetUpBlock(c * 4, c * 2, c * 2, dropout=dropout, groups=groups)
        self.up1 = UNetUpBlock(c * 2, c, c, dropout=dropout, groups=groups)

        self.output_head = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups if c % groups == 0 else 1, c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, 1, kernel_size=1),
        )

    def forward(self, piano_roll: torch.Tensor) -> torch.Tensor:
        # [B, 3, T, 88] -> [B, 3, 80, T]
        x = self.conditioner(piano_roll)

        e1 = self.enc1(x)                  # [B, C, 80, T]
        e2 = self.enc2(self.pool(e1))      # [B, 2C, 40, T/2]
        e3 = self.enc3(self.pool(e2))      # [B, 4C, 20, T/4]

        b = self.bottleneck(self.pool(e3)) # [B, 8C, 10, T/8]

        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        out = self.output_head(d1)         # [B, 1, 80, T]
        return out.squeeze(1)              # [B, 80, T]



class MelProjectedResidualBlock(nn.Module):
    """
    Residual CNN block on mel-aligned conditioning.

    Shape:
        input/output: [B, C, 80, T]

    This uses temporal dilation to increase receptive field without using
    U-Net downsampling or skip connections.
    """

    def __init__(
        self,
        channels: int,
        dilation_time: int = 1,
        dropout: float = 0.05,
        groups: int = 8,
    ) -> None:
        super().__init__()

        if channels % groups != 0:
            groups = 1

        self.net = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=(1, dilation_time),
                dilation=(1, dilation_time),
                bias=False,
            ),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, channels),
        )

        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class MelProjectedResidualCNN(nn.Module):
    """
    Fair learned CNN baseline for Option 4.

    Input:
        piano_roll: [B, 3, T, 88]

    Internal conditioning:
        fixed harmonic pitch-to-mel projection:
        [B, 3, T, 88] -> [B, 3, 80, T]

    Output:
        pred_log_mel: [B, 80, T]

    This baseline uses the same mel-aligned conditioning as ContourNet-lite
    U-Net, but it does not use encoder-decoder downsampling, upsampling, or
    U-Net skip connections. Therefore it is a fairer architecture baseline
    than the failed direct pitch-axis CNN.
    """

    def __init__(
        self,
        input_channels: int = 3,
        n_mels: int = 80,
        midi_low: int = 21,
        fmin: float = 30.0,
        fmax: float = 11025.0,
        hidden_channels: int = 64,
        num_blocks: int = 8,
        dropout: float = 0.05,
        groups: int = 8,
        condition_strength: float = 1.0,
    ) -> None:
        super().__init__()

        from app.option4.mel_conditioning import PianoRollToMelCondition

        self.conditioner = PianoRollToMelCondition(
            n_pitches=88,
            midi_low=midi_low,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            sigma_bins=1.25,
            n_harmonics=8,
            harmonic_decay=1.0,
            condition_strength=condition_strength,
            log_scale=True,
        )

        if hidden_channels % groups != 0:
            groups = 1

        self.input_projection = nn.Sequential(
            nn.Conv2d(
                input_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
        )

        dilation_pattern = [1, 2, 4, 8]
        self.blocks = nn.Sequential(
            *[
                MelProjectedResidualBlock(
                    channels=hidden_channels,
                    dilation_time=dilation_pattern[i % len(dilation_pattern)],
                    dropout=dropout,
                    groups=groups,
                )
                for i in range(num_blocks)
            ]
        )

        self.output_head = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels // 2,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(max(1, groups // 2), hidden_channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels // 2,
                1,
                kernel_size=1,
            ),
        )

    def forward(self, piano_roll: torch.Tensor) -> torch.Tensor:
        if piano_roll.ndim != 4:
            raise ValueError(
                f"Expected piano_roll shape [B, C, T, 88], got {piano_roll.shape}"
            )

        # [B, 3, T, 88] -> [B, 3, 80, T]
        x = self.conditioner(piano_roll)

        x = self.input_projection(x)
        x = self.blocks(x)
        x = self.output_head(x)

        return x.squeeze(1)  # [B, 80, T]



class StrongResidualConvBlock(nn.Module):
    """
    Residual convolution block used in StrongContourNetLiteUNet.

    Compared with the earlier simple U-Net conv block, this block is easier to
    train when the network is deeper because it uses residual shortcuts.

    Shape:
        input:  [B, C_in, H, W]
        output: [B, C_out, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation_time: int = 1,
        dropout: float = 0.0,
        groups: int = 8,
    ) -> None:
        super().__init__()

        if out_channels % groups != 0:
            groups = 1

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=(1, dilation_time),
                dilation=(1, dilation_time),
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
        )

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False,
            )

        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.shortcut(x) + self.net(x))


class StrongResidualConvStack(nn.Module):
    """
    Stack of residual convolution blocks.

    The first block may change channel dimension. Later blocks keep the same
    channel dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 2,
        dropout: float = 0.0,
        groups: int = 8,
        dilation_time: int = 1,
    ) -> None:
        super().__init__()

        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")

        blocks = [
            StrongResidualConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                dilation_time=dilation_time,
                dropout=dropout,
                groups=groups,
            )
        ]

        for _ in range(num_blocks - 1):
            blocks.append(
                StrongResidualConvBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    dilation_time=dilation_time,
                    dropout=dropout,
                    groups=groups,
                )
            )

        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnedDownsample2d(nn.Module):
    """
    Learned downsampling layer.

    This replaces fixed max-pooling with a stride-2 convolution. For spectrogram
    regression, learned downsampling is often more flexible than max-pooling.
    """

    def __init__(
        self,
        channels: int,
        dropout: float = 0.0,
        groups: int = 8,
    ) -> None:
        super().__init__()

        if channels % groups != 0:
            groups = 1

        self.net = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StrongUNetUpBlock(nn.Module):
    """
    Bilinear upsample + skip concatenation + residual conv stack.

    Bilinear upsampling avoids transposed-convolution checkerboard artifacts.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_blocks: int = 2,
        dropout: float = 0.0,
        groups: int = 8,
    ) -> None:
        super().__init__()

        self.conv = StrongResidualConvStack(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            dropout=dropout,
            groups=groups,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class StrongContourNetLiteUNet(nn.Module):
    """
    Strong PerformanceNet-inspired ContourNet-lite U-Net.

    Input:
        piano_roll: [B, 3, T, 88]

    Conditioning:
        fixed harmonic pitch-to-mel projection:
        [B, 3, T, 88] -> [B, 3, 80, T]

    Output:
        pred_log_mel: [B, 80, T]

    Design choices:
    - Same mel-projected conditioning as the fair Residual CNN baseline.
    - Residual conv blocks instead of plain conv blocks.
    - Learned downsampling instead of max pooling.
    - Temporal-dilated bottleneck blocks for longer time context.
    - U-Net skip connections for preserving note timing and local detail.
    - Post-U-Net residual refinement inspired by PerformanceNet's texture
      refinement idea, but kept compact for 80-bin log-mel targets.
    """

    def __init__(
        self,
        input_channels: int = 3,
        n_mels: int = 80,
        midi_low: int = 21,
        fmin: float = 30.0,
        fmax: float = 11025.0,
        base_channels: int = 48,
        blocks_per_level: int = 2,
        bottleneck_dilations: tuple[int, ...] = (1, 2, 4, 8),
        refinement_blocks: int = 3,
        dropout: float = 0.05,
        groups: int = 8,
        condition_strength: float = 1.0,
    ) -> None:
        super().__init__()

        from app.option4.mel_conditioning import PianoRollToMelCondition

        self.conditioner = PianoRollToMelCondition(
            n_pitches=88,
            midi_low=midi_low,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            sigma_bins=1.25,
            n_harmonics=8,
            harmonic_decay=1.0,
            condition_strength=condition_strength,
            log_scale=True,
        )

        c = base_channels

        self.stem = StrongResidualConvStack(
            in_channels=input_channels,
            out_channels=c,
            num_blocks=blocks_per_level,
            dropout=dropout,
            groups=groups,
        )

        self.down1 = LearnedDownsample2d(c, dropout=dropout, groups=groups)
        self.enc2 = StrongResidualConvStack(
            in_channels=c,
            out_channels=c * 2,
            num_blocks=blocks_per_level,
            dropout=dropout,
            groups=groups,
        )

        self.down2 = LearnedDownsample2d(c * 2, dropout=dropout, groups=groups)
        self.enc3 = StrongResidualConvStack(
            in_channels=c * 2,
            out_channels=c * 4,
            num_blocks=blocks_per_level,
            dropout=dropout,
            groups=groups,
        )

        self.down3 = LearnedDownsample2d(c * 4, dropout=dropout, groups=groups)
        self.bottleneck_in = StrongResidualConvStack(
            in_channels=c * 4,
            out_channels=c * 8,
            num_blocks=blocks_per_level,
            dropout=dropout,
            groups=groups,
        )

        bottleneck_blocks = []
        for dilation in bottleneck_dilations:
            bottleneck_blocks.append(
                StrongResidualConvBlock(
                    in_channels=c * 8,
                    out_channels=c * 8,
                    dilation_time=int(dilation),
                    dropout=dropout,
                    groups=groups,
                )
            )
        self.bottleneck_dilated = nn.Sequential(*bottleneck_blocks)

        self.up3 = StrongUNetUpBlock(
            in_channels=c * 8,
            skip_channels=c * 4,
            out_channels=c * 4,
            num_blocks=blocks_per_level,
            dropout=dropout,
            groups=groups,
        )

        self.up2 = StrongUNetUpBlock(
            in_channels=c * 4,
            skip_channels=c * 2,
            out_channels=c * 2,
            num_blocks=blocks_per_level,
            dropout=dropout,
            groups=groups,
        )

        self.up1 = StrongUNetUpBlock(
            in_channels=c * 2,
            skip_channels=c,
            out_channels=c,
            num_blocks=blocks_per_level,
            dropout=dropout,
            groups=groups,
        )

        if refinement_blocks > 0:
            self.refinement = StrongResidualConvStack(
                in_channels=c,
                out_channels=c,
                num_blocks=refinement_blocks,
                dropout=dropout,
                groups=groups,
            )
        else:
            self.refinement = nn.Identity()

        self.output_head = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups if c % groups == 0 else 1, c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, 1, kernel_size=1),
        )

    def forward(self, piano_roll: torch.Tensor) -> torch.Tensor:
        if piano_roll.ndim != 4:
            raise ValueError(
                f"Expected piano_roll shape [B, C, T, 88], got {piano_roll.shape}"
            )

        # [B, 3, T, 88] -> [B, 3, 80, T]
        x = self.conditioner(piano_roll)

        e1 = self.stem(x)              # [B, C, 80, T]
        e2 = self.enc2(self.down1(e1)) # [B, 2C, 40, ~T/2]
        e3 = self.enc3(self.down2(e2)) # [B, 4C, 20, ~T/4]

        b = self.bottleneck_in(self.down3(e3)) # [B, 8C, 10, ~T/8]
        b = self.bottleneck_dilated(b)

        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        refined = self.refinement(d1)
        out = self.output_head(refined)

        return out.squeeze(1)  # [B, 80, T]



class LinearProjectedStftResidualCNN(nn.Module):
    """
    513-bin STFT learned residual CNN baseline.

    Input:
        piano_roll: [B, 3, T, 88]

    Conditioning:
        fixed harmonic pitch-to-linear-frequency projection:
        [B, 3, T, 88] -> [B, 3, F, T]

    Output:
        predicted log1p(STFT magnitude): [B, F, T]
    """

    def __init__(
        self,
        input_channels: int = 3,
        n_freq_bins: int = 513,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        midi_low: int = 21,
        hidden_channels: int = 48,
        num_blocks: int = 8,
        dropout: float = 0.05,
        groups: int = 8,
        condition_strength: float = 1.0,
    ) -> None:
        super().__init__()

        from app.option4.linear_conditioning import PianoRollToLinearFrequencyCondition

        self.conditioner = PianoRollToLinearFrequencyCondition(
            n_pitches=88,
            midi_low=midi_low,
            n_freq_bins=n_freq_bins,
            sample_rate=sample_rate,
            n_fft=n_fft,
            sigma_bins=1.25,
            n_harmonics=8,
            harmonic_decay=1.0,
            condition_strength=condition_strength,
            log_scale=True,
        )

        if hidden_channels % groups != 0:
            groups = 1

        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
        )

        dilation_pattern = [1, 2, 4, 8]
        self.blocks = nn.Sequential(
            *[
                MelProjectedResidualBlock(
                    channels=hidden_channels,
                    dilation_time=dilation_pattern[i % len(dilation_pattern)],
                    dropout=dropout,
                    groups=groups,
                )
                for i in range(num_blocks)
            ]
        )

        self.output_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(max(1, groups // 2), hidden_channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
        )

    def forward(self, piano_roll: torch.Tensor) -> torch.Tensor:
        x = self.conditioner(piano_roll)  # [B, 3, F, T]
        x = self.input_projection(x)
        x = self.blocks(x)
        x = self.output_head(x)
        return x.squeeze(1)


class LinearProjectedStftUNet(nn.Module):
    """
    513-bin STFT U-Net model.

    This is closer to the PerformanceNet-style setting than the 80-bin mel
    U-Net because the output frequency axis is much higher resolution.

    Input:
        piano_roll: [B, 3, T, 88]

    Output:
        predicted log1p(STFT magnitude): [B, F, T]
    """

    def __init__(
        self,
        input_channels: int = 3,
        n_freq_bins: int = 513,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        midi_low: int = 21,
        base_channels: int = 24,
        blocks_per_level: int = 2,
        bottleneck_dilations: tuple[int, ...] = (1, 2, 4, 8),
        refinement_blocks: int = 2,
        dropout: float = 0.05,
        groups: int = 8,
        condition_strength: float = 1.0,
    ) -> None:
        super().__init__()

        from app.option4.linear_conditioning import PianoRollToLinearFrequencyCondition

        self.conditioner = PianoRollToLinearFrequencyCondition(
            n_pitches=88,
            midi_low=midi_low,
            n_freq_bins=n_freq_bins,
            sample_rate=sample_rate,
            n_fft=n_fft,
            sigma_bins=1.25,
            n_harmonics=8,
            harmonic_decay=1.0,
            condition_strength=condition_strength,
            log_scale=True,
        )

        c = base_channels

        self.stem = StrongResidualConvStack(
            in_channels=input_channels,
            out_channels=c,
            num_blocks=blocks_per_level,
            dropout=dropout,
            groups=groups,
        )

        self.down1 = LearnedDownsample2d(c, dropout=dropout, groups=groups)
        self.enc2 = StrongResidualConvStack(
            in_channels=c,
            out_channels=c * 2,
            num_blocks=blocks_per_level,
            dropout=dropout,
            groups=groups,
        )

        self.down2 = LearnedDownsample2d(c * 2, dropout=dropout, groups=groups)
        self.enc3 = StrongResidualConvStack(
            in_channels=c * 2,
            out_channels=c * 4,
            num_blocks=blocks_per_level,
            dropout=dropout,
            groups=groups,
        )

        self.down3 = LearnedDownsample2d(c * 4, dropout=dropout, groups=groups)
        self.bottleneck_in = StrongResidualConvStack(
            in_channels=c * 4,
            out_channels=c * 8,
            num_blocks=blocks_per_level,
            dropout=dropout,
            groups=groups,
        )

        bottleneck_blocks = []
        for dilation in bottleneck_dilations:
            bottleneck_blocks.append(
                StrongResidualConvBlock(
                    in_channels=c * 8,
                    out_channels=c * 8,
                    dilation_time=int(dilation),
                    dropout=dropout,
                    groups=groups,
                )
            )
        self.bottleneck_dilated = nn.Sequential(*bottleneck_blocks)

        self.up3 = StrongUNetUpBlock(
            in_channels=c * 8,
            skip_channels=c * 4,
            out_channels=c * 4,
            num_blocks=blocks_per_level,
            dropout=dropout,
            groups=groups,
        )

        self.up2 = StrongUNetUpBlock(
            in_channels=c * 4,
            skip_channels=c * 2,
            out_channels=c * 2,
            num_blocks=blocks_per_level,
            dropout=dropout,
            groups=groups,
        )

        self.up1 = StrongUNetUpBlock(
            in_channels=c * 2,
            skip_channels=c,
            out_channels=c,
            num_blocks=blocks_per_level,
            dropout=dropout,
            groups=groups,
        )

        if refinement_blocks > 0:
            self.refinement = StrongResidualConvStack(
                in_channels=c,
                out_channels=c,
                num_blocks=refinement_blocks,
                dropout=dropout,
                groups=groups,
            )
        else:
            self.refinement = nn.Identity()

        self.output_head = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups if c % groups == 0 else 1, c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, 1, kernel_size=1),
        )

    def forward(self, piano_roll: torch.Tensor) -> torch.Tensor:
        x = self.conditioner(piano_roll)  # [B, 3, F, T]

        e1 = self.stem(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        b = self.bottleneck_in(self.down3(e3))
        b = self.bottleneck_dilated(b)

        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        refined = self.refinement(d1)
        out = self.output_head(refined)

        return out.squeeze(1)
