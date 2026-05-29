from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn

from app.option4.linear_conditioning import PianoRollToLinearFrequencyCondition


def _choose_groups(channels: int, preferred: int = 8) -> int:
    if channels % preferred == 0:
        return preferred
    for g in [4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class TextureResidualBlock(nn.Module):
    """
    Small residual block for one frequency band.
    """

    def __init__(
        self,
        channels: int,
        dropout: float = 0.05,
        dilation_time: int = 1,
    ) -> None:
        super().__init__()

        groups = _choose_groups(channels)

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
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
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


class BandTextureRefiner(nn.Module):
    """
    Band-specific residual correction network.

    Input:
        [B, C_in, F_band, T]

    Output:
        correction [B, 1, F_band, T]
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        num_blocks: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        groups = _choose_groups(hidden_channels)

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
        )

        dilation_pattern = [1, 2, 4]
        self.blocks = nn.Sequential(
            *[
                TextureResidualBlock(
                    channels=hidden_channels,
                    dropout=dropout,
                    dilation_time=dilation_pattern[i % len(dilation_pattern)],
                )
                for i in range(num_blocks)
            ]
        )

        self.output_proj = nn.Conv2d(hidden_channels, 1, kernel_size=1)

        # Start as identity refinement: correction ≈ 0.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.blocks(h)
        return self.output_proj(h)


@dataclass(frozen=True)
class TextureRefinerConfig:
    n_freq_bins: int = 513
    n_bands: int = 8
    hidden_channels: int = 32
    num_blocks_per_band: int = 3
    dropout: float = 0.05
    residual_scale: float = 0.2
    use_condition: bool = True


class MultiBandTextureRefiner(nn.Module):
    """
    PerformanceNet-inspired TextureNet-lite refinement.

    It refines a frozen STFT U-Net prediction using band-specific residual
    CNNs. Different frequency regions get different filters.

    Inputs:
        piano_roll:       [B, 3, T, 88]
        initial_log_stft: [B, F, T]

    Output:
        refined_log_stft: [B, F, T]
    """

    def __init__(
        self,
        n_freq_bins: int = 513,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        midi_low: int = 21,
        n_bands: int = 8,
        hidden_channels: int = 32,
        num_blocks_per_band: int = 3,
        dropout: float = 0.05,
        residual_scale: float = 0.2,
        condition_strength: float = 1.0,
        condition_channels: int = 3,
        use_condition: bool = True,
    ) -> None:
        super().__init__()

        if n_bands <= 0:
            raise ValueError(f"n_bands must be positive, got {n_bands}")
        if n_bands > n_freq_bins:
            raise ValueError(f"n_bands={n_bands} cannot exceed n_freq_bins={n_freq_bins}")

        self.n_freq_bins = int(n_freq_bins)
        self.n_bands = int(n_bands)
        self.residual_scale = float(residual_scale)
        self.use_condition = bool(use_condition)
        self.condition_channels = int(condition_channels)

        self.band_slices = self._make_band_slices(self.n_freq_bins, self.n_bands)

        if self.use_condition:
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
            in_channels = 1 + self.condition_channels  # initial prediction + projected condition channels
        else:
            self.conditioner = None
            in_channels = 1

        self.band_refiners = nn.ModuleList(
            [
                BandTextureRefiner(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    num_blocks=num_blocks_per_band,
                    dropout=dropout,
                )
                for _ in self.band_slices
            ]
        )

    @staticmethod
    def _make_band_slices(n_freq_bins: int, n_bands: int) -> List[Tuple[int, int]]:
        edges = torch.linspace(0, n_freq_bins, steps=n_bands + 1).round().long().tolist()
        slices: List[Tuple[int, int]] = []

        for i in range(n_bands):
            start = int(edges[i])
            end = int(edges[i + 1])

            if end <= start:
                end = start + 1

            slices.append((start, min(end, n_freq_bins)))

        # Make sure the final band ends exactly at n_freq_bins.
        slices[-1] = (slices[-1][0], n_freq_bins)
        return slices

    def forward(
        self,
        piano_roll: torch.Tensor,
        initial_log_stft: torch.Tensor,
    ) -> torch.Tensor:
        if initial_log_stft.ndim != 3:
            raise ValueError(
                f"Expected initial_log_stft [B, F, T], got {initial_log_stft.shape}"
            )

        if initial_log_stft.shape[1] != self.n_freq_bins:
            raise ValueError(
                f"Expected {self.n_freq_bins} frequency bins, got {initial_log_stft.shape[1]}"
            )

        initial = initial_log_stft.unsqueeze(1)  # [B, 1, F, T]

        if self.use_condition:
            assert self.conditioner is not None
            cond = self.conditioner(piano_roll)  # [B, C, F, T]
            if cond.shape[1] != self.condition_channels:
                raise ValueError(
                    f"Expected {self.condition_channels} condition channels, got {cond.shape[1]}"
                )
            x = torch.cat([initial, cond], dim=1)
        else:
            x = initial

        corrections = []

        for (start, end), refiner in zip(self.band_slices, self.band_refiners):
            band_x = x[:, :, start:end, :]
            band_corr = refiner(band_x)
            corrections.append(band_corr)

        correction = torch.cat(corrections, dim=2)  # [B, 1, F, T]

        refined = initial + self.residual_scale * correction
        return refined.squeeze(1)
