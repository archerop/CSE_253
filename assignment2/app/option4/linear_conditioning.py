from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn as nn


def midi_pitch_to_hz(pitch: torch.Tensor) -> torch.Tensor:
    return 440.0 * (2.0 ** ((pitch - 69.0) / 12.0))


@lru_cache(maxsize=32)
def build_pitch_to_linear_frequency_weight_cpu(
    n_pitches: int,
    midi_low: int,
    n_freq_bins: int,
    sample_rate: int,
    n_fft: int,
    sigma_bins: float,
    n_harmonics: int,
    harmonic_decay: float,
) -> torch.Tensor:
    """
    Build fixed harmonic pitch-to-linear-STFT-frequency projection.

    Shape:
        [n_pitches, n_freq_bins]

    This does not use audio targets. It only uses pitch frequencies and
    harmonic frequencies.
    """
    if n_freq_bins != n_fft // 2 + 1:
        raise ValueError(
            f"Expected n_freq_bins = n_fft//2+1, got {n_freq_bins=} {n_fft=}"
        )

    pitches = torch.arange(midi_low, midi_low + n_pitches, dtype=torch.float32)
    freq_bins = torch.arange(n_freq_bins, dtype=torch.float32)

    weights = torch.zeros((n_pitches, n_freq_bins), dtype=torch.float32)

    nyquist = float(sample_rate) / 2.0

    for pitch_idx, pitch in enumerate(pitches):
        fundamental = midi_pitch_to_hz(pitch)

        for harmonic in range(1, n_harmonics + 1):
            freq = fundamental * float(harmonic)

            if freq <= 0 or freq > nyquist:
                continue

            bin_pos = freq / nyquist * (n_freq_bins - 1)
            harmonic_weight = 1.0 / (float(harmonic) ** harmonic_decay)

            gaussian = torch.exp(-0.5 * ((freq_bins - bin_pos) / sigma_bins) ** 2)
            weights[pitch_idx] += harmonic_weight * gaussian

    row_sums = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    weights = weights / row_sums
    return weights


class PianoRollToLinearFrequencyCondition(nn.Module):
    """
    Fixed MIDI piano-roll to linear-frequency STFT conditioning.

    Input:
        piano_roll: [B, C, T, 88]

    Output:
        condition: [B, C, F, T]
        where F = n_fft // 2 + 1, e.g. 513 for n_fft=1024.
    """

    def __init__(
        self,
        n_pitches: int = 88,
        midi_low: int = 21,
        n_freq_bins: int = 513,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        sigma_bins: float = 1.25,
        n_harmonics: int = 8,
        harmonic_decay: float = 1.0,
        condition_strength: float = 1.0,
        log_scale: bool = True,
    ) -> None:
        super().__init__()

        weight = build_pitch_to_linear_frequency_weight_cpu(
            n_pitches=n_pitches,
            midi_low=midi_low,
            n_freq_bins=n_freq_bins,
            sample_rate=sample_rate,
            n_fft=n_fft,
            sigma_bins=sigma_bins,
            n_harmonics=n_harmonics,
            harmonic_decay=harmonic_decay,
        )

        self.register_buffer("pitch_to_freq_weight", weight, persistent=False)

        self.n_pitches = n_pitches
        self.n_freq_bins = n_freq_bins
        self.condition_strength = float(condition_strength)
        self.log_scale = bool(log_scale)

    def forward(self, piano_roll: torch.Tensor) -> torch.Tensor:
        if piano_roll.ndim != 4:
            raise ValueError(
                f"Expected piano_roll shape [B, C, T, P], got {piano_roll.shape}"
            )

        batch_size, channels, frames, n_pitches = piano_roll.shape

        if n_pitches != self.n_pitches:
            raise ValueError(f"Expected {self.n_pitches} pitches, got {n_pitches}")

        weight = self.pitch_to_freq_weight.to(
            device=piano_roll.device,
            dtype=piano_roll.dtype,
        )

        projected_channels = []

        for c in range(channels):
            x = piano_roll[:, c]  # [B, T, P]
            freq = torch.einsum("btp,pf->btf", x, weight)  # [B, T, F]
            freq = freq.transpose(1, 2).contiguous()       # [B, F, T]

            if self.log_scale:
                freq = torch.log1p(self.condition_strength * freq.clamp_min(0.0))
            else:
                freq = self.condition_strength * freq

            projected_channels.append(freq)

        return torch.stack(projected_channels, dim=1)  # [B, C, F, T]
