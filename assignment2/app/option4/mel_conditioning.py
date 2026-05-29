from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn as nn


def midi_pitch_to_hz(pitch: torch.Tensor) -> torch.Tensor:
    return 440.0 * (2.0 ** ((pitch - 69.0) / 12.0))


def hz_to_mel(freq_hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq_hz / 700.0)


@lru_cache(maxsize=32)
def build_pitch_to_mel_weight_cpu(
    n_pitches: int,
    midi_low: int,
    n_mels: int,
    fmin: float,
    fmax: float,
    sigma_bins: float,
    n_harmonics: int,
    harmonic_decay: float,
) -> torch.Tensor:
    """
    Build fixed pitch-to-mel harmonic projection.

    Shape:
        [n_pitches, n_mels]

    This does not use audio targets. It only encodes the basic acoustic fact
    that a MIDI pitch contributes energy near its fundamental and harmonics.
    """
    if n_pitches <= 0:
        raise ValueError(f"n_pitches must be positive, got {n_pitches}")
    if n_mels <= 0:
        raise ValueError(f"n_mels must be positive, got {n_mels}")
    if fmax <= fmin:
        raise ValueError(f"fmax must be greater than fmin, got {fmin=} {fmax=}")

    pitches = torch.arange(midi_low, midi_low + n_pitches, dtype=torch.float32)
    mel_min = hz_to_mel(torch.tensor(float(fmin)))
    mel_max = hz_to_mel(torch.tensor(float(fmax)))
    mel_bins = torch.arange(n_mels, dtype=torch.float32)

    weights = torch.zeros((n_pitches, n_mels), dtype=torch.float32)

    for pitch_idx, pitch in enumerate(pitches):
        fundamental = midi_pitch_to_hz(pitch)

        for harmonic in range(1, n_harmonics + 1):
            freq = fundamental * float(harmonic)
            if freq < fmin or freq > fmax:
                continue

            mel_value = hz_to_mel(freq)
            mel_pos = (mel_value - mel_min) / (mel_max - mel_min) * (n_mels - 1)

            harmonic_weight = 1.0 / (float(harmonic) ** harmonic_decay)
            gaussian = torch.exp(-0.5 * ((mel_bins - mel_pos) / sigma_bins) ** 2)
            weights[pitch_idx] += harmonic_weight * gaussian

    row_sums = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    weights = weights / row_sums
    return weights


class PianoRollToMelCondition(nn.Module):
    """
    Fixed MIDI piano-roll to mel-axis conditioning layer.

    Input:
        piano_roll: [B, C, T, 88]

    Output:
        mel_condition: [B, C, 80, T]

    Each input channel is projected independently:
        active notes      -> mel-axis active condition
        onsets            -> mel-axis onset condition
        velocity-onsets   -> mel-axis velocity condition

    This gives the downstream CNN/U-Net a target-aligned frequency axis.
    """

    def __init__(
        self,
        n_pitches: int = 88,
        midi_low: int = 21,
        n_mels: int = 80,
        fmin: float = 30.0,
        fmax: float = 11025.0,
        sigma_bins: float = 1.25,
        n_harmonics: int = 8,
        harmonic_decay: float = 1.0,
        condition_strength: float = 1.0,
        log_scale: bool = True,
    ) -> None:
        super().__init__()

        weight = build_pitch_to_mel_weight_cpu(
            n_pitches=n_pitches,
            midi_low=midi_low,
            n_mels=n_mels,
            fmin=float(fmin),
            fmax=float(fmax),
            sigma_bins=float(sigma_bins),
            n_harmonics=int(n_harmonics),
            harmonic_decay=float(harmonic_decay),
        )

        self.register_buffer("pitch_to_mel_weight", weight, persistent=False)

        self.n_pitches = n_pitches
        self.n_mels = n_mels
        self.condition_strength = float(condition_strength)
        self.log_scale = bool(log_scale)

    def forward(self, piano_roll: torch.Tensor) -> torch.Tensor:
        if piano_roll.ndim != 4:
            raise ValueError(
                f"Expected piano_roll shape [B, C, T, P], got {piano_roll.shape}"
            )

        batch_size, channels, frames, n_pitches = piano_roll.shape
        if n_pitches != self.n_pitches:
            raise ValueError(
                f"Expected {self.n_pitches} pitches, got {n_pitches}"
            )

        weight = self.pitch_to_mel_weight.to(
            device=piano_roll.device,
            dtype=piano_roll.dtype,
        )

        projected_channels = []

        for c in range(channels):
            x = piano_roll[:, c]  # [B, T, P]
            mel = torch.einsum("btp,pm->btm", x, weight)  # [B, T, M]
            mel = mel.transpose(1, 2).contiguous()        # [B, M, T]

            if self.log_scale:
                mel = torch.log1p(self.condition_strength * mel.clamp_min(0.0))
            else:
                mel = self.condition_strength * mel

            projected_channels.append(mel)

        return torch.stack(projected_channels, dim=1)  # [B, C, M, T]
