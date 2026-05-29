from __future__ import annotations

from functools import lru_cache
from typing import Optional

import torch


def silence_baseline_like(target_log_mel: torch.Tensor) -> torch.Tensor:
    """
    Silence baseline for log-mel prediction.

    Since our target is log1p(mel_power), zero corresponds to no mel energy.
    """
    return torch.zeros_like(target_log_mel)


def midi_pitch_to_hz(midi_pitch: torch.Tensor) -> torch.Tensor:
    """
    Convert MIDI pitch to frequency in Hz.
    """
    return 440.0 * (2.0 ** ((midi_pitch - 69.0) / 12.0))


def hz_to_mel(freq_hz: torch.Tensor) -> torch.Tensor:
    """
    Convert Hz to mel scale using the common HTK formula.
    """
    return 2595.0 * torch.log10(1.0 + freq_hz / 700.0)


@lru_cache(maxsize=16)
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
    Build a pitch-to-mel-bin weight matrix on CPU.

    Shape:
        [n_pitches, n_mels]

    For each MIDI pitch, we activate a small Gaussian neighborhood around
    the fundamental and harmonic frequencies. This creates a crude symbolic
    note-to-spectral-energy mapping.

    This is a heuristic baseline, not a real piano synthesizer.
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

    mel_bin_positions = torch.arange(n_mels, dtype=torch.float32)
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

            gaussian = torch.exp(
                -0.5 * ((mel_bin_positions - mel_pos) / sigma_bins) ** 2
            )

            weights[pitch_idx] += harmonic_weight * gaussian

    # Normalize each pitch so that high-pitch notes are not unfairly small
    # just because they have fewer harmonics below fmax.
    row_sums = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    weights = weights / row_sums

    return weights


def heuristic_note_to_logmel_baseline(
    piano_roll: torch.Tensor,
    n_mels: int,
    midi_low: int,
    fmin: float,
    fmax: float,
    strength: float = 0.25,
    sigma_bins: float = 1.25,
    n_harmonics: int = 8,
    harmonic_decay: float = 1.0,
    onset_boost: float = 1.5,
    velocity_boost: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Heuristic MIDI-aware baseline.

    Input:
        piano_roll: [B, 3, T, 88]
            channel 0: active
            channel 1: onset
            channel 2: velocity_onset

    Output:
        pred_log_mel: [B, n_mels, T]

    This baseline maps active MIDI pitches to approximate mel-bin energy.
    It also boosts note-onset frames using onset and velocity channels.

    It is intentionally simple:
    - no learned weights
    - no real phase
    - no piano resonance model
    - no pedal model

    Its purpose is to test whether a learned model improves over a basic
    symbolic pitch-to-frequency mapping.
    """
    if piano_roll.ndim != 4:
        raise ValueError(
            f"Expected piano_roll shape [B, 3, T, 88], got {piano_roll.shape}"
        )

    batch_size, channels, frames, n_pitches = piano_roll.shape

    if channels < 3:
        raise ValueError(f"Expected at least 3 channels, got {channels}")

    device = piano_roll.device
    dtype = piano_roll.dtype

    active = piano_roll[:, 0]          # [B, T, P]
    onset = piano_roll[:, 1]           # [B, T, P]
    velocity_onset = piano_roll[:, 2]  # [B, T, P]

    weights = build_pitch_to_mel_weight_cpu(
        n_pitches=n_pitches,
        midi_low=midi_low,
        n_mels=n_mels,
        fmin=float(fmin),
        fmax=float(fmax),
        sigma_bins=float(sigma_bins),
        n_harmonics=int(n_harmonics),
        harmonic_decay=float(harmonic_decay),
    ).to(device=device, dtype=dtype)

    # Basic note energy comes from active notes.
    note_energy = active.clone()

    # Add an attack boost around onset frames.
    note_energy = note_energy + onset_boost * onset

    # Velocity affects attack strength.
    note_energy = note_energy + velocity_boost * velocity_onset

    # Map pitch axis to mel axis:
    #   note_energy [B, T, P]
    #   weights     [P, M]
    #   mel_energy  [B, T, M]
    mel_energy = torch.einsum("btp,pm->btm", note_energy, weights)

    # Move to [B, M, T] and apply log1p-like scaling.
    mel_energy = mel_energy.transpose(1, 2).contiguous()

    pred_log_mel = torch.log1p(strength * mel_energy.clamp_min(0.0) + eps)

    return pred_log_mel
