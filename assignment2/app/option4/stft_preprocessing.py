from __future__ import annotations

from typing import Tuple

import librosa
import numpy as np


def audio_to_log_stft_magnitude(
    audio: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    center: bool = True,
    expected_frames: int | None = None,
) -> np.ndarray:
    """
    Convert waveform to log1p(STFT magnitude).

    Output shape:
        [n_fft // 2 + 1, frames]
    """
    audio = np.asarray(audio, dtype=np.float32)

    stft = librosa.stft(
        y=audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=center,
    )

    magnitude = np.abs(stft).astype(np.float32)
    log_mag = np.log1p(magnitude).astype(np.float32)

    if expected_frames is not None:
        log_mag = fix_num_frames(log_mag, expected_frames)

    return log_mag


def fix_num_frames(x: np.ndarray, expected_frames: int) -> np.ndarray:
    """
    Force spectrogram time frames to expected_frames.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram, got shape {x.shape}")

    current = x.shape[1]

    if current == expected_frames:
        return x

    if current > expected_frames:
        return x[:, :expected_frames]

    pad = expected_frames - current
    return np.pad(x, ((0, 0), (0, pad)), mode="constant")


def log_stft_magnitude_to_magnitude(
    log_stft_mag: np.ndarray,
    clamp_min: float = 0.0,
) -> np.ndarray:
    """
    Convert log1p(STFT magnitude) back to STFT magnitude.
    """
    x = np.asarray(log_stft_mag, dtype=np.float32)
    x = np.maximum(x, clamp_min)
    mag = np.expm1(x)
    return np.maximum(mag, 0.0).astype(np.float32)
