from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf


def _pad_or_crop_1d(audio: np.ndarray, target_samples: int) -> np.ndarray:
    """
    Ensure a 1D audio array has exactly target_samples samples.
    """
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio with shape (samples,), got {audio.shape}")

    if len(audio) < target_samples:
        pad_width = target_samples - len(audio)
        audio = np.pad(audio, (0, pad_width), mode="constant")
    elif len(audio) > target_samples:
        audio = audio[:target_samples]

    return audio.astype(np.float32)


def _pad_or_crop_2d_time(matrix: np.ndarray, target_frames: int) -> np.ndarray:
    """
    Ensure a 2D time-frequency matrix has exactly target_frames columns.

    Expected input shape:
        frequency_bins × time_frames
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {matrix.shape}")

    current_frames = matrix.shape[1]

    if current_frames < target_frames:
        pad_width = target_frames - current_frames
        matrix = np.pad(matrix, ((0, 0), (0, pad_width)), mode="constant")
    elif current_frames > target_frames:
        matrix = matrix[:, :target_frames]

    return matrix.astype(np.float32)


def load_audio_window(
    audio_path: str | Path,
    start_sec: float,
    clip_seconds: float,
    sample_rate: int,
) -> Tuple[np.ndarray, Dict[str, float | int | str]]:
    """
    Load an audio window from a MAESTRO audio file.

    Steps:
    - Read only the requested window from disk using soundfile.
    - Convert stereo to mono.
    - Resample to sample_rate if needed.
    - Pad/crop to exactly clip_seconds * sample_rate samples.

    Returns:
        audio: mono float32 waveform with shape (target_samples,)
        metadata: dictionary with loading information
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if start_sec < 0:
        raise ValueError(f"start_sec must be non-negative, got {start_sec}")

    if clip_seconds <= 0:
        raise ValueError(f"clip_seconds must be positive, got {clip_seconds}")

    info = sf.info(str(audio_path))
    native_sample_rate = int(info.samplerate)
    native_channels = int(info.channels)
    native_total_frames = int(info.frames)
    native_duration = native_total_frames / native_sample_rate

    start_frame = int(round(start_sec * native_sample_rate))
    requested_frames = int(round(clip_seconds * native_sample_rate))

    if start_frame >= native_total_frames:
        # Return silence if the requested window starts beyond the file.
        raw = np.zeros((0, native_channels), dtype=np.float32)
    else:
        frames_to_read = min(requested_frames, native_total_frames - start_frame)
        raw, _ = sf.read(
            str(audio_path),
            start=start_frame,
            frames=frames_to_read,
            dtype="float32",
            always_2d=True,
        )

    # Convert to mono.
    if raw.size == 0:
        mono_native = np.zeros((0,), dtype=np.float32)
    else:
        mono_native = raw.mean(axis=1).astype(np.float32)

    # Resample if needed.
    if native_sample_rate != sample_rate and len(mono_native) > 0:
        audio = librosa.resample(
            mono_native,
            orig_sr=native_sample_rate,
            target_sr=sample_rate,
        ).astype(np.float32)
    else:
        audio = mono_native.astype(np.float32)

    target_samples = int(round(clip_seconds * sample_rate))
    audio = _pad_or_crop_1d(audio, target_samples)

    metadata: Dict[str, float | int | str] = {
        "audio_path": str(audio_path),
        "native_sample_rate": native_sample_rate,
        "target_sample_rate": int(sample_rate),
        "native_channels": native_channels,
        "native_total_frames": native_total_frames,
        "native_duration": float(native_duration),
        "start_sec": float(start_sec),
        "clip_seconds": float(clip_seconds),
        "target_samples": int(target_samples),
        "loaded_samples": int(len(audio)),
    }

    return audio, metadata


def audio_to_logmel(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    fmin: float,
    fmax: Optional[float],
    center: bool,
    expected_frames: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, float | int | str]]:
    """
    Convert mono waveform to log-mel spectrogram.

    Output shape:
        n_mels × time_frames

    We use log(1 + mel_power) as a stable regression target.
    """
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio with shape (samples,), got {audio.shape}")

    if not np.isfinite(audio).all():
        raise ValueError("Audio contains NaN or Inf.")

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=center,
        power=2.0,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    ).astype(np.float32)

    log_mel = np.log1p(mel).astype(np.float32)

    if expected_frames is not None:
        log_mel = _pad_or_crop_2d_time(log_mel, expected_frames)

    metadata: Dict[str, float | int | str] = {
        "sample_rate": int(sample_rate),
        "n_fft": int(n_fft),
        "hop_length": int(hop_length),
        "win_length": int(win_length),
        "n_mels": int(n_mels),
        "fmin": float(fmin),
        "fmax": float(fmax) if fmax is not None else "None",
        "center": str(center),
        "num_samples": int(len(audio)),
        "num_frames": int(log_mel.shape[1]),
        "log_mel_min": float(log_mel.min()),
        "log_mel_mean": float(log_mel.mean()),
        "log_mel_max": float(log_mel.max()),
    }

    return log_mel, metadata


def load_audio_window_to_logmel(
    audio_path: str | Path,
    start_sec: float,
    clip_seconds: float,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    fmin: float,
    fmax: Optional[float],
    center: bool,
    expected_frames: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float | int | str]]:
    """
    Convenience wrapper:
        audio file window -> mono waveform -> log-mel spectrogram

    Returns:
        audio: mono waveform
        log_mel: log-mel spectrogram
        metadata: combined metadata
    """
    audio, audio_metadata = load_audio_window(
        audio_path=audio_path,
        start_sec=start_sec,
        clip_seconds=clip_seconds,
        sample_rate=sample_rate,
    )

    log_mel, mel_metadata = audio_to_logmel(
        audio=audio,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        center=center,
        expected_frames=expected_frames,
    )

    metadata = {
        **audio_metadata,
        **{f"mel_{k}": v for k, v in mel_metadata.items()},
    }

    return audio, log_mel, metadata


def validate_audio_and_logmel(
    audio: np.ndarray,
    log_mel: np.ndarray,
    n_mels: int,
    expected_frames: Optional[int] = None,
) -> None:
    """
    Basic sanity checks for audio and log-mel features.
    """
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio shape (samples,), got {audio.shape}")

    if log_mel.ndim != 2:
        raise ValueError(f"Expected log_mel shape (n_mels, frames), got {log_mel.shape}")

    if log_mel.shape[0] != n_mels:
        raise ValueError(f"Expected {n_mels} mel bins, got {log_mel.shape[0]}")

    if expected_frames is not None and log_mel.shape[1] != expected_frames:
        raise ValueError(
            f"Expected {expected_frames} frames, got {log_mel.shape[1]}"
        )

    if not np.isfinite(audio).all():
        raise ValueError("Audio contains NaN or Inf.")

    if not np.isfinite(log_mel).all():
        raise ValueError("log_mel contains NaN or Inf.")

    if log_mel.min() < 0:
        raise ValueError(f"log_mel should be non-negative with log1p, got min={log_mel.min()}")


def summarize_audio_and_logmel(
    audio: np.ndarray,
    log_mel: np.ndarray,
) -> Dict[str, float | int]:
    """
    Return simple summary statistics for sanity checking and reporting.
    """
    energy_per_frame = log_mel.sum(axis=0)

    stats: Dict[str, float | int] = {
        "audio_num_samples": int(len(audio)),
        "audio_min": float(audio.min()) if len(audio) else 0.0,
        "audio_mean": float(audio.mean()) if len(audio) else 0.0,
        "audio_max": float(audio.max()) if len(audio) else 0.0,
        "audio_rms": float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0,
        "log_mel_num_mels": int(log_mel.shape[0]),
        "log_mel_num_frames": int(log_mel.shape[1]),
        "log_mel_min": float(log_mel.min()),
        "log_mel_mean": float(log_mel.mean()),
        "log_mel_max": float(log_mel.max()),
        "energy_min": float(energy_per_frame.min()),
        "energy_mean": float(energy_per_frame.mean()),
        "energy_max": float(energy_per_frame.max()),
    }

    return stats


def save_logmel_npz(
    output_path: str | Path,
    audio: np.ndarray,
    log_mel: np.ndarray,
    metadata: Dict[str, float | int | str],
) -> None:
    """
    Save one audio/log-mel example to a compressed .npz file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        audio=audio.astype(np.float32),
        log_mel=log_mel.astype(np.float32),
        metadata_keys=np.array(list(metadata.keys())),
        metadata_values=np.array([str(v) for v in metadata.values()]),
    )
