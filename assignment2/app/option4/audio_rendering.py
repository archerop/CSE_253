from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch


def to_numpy(x: Any) -> np.ndarray:
    """
    Convert torch.Tensor / numpy array / list-like object to numpy array.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def logmel_to_mel_power(
    log_mel: Any,
    clamp_min: float = 0.0,
) -> np.ndarray:
    """
    Convert log1p(mel_power) back to mel_power.

    This assumes the preprocessing target was computed as:
        log_mel = log1p(mel_power)

    We clamp negative predictions before expm1 because mel power must be
    non-negative. Model outputs are not constrained to be positive during
    training.
    """
    log_mel_np = to_numpy(log_mel).astype(np.float32)

    if log_mel_np.ndim != 2:
        raise ValueError(f"Expected log_mel shape [n_mels, frames], got {log_mel_np.shape}")

    log_mel_np = np.maximum(log_mel_np, clamp_min)
    mel_power = np.expm1(log_mel_np)
    mel_power = np.maximum(mel_power, 0.0)

    return mel_power.astype(np.float32)


def trim_or_pad_audio(
    audio: np.ndarray,
    target_num_samples: int,
) -> np.ndarray:
    """
    Force audio to exactly target_num_samples.
    """
    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) > target_num_samples:
        return audio[:target_num_samples]

    if len(audio) < target_num_samples:
        pad = target_num_samples - len(audio)
        return np.pad(audio, (0, pad), mode="constant")

    return audio


def peak_normalize(
    audio: np.ndarray,
    peak: float = 0.95,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Peak-normalize audio for safe listening/export.

    This is for audio examples. It should not be interpreted as preserving
    model-predicted loudness.
    """
    audio = np.asarray(audio, dtype=np.float32)

    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0

    if max_abs < eps:
        return audio

    return (audio / max_abs * peak).astype(np.float32)


def logmel_to_audio_griffinlim(
    log_mel: Any,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    fmin: float,
    fmax: float,
    n_iter: int = 64,
    target_num_samples: int | None = None,
    normalize: bool = True,
    peak: float = 0.95,
) -> np.ndarray:
    """
    Convert predicted/target log-mel spectrogram to waveform.

    Pipeline:
        log-mel
        -> mel power
        -> librosa.feature.inverse.mel_to_audio
        -> Griffin-Lim phase reconstruction internally
        -> waveform

    This is the first-version deterministic renderer.
    """
    mel_power = logmel_to_mel_power(log_mel)

    if mel_power.shape[0] != n_mels:
        raise ValueError(
            f"Expected {n_mels} mel bins, got {mel_power.shape[0]}"
        )

    audio = librosa.feature.inverse.mel_to_audio(
        M=mel_power,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax,
        n_iter=n_iter,
        power=2.0,
    ).astype(np.float32)

    if target_num_samples is not None:
        audio = trim_or_pad_audio(audio, target_num_samples)

    if normalize:
        audio = peak_normalize(audio, peak=peak)

    return audio.astype(np.float32)


def save_wav(
    path: str | Path,
    audio: np.ndarray,
    sample_rate: int,
) -> None:
    """
    Save waveform as WAV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float32), sample_rate)


def maybe_save_mp3_from_wav(
    wav_path: str | Path,
    mp3_path: str | Path | None = None,
) -> Path | None:
    """
    Convert WAV to MP3 using ffmpeg if available.

    Returns the MP3 path if conversion succeeds. Returns None if ffmpeg is not
    installed.
    """
    wav_path = Path(wav_path)

    if mp3_path is None:
        mp3_path = wav_path.with_suffix(".mp3")
    else:
        mp3_path = Path(mp3_path)

    if shutil.which("ffmpeg") is None:
        print(f"[warn] ffmpeg not found; skip mp3 export for {wav_path}")
        return None

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(wav_path),
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "2",
        str(mp3_path),
    ]

    subprocess.run(cmd, check=True)
    return mp3_path


def save_audio_pair(
    wav_path: str | Path,
    audio: np.ndarray,
    sample_rate: int,
    make_mp3: bool = False,
) -> dict[str, str | None]:
    """
    Save WAV and optionally MP3.
    """
    wav_path = Path(wav_path)
    save_wav(wav_path, audio, sample_rate)

    mp3_path = None
    if make_mp3:
        maybe_path = maybe_save_mp3_from_wav(wav_path)
        mp3_path = str(maybe_path) if maybe_path is not None else None

    return {
        "wav": str(wav_path),
        "mp3": mp3_path,
    }



def audio_to_stft_magnitude(
    audio: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    center: bool = True,
) -> np.ndarray:
    """
    Compute linear STFT magnitude from waveform.

    This is used for oracle rendering ablation:
        ground-truth audio -> STFT magnitude -> Griffin-Lim audio

    Shape:
        [frequency_bins, frames]
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
    return magnitude


def stft_magnitude_to_audio_griffinlim(
    magnitude: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    center: bool = True,
    n_iter: int = 64,
    target_num_samples: int | None = None,
    normalize: bool = True,
    peak: float = 0.95,
) -> np.ndarray:
    """
    Reconstruct waveform from linear STFT magnitude with Griffin-Lim.

    This is less lossy than mel_to_audio because it skips the mel-to-STFT
    pseudo-inversion step.
    """
    magnitude = np.asarray(magnitude, dtype=np.float32)

    if magnitude.ndim != 2:
        raise ValueError(
            f"Expected STFT magnitude shape [frequency_bins, frames], got {magnitude.shape}"
        )

    audio = librosa.griffinlim(
        S=magnitude,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        center=center,
        length=target_num_samples,
    ).astype(np.float32)

    if target_num_samples is not None:
        audio = trim_or_pad_audio(audio, target_num_samples)

    if normalize:
        audio = peak_normalize(audio, peak=peak)

    return audio.astype(np.float32)
