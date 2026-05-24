from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from app.shared.config import (
    SAMPLE_RATE,
    HOP_LENGTH,
    FRAME_RATE,
    MIDI_LOW,
    MIDI_HIGH,
    ONSET_WIDTH_FRAMES,
    N_FFT,
    WIN_LENGTH,
    N_MELS,
    FMIN,
    FMAX,
    CENTER,
)
from app.option4.midi_features import midi_to_pianoroll_features
from app.option4.audio_preprocessing import load_audio_window_to_logmel
from app.option4.window_index import load_window_index


class Option4MidiToAudioDataset(Dataset):
    """
    PyTorch Dataset for Option 4:
        MIDI-derived piano-roll features -> log-mel spectrogram

    Each item:
        piano_roll: FloatTensor [3, T, 88]
        log_mel:    FloatTensor [80, T]

    This dataset performs preprocessing on-the-fly to avoid storing all
    piano-roll/log-mel tensors on disk.
    """

    def __init__(
        self,
        index_csv: str | Path,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        frame_rate: float = FRAME_RATE,
        midi_low: int = MIDI_LOW,
        midi_high: int = MIDI_HIGH,
        onset_width_frames: int = ONSET_WIDTH_FRAMES,
        n_fft: int = N_FFT,
        win_length: int = WIN_LENGTH,
        n_mels: int = N_MELS,
        fmin: float = FMIN,
        fmax: Optional[float] = FMAX,
        center: bool = CENTER,
        return_audio: bool = False,
    ) -> None:
        self.index_csv = Path(index_csv)
        self.index = load_window_index(self.index_csv)

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_rate = frame_rate

        self.midi_low = midi_low
        self.midi_high = midi_high
        self.onset_width_frames = onset_width_frames

        self.n_fft = n_fft
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.center = center

        self.return_audio = return_audio

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.index.iloc[idx]

        start_sec = float(row["start_sec"])
        clip_seconds = float(row["clip_seconds"])

        expected_frames = int(np.ceil(clip_seconds * self.frame_rate))

        piano_roll, midi_metadata = midi_to_pianoroll_features(
            midi_path=row["midi_path"],
            frame_rate=self.frame_rate,
            midi_low=self.midi_low,
            midi_high=self.midi_high,
            start_sec=start_sec,
            clip_seconds=clip_seconds,
            onset_width_frames=self.onset_width_frames,
        )

        audio, log_mel, audio_metadata = load_audio_window_to_logmel(
            audio_path=row["audio_path"],
            start_sec=start_sec,
            clip_seconds=clip_seconds,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            center=self.center,
            expected_frames=expected_frames,
        )

        if piano_roll.shape[1] != log_mel.shape[1]:
            raise ValueError(
                f"Frame mismatch for window_id={row['window_id']}: "
                f"piano_roll frames={piano_roll.shape[1]}, "
                f"log_mel frames={log_mel.shape[1]}"
            )

        item: Dict[str, Any] = {
            "piano_roll": torch.from_numpy(piano_roll.astype(np.float32)),
            "log_mel": torch.from_numpy(log_mel.astype(np.float32)),
            "window_id": str(row["window_id"]),
            "piece_id": str(row["piece_id"]),
            "split": str(row["split"]),
            "composer": str(row.get("composer", "")),
            "title": str(row.get("title", "")),
            "start_sec": torch.tensor(start_sec, dtype=torch.float32),
            "clip_seconds": torch.tensor(clip_seconds, dtype=torch.float32),
        }

        if self.return_audio:
            item["audio"] = torch.from_numpy(audio.astype(np.float32))

        return item


def make_option4_dataloader(
    index_csv: str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    return_audio: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Convenience function for a standard DataLoader.
    """
    dataset = Option4MidiToAudioDataset(
        index_csv=index_csv,
        return_audio=return_audio,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
