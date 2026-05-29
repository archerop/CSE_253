from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from app.shared.config import OPTION4_CACHE_DIR


def option4_cache_dir(subset_name: str, split: str) -> Path:
    """
    Standard cache directory for one Option 4 subset/split.
    """
    return OPTION4_CACHE_DIR / "precomputed" / f"{subset_name}_{split}"


class CachedOption4MidiToAudioDataset(Dataset):
    """
    Cached version of Option4MidiToAudioDataset.

    Expected files under cache_dir:
        piano_roll.npy
        log_mel.npy
        metadata.csv
        cache_config.json

    Returns the same main keys as Option4MidiToAudioDataset:
        piano_roll: FloatTensor [3, T, 88]
        log_mel:    FloatTensor [80, T]
        window_id, piece_id, split, composer, title, start_sec, clip_seconds

    Arrays are memory-mapped, so loading the dataset does not read the whole
    cache into RAM.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)

        self.piano_roll_path = self.cache_dir / "piano_roll.npy"
        self.log_mel_path = self.cache_dir / "log_mel.npy"
        self.metadata_path = self.cache_dir / "metadata.csv"
        self.config_path = self.cache_dir / "cache_config.json"

        missing = [
            path
            for path in [
                self.piano_roll_path,
                self.log_mel_path,
                self.metadata_path,
                self.config_path,
            ]
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Cached dataset is incomplete. Missing files:\n"
                + "\n".join(str(p) for p in missing)
            )

        self.metadata = pd.read_csv(self.metadata_path)

        with self.config_path.open("r") as f:
            self.cache_config = json.load(f)

        self._piano_roll = None
        self._log_mel = None

    def __getstate__(self):
        """
        Make DataLoader worker spawning safer by reopening memmaps per worker.
        """
        state = self.__dict__.copy()
        state["_piano_roll"] = None
        state["_log_mel"] = None
        return state

    def _ensure_arrays_open(self) -> None:
        if self._piano_roll is None:
            self._piano_roll = np.load(self.piano_roll_path, mmap_mode="r")
        if self._log_mel is None:
            self._log_mel = np.load(self.log_mel_path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        self._ensure_arrays_open()

        row = self.metadata.iloc[idx]

        # np.array(..., copy=True) avoids non-writable memmap warnings from torch.
        piano_roll_np = np.array(self._piano_roll[idx], copy=True)
        log_mel_np = np.array(self._log_mel[idx], copy=True)

        item: Dict[str, Any] = {
            "piano_roll": torch.from_numpy(piano_roll_np).float(),
            "log_mel": torch.from_numpy(log_mel_np).float(),
            "window_id": str(row["window_id"]),
            "piece_id": str(row["piece_id"]),
            "split": str(row["split"]),
            "composer": str(row.get("composer", "")),
            "title": str(row.get("title", "")),
            "start_sec": torch.tensor(float(row["start_sec"]), dtype=torch.float32),
            "clip_seconds": torch.tensor(float(row["clip_seconds"]), dtype=torch.float32),
        }

        return item


def make_cached_option4_dataloader(
    cache_dir: str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    dataset = CachedOption4MidiToAudioDataset(cache_dir=cache_dir)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
