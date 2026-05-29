from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from app.shared.config import OPTION4_CACHE_DIR


def option4_stft_cache_dir(subset_name: str, split: str, n_fft: int) -> Path:
    return OPTION4_CACHE_DIR / "precomputed_stft" / f"{subset_name}_{split}_nfft{n_fft}"


class CachedOption4StftDataset(Dataset):
    """
    Cached dataset for 513-bin STFT target experiments.

    Expected files:
        piano_roll.npy      [N, 3, T, 88]
        log_stft_mag.npy    [N, F, T]
        metadata.csv
        cache_config.json
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)

        self.piano_roll_path = self.cache_dir / "piano_roll.npy"
        self.log_stft_mag_path = self.cache_dir / "log_stft_mag.npy"
        self.metadata_path = self.cache_dir / "metadata.csv"
        self.config_path = self.cache_dir / "cache_config.json"

        missing = [
            p
            for p in [
                self.piano_roll_path,
                self.log_stft_mag_path,
                self.metadata_path,
                self.config_path,
            ]
            if not p.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Cached STFT dataset is incomplete. Missing:\n"
                + "\n".join(str(p) for p in missing)
            )

        self.metadata = pd.read_csv(self.metadata_path)

        with self.config_path.open("r") as f:
            self.cache_config = json.load(f)

        self._piano_roll = None
        self._log_stft_mag = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_piano_roll"] = None
        state["_log_stft_mag"] = None
        return state

    def _ensure_arrays_open(self) -> None:
        if self._piano_roll is None:
            self._piano_roll = np.load(self.piano_roll_path, mmap_mode="r")
        if self._log_stft_mag is None:
            self._log_stft_mag = np.load(self.log_stft_mag_path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        self._ensure_arrays_open()

        row = self.metadata.iloc[idx]

        piano_roll_np = np.array(self._piano_roll[idx], copy=True)
        log_stft_np = np.array(self._log_stft_mag[idx], copy=True)

        return {
            "piano_roll": torch.from_numpy(piano_roll_np).float(),
            "target": torch.from_numpy(log_stft_np).float(),
            "log_stft_mag": torch.from_numpy(log_stft_np).float(),
            "window_id": str(row["window_id"]),
            "piece_id": str(row["piece_id"]),
            "split": str(row["split"]),
            "composer": str(row.get("composer", "")),
            "title": str(row.get("title", "")),
            "start_sec": torch.tensor(float(row["start_sec"]), dtype=torch.float32),
            "clip_seconds": torch.tensor(float(row["clip_seconds"]), dtype=torch.float32),
        }


def make_cached_option4_stft_dataloader(
    cache_dir: str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    dataset = CachedOption4StftDataset(cache_dir=cache_dir)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
