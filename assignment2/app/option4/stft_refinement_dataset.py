from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from app.shared.config import OPTION4_CACHE_DIR
from app.option4.stft_cached_dataset import CachedOption4StftDataset


def option4_stft_prediction_cache_dir(
    subset_name: str,
    split: str,
    n_fft: int,
    prediction_cache_name: str,
) -> Path:
    return (
        OPTION4_CACHE_DIR
        / "stft_unet_predictions"
        / f"{prediction_cache_name}_{subset_name}_{split}_nfft{n_fft}"
    )


class CachedStftRefinementDataset(Dataset):
    """
    Dataset for training TextureNet-lite refinement.

    It combines:
      - original STFT cached dataset:
          piano_roll.npy
          log_stft_mag.npy
      - precomputed U-Net prediction:
          initial_pred_log_stft.npy
    """

    def __init__(
        self,
        stft_cache_dir: str | Path,
        prediction_cache_dir: str | Path,
    ) -> None:
        self.base_dataset = CachedOption4StftDataset(stft_cache_dir)
        self.prediction_cache_dir = Path(prediction_cache_dir)

        self.initial_pred_path = self.prediction_cache_dir / "initial_pred_log_stft.npy"
        self.prediction_config_path = self.prediction_cache_dir / "prediction_cache_config.json"

        if not self.initial_pred_path.exists():
            raise FileNotFoundError(f"Missing initial predictions: {self.initial_pred_path}")
        if not self.prediction_config_path.exists():
            raise FileNotFoundError(f"Missing prediction config: {self.prediction_config_path}")

        with self.prediction_config_path.open("r") as f:
            self.prediction_config = json.load(f)

        self.metadata = self.base_dataset.metadata
        self.cache_config = self.base_dataset.cache_config

        self._initial_pred = None

        expected_n = len(self.base_dataset)
        config_n = int(self.prediction_config["num_examples"])

        if config_n != expected_n:
            raise ValueError(
                f"Prediction cache size mismatch: prediction={config_n}, base={expected_n}"
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_initial_pred"] = None
        return state

    def _ensure_pred_open(self) -> None:
        if self._initial_pred is None:
            self._initial_pred = np.load(self.initial_pred_path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        self._ensure_pred_open()

        base = self.base_dataset[idx]
        initial_pred_np = np.array(self._initial_pred[idx], copy=True)

        base["initial_pred"] = torch.from_numpy(initial_pred_np).float()
        return base
