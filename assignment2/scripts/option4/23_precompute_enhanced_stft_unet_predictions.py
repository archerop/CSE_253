from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.shared.config import SAMPLE_RATE, MIDI_LOW
from app.option4.audio_models import LinearProjectedStftUNet
from app.option4.stft_cached_dataset import (
    CachedOption4StftDataset,
    option4_stft_cache_dir,
)
from app.option4.stft_refinement_dataset import option4_stft_prediction_cache_dir
from app.option4.performance_state_features import derive_performance_state_features


def safe_load(path: str | Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_splits(split: str) -> list[str]:
    if split == "all":
        return ["train", "validation", "test"]
    return [split]


def dtype_from_name(name: str):
    if name == "float32":
        return np.float32
    if name == "float16":
        return np.float16
    raise ValueError(f"Unsupported dtype: {name}")


def load_enhanced_unet(
    checkpoint_path: str | Path,
    n_freq_bins: int,
    n_fft: int,
    device: torch.device,
) -> LinearProjectedStftUNet:
    ckpt = safe_load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})

    model = LinearProjectedStftUNet(
        input_channels=6,
        n_freq_bins=n_freq_bins,
        sample_rate=SAMPLE_RATE,
        n_fft=n_fft,
        midi_low=MIDI_LOW,
        base_channels=int(ckpt_args.get("base_channels", 24)),
        blocks_per_level=int(ckpt_args.get("blocks_per_level", 2)),
        refinement_blocks=int(ckpt_args.get("refinement_blocks", 2)),
        dropout=float(ckpt_args.get("dropout", 0.0)),
        condition_strength=float(ckpt_args.get("condition_strength", 1.0)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def precompute_one_split(
    subset_name: str,
    split: str,
    n_fft: int,
    checkpoint_path: str | Path,
    prediction_cache_name: str,
    batch_size: int,
    num_workers: int,
    storage_dtype_name: str,
    overwrite: bool,
    amp: bool,
    max_age_frames: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(amp and device.type == "cuda")

    stft_cache_dir = option4_stft_cache_dir(subset_name, split, n_fft)
    dataset = CachedOption4StftDataset(stft_cache_dir)

    n = len(dataset)
    first = dataset[0]
    target_shape = tuple(first["target"].shape)
    n_freq_bins = int(target_shape[0])

    pred_cache_dir = option4_stft_prediction_cache_dir(
        subset_name=subset_name,
        split=split,
        n_fft=n_fft,
        prediction_cache_name=prediction_cache_name,
    )

    if pred_cache_dir.exists():
        if overwrite:
            shutil.rmtree(pred_cache_dir)
        else:
            raise FileExistsError(f"Prediction cache already exists: {pred_cache_dir}")

    pred_cache_dir.mkdir(parents=True, exist_ok=True)

    model = load_enhanced_unet(
        checkpoint_path=checkpoint_path,
        n_freq_bins=n_freq_bins,
        n_fft=n_fft,
        device=device,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    storage_dtype = dtype_from_name(storage_dtype_name)

    pred_path = pred_cache_dir / "initial_pred_log_stft.npy"
    pred_mm = np.lib.format.open_memmap(
        pred_path,
        mode="w+",
        dtype=storage_dtype,
        shape=(n, *target_shape),
    )

    print("=" * 80)
    print("Precompute enhanced STFT U-Net predictions")
    print("=" * 80)
    print(f"subset/split:          {subset_name}/{split}")
    print(f"n_fft:                 {n_fft}")
    print(f"num_examples:          {n}")
    print(f"target_shape:          {target_shape}")
    print(f"checkpoint:            {checkpoint_path}")
    print(f"prediction_cache_name: {prediction_cache_name}")
    print(f"output_dir:            {pred_cache_dir}")
    print(f"device:                {device}")
    print(f"use_amp:               {use_amp}")
    print(f"storage_dtype:         {storage_dtype_name}")

    t0 = time.time()
    offset = 0

    for batch in tqdm(loader, desc=f"precompute enhanced {subset_name}/{split}"):
        piano_roll = batch["piano_roll"].to(device, non_blocking=True)
        enhanced = derive_performance_state_features(
            piano_roll,
            max_age_frames=max_age_frames,
        )

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(enhanced)

        pred_np = pred.detach().cpu().numpy().astype(storage_dtype, copy=False)
        b = pred_np.shape[0]

        pred_mm[offset : offset + b] = pred_np
        offset += b

    pred_mm.flush()

    if offset != n:
        raise RuntimeError(f"Expected {n} predictions, wrote {offset}")

    config = {
        "subset_name": subset_name,
        "split": split,
        "n_fft": n_fft,
        "num_examples": n,
        "target_shape": list(target_shape),
        "checkpoint_path": str(checkpoint_path),
        "prediction_cache_name": prediction_cache_name,
        "feature_mode": "enhanced",
        "max_age_frames": max_age_frames,
        "storage_dtype": storage_dtype_name,
        "elapsed_seconds": time.time() - t0,
        "created_seconds": time.time(),
    }

    config_path = pred_cache_dir / "prediction_cache_config.json"
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)

    print()
    print(f"Saved predictions: {pred_path}")
    print(f"Saved config:      {config_path}")
    print(f"Elapsed:           {time.time() - t0:.1f}s")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute enhanced STFT U-Net predictions for TextureNet-lite."
    )

    parser.add_argument("--subset-name", type=str, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "validation", "test", "all"],
    )
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--prediction-cache-name",
        type=str,
        default="enhanced_stft_unet_small_best",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--storage-dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
    )
    parser.add_argument("--max-age-frames", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()

    for split in parse_splits(args.split):
        precompute_one_split(
            subset_name=args.subset_name,
            split=split,
            n_fft=args.n_fft,
            checkpoint_path=args.checkpoint,
            prediction_cache_name=args.prediction_cache_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            storage_dtype_name=args.storage_dtype,
            overwrite=args.overwrite,
            amp=args.amp,
            max_age_frames=args.max_age_frames,
        )


if __name__ == "__main__":
    main()
