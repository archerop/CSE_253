from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil
import sys
import time

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.shared.config import (
    WINDOW_INDEX_CACHE_DIR,
    SAMPLE_RATE,
    HOP_LENGTH,
    WIN_LENGTH,
    CENTER,
)
from app.option4.option4_dataset import Option4MidiToAudioDataset
from app.option4.stft_preprocessing import audio_to_log_stft_magnitude
from app.option4.stft_cached_dataset import option4_stft_cache_dir


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


def cache_one_split(
    subset_name: str,
    split: str,
    n_fft: int,
    win_length: int,
    batch_size: int,
    num_workers: int,
    storage_dtype_name: str,
    overwrite: bool,
) -> None:
    index_csv = WINDOW_INDEX_CACHE_DIR / f"option4_{subset_name}_{split}_windows.csv"

    if not index_csv.exists():
        raise FileNotFoundError(f"Window index not found: {index_csv}")

    cache_dir = option4_stft_cache_dir(subset_name, split, n_fft)

    if cache_dir.exists():
        if overwrite:
            shutil.rmtree(cache_dir)
        else:
            raise FileExistsError(f"Cache already exists: {cache_dir}")

    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Cache STFT tensors: subset={subset_name}, split={split}, n_fft={n_fft}")
    print("=" * 80)
    print(f"index_csv:      {index_csv}")
    print(f"cache_dir:      {cache_dir}")
    print(f"sample_rate:    {SAMPLE_RATE}")
    print(f"hop_length:     {HOP_LENGTH}")
    print(f"win_length:     {win_length}")
    print(f"center:         {CENTER}")
    print(f"storage_dtype:  {storage_dtype_name}")

    dataset = Option4MidiToAudioDataset(index_csv=index_csv, return_audio=True)
    n = len(dataset)

    if n == 0:
        raise ValueError(f"Empty dataset: {index_csv}")

    sample = dataset[0]
    piano_shape = tuple(sample["piano_roll"].shape)
    audio = sample["audio"].detach().cpu().numpy()
    expected_frames = sample["log_mel"].shape[-1]

    log_stft = audio_to_log_stft_magnitude(
        audio=audio,
        n_fft=n_fft,
        hop_length=HOP_LENGTH,
        win_length=win_length,
        center=CENTER,
        expected_frames=expected_frames,
    )
    stft_shape = tuple(log_stft.shape)

    storage_dtype = dtype_from_name(storage_dtype_name)

    piano_path = cache_dir / "piano_roll.npy"
    stft_path = cache_dir / "log_stft_mag.npy"
    metadata_path = cache_dir / "metadata.csv"
    config_path = cache_dir / "cache_config.json"

    piano_mm = np.lib.format.open_memmap(
        piano_path,
        mode="w+",
        dtype=storage_dtype,
        shape=(n, *piano_shape),
    )

    stft_mm = np.lib.format.open_memmap(
        stft_path,
        mode="w+",
        dtype=storage_dtype,
        shape=(n, *stft_shape),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    t0 = time.time()
    offset = 0

    for batch in tqdm(loader, desc=f"cache STFT {subset_name}/{split}"):
        piano = batch["piano_roll"].numpy().astype(storage_dtype, copy=False)
        audio_batch = batch["audio"].numpy()

        b = piano.shape[0]
        piano_mm[offset : offset + b] = piano

        for i in range(b):
            log_stft_i = audio_to_log_stft_magnitude(
                audio=audio_batch[i],
                n_fft=n_fft,
                hop_length=HOP_LENGTH,
                win_length=win_length,
                center=CENTER,
                expected_frames=expected_frames,
            )
            stft_mm[offset + i] = log_stft_i.astype(storage_dtype, copy=False)

        offset += b

    piano_mm.flush()
    stft_mm.flush()

    if offset != n:
        raise RuntimeError(f"Expected {n}, wrote {offset}")

    metadata = dataset.index.copy()
    metadata.to_csv(metadata_path, index=False)

    config = {
        "subset_name": subset_name,
        "split": split,
        "num_examples": int(n),
        "source_index_csv": str(index_csv),
        "sample_rate": SAMPLE_RATE,
        "n_fft": n_fft,
        "hop_length": HOP_LENGTH,
        "win_length": win_length,
        "center": CENTER,
        "expected_frames": int(expected_frames),
        "piano_roll_shape": [int(x) for x in piano_shape],
        "log_stft_mag_shape": [int(x) for x in stft_shape],
        "storage_dtype": storage_dtype_name,
        "created_seconds": time.time(),
        "elapsed_seconds": time.time() - t0,
    }

    with config_path.open("w") as f:
        json.dump(config, f, indent=2)

    total_bytes = piano_path.stat().st_size + stft_path.stat().st_size

    print()
    print(f"Cached {n} examples in {time.time() - t0:.1f}s")
    print(f"array size: {total_bytes / (1024 ** 3):.2f} GB")
    print(f"piano:      {piano_path}")
    print(f"stft:       {stft_path}")
    print(f"metadata:   {metadata_path}")
    print(f"config:     {config_path}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache Option 4 STFT targets.")

    parser.add_argument("--subset-name", type=str, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "validation", "test", "all"],
    )
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--storage-dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
    )
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    for split in parse_splits(args.split):
        cache_one_split(
            subset_name=args.subset_name,
            split=split,
            n_fft=args.n_fft,
            win_length=args.win_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            storage_dtype_name=args.storage_dtype,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
