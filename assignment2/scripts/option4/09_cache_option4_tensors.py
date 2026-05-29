from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.shared.config import WINDOW_INDEX_CACHE_DIR
from app.option4.option4_dataset import Option4MidiToAudioDataset
from app.option4.cached_dataset import option4_cache_dir


def _parse_splits(split: str) -> list[str]:
    if split == "all":
        return ["train", "validation", "test"]
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split={split!r}")
    return [split]


def _dtype_from_name(name: str):
    if name == "float32":
        return np.float32
    if name == "float16":
        return np.float16
    raise ValueError(f"Unsupported storage dtype: {name}")


def cache_one_split(
    subset_name: str,
    split: str,
    batch_size: int,
    num_workers: int,
    storage_dtype_name: str,
    overwrite: bool,
) -> None:
    index_csv = WINDOW_INDEX_CACHE_DIR / f"option4_{subset_name}_{split}_windows.csv"

    if not index_csv.exists():
        raise FileNotFoundError(
            f"Window index not found: {index_csv}\n"
            "Build it first with scripts/option4/04_build_option4_window_index.py"
        )

    cache_dir = option4_cache_dir(subset_name, split)

    if cache_dir.exists():
        if overwrite:
            shutil.rmtree(cache_dir)
        else:
            raise FileExistsError(
                f"Cache already exists: {cache_dir}\n"
                "Use --overwrite to rebuild it."
            )

    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Caching Option 4 tensors: subset={subset_name}, split={split}")
    print("=" * 80)
    print(f"index_csv:      {index_csv}")
    print(f"cache_dir:      {cache_dir}")
    print(f"batch_size:     {batch_size}")
    print(f"num_workers:    {num_workers}")
    print(f"storage_dtype:  {storage_dtype_name}")

    dataset = Option4MidiToAudioDataset(index_csv=index_csv, return_audio=False)
    n = len(dataset)

    if n == 0:
        raise ValueError(f"Dataset is empty: {index_csv}")

    # Read one sample to determine shapes.
    sample = dataset[0]
    piano_shape = tuple(sample["piano_roll"].shape)
    logmel_shape = tuple(sample["log_mel"].shape)

    storage_dtype = _dtype_from_name(storage_dtype_name)

    piano_path = cache_dir / "piano_roll.npy"
    logmel_path = cache_dir / "log_mel.npy"
    metadata_path = cache_dir / "metadata.csv"
    config_path = cache_dir / "cache_config.json"

    piano_mm = np.lib.format.open_memmap(
        piano_path,
        mode="w+",
        dtype=storage_dtype,
        shape=(n, *piano_shape),
    )

    logmel_mm = np.lib.format.open_memmap(
        logmel_path,
        mode="w+",
        dtype=storage_dtype,
        shape=(n, *logmel_shape),
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

    for batch in tqdm(loader, desc=f"cache {subset_name}/{split}"):
        piano = batch["piano_roll"].numpy().astype(storage_dtype, copy=False)
        logmel = batch["log_mel"].numpy().astype(storage_dtype, copy=False)

        b = piano.shape[0]

        piano_mm[offset : offset + b] = piano
        logmel_mm[offset : offset + b] = logmel

        offset += b

    piano_mm.flush()
    logmel_mm.flush()

    if offset != n:
        raise RuntimeError(f"Expected to cache {n} samples, but wrote {offset}")

    metadata = dataset.index.copy()
    metadata.to_csv(metadata_path, index=False)

    config = {
        "subset_name": subset_name,
        "split": split,
        "source_index_csv": str(index_csv),
        "num_examples": int(n),
        "piano_roll_shape": [int(x) for x in piano_shape],
        "log_mel_shape": [int(x) for x in logmel_shape],
        "storage_dtype": storage_dtype_name,
        "piano_roll_path": str(piano_path),
        "log_mel_path": str(logmel_path),
        "metadata_path": str(metadata_path),
        "created_seconds": time.time(),
        "elapsed_seconds": time.time() - t0,
    }

    with config_path.open("w") as f:
        json.dump(config, f, indent=2)

    elapsed = time.time() - t0

    print()
    print(f"Cached {n} examples in {elapsed:.1f}s")
    print(f"piano_roll.npy: {piano_path}")
    print(f"log_mel.npy:    {logmel_path}")
    print(f"metadata.csv:   {metadata_path}")
    print(f"config:         {config_path}")

    total_bytes = piano_path.stat().st_size + logmel_path.stat().st_size
    print(f"array size:     {total_bytes / (1024 ** 3):.2f} GB")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute Option 4 piano_roll/log_mel tensors for faster training."
    )

    parser.add_argument("--subset-name", type=str, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "validation", "test", "all"],
    )
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

    for split in _parse_splits(args.split):
        cache_one_split(
            subset_name=args.subset_name,
            split=split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            storage_dtype_name=args.storage_dtype,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
