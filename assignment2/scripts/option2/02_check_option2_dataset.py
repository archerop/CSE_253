"""
Validate the Option 2 dataset by loading a small batch and inspecting shapes.

Run from assignment2/:
    python scripts/option2/02_check_option2_dataset.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader

from app.shared.config import OPTION2_CACHE_DIR, OPTION2_FRAME_RATE, OPTION2_PREFIX_SECONDS, OPTION2_CONTINUATION_SECONDS
from app.option2.symbolic_dataset import build_window_index, SymbolicDataset


def main() -> None:
    print("=== Option 2 Dataset Check ===\n")

    # Use cached index if available; otherwise build a small debug subset
    cache_path = OPTION2_CACHE_DIR / "train_windows.pkl"
    windows = build_window_index(split="train", max_windows=50, cache_path=None)

    print(f"Windows loaded: {len(windows)}")
    print(f"Frame rate:     {OPTION2_FRAME_RATE} fps")
    expected_prefix_len = int(OPTION2_PREFIX_SECONDS * OPTION2_FRAME_RATE)
    expected_cont_len = int(OPTION2_CONTINUATION_SECONDS * OPTION2_FRAME_RATE)
    print(f"Expected prefix frames:       {expected_prefix_len}")
    print(f"Expected continuation frames: {expected_cont_len}\n")

    dataset = SymbolicDataset(windows)
    loader = DataLoader(dataset, batch_size=4, num_workers=0)

    prefix, continuation = next(iter(loader))
    print(f"Batch shapes:")
    print(f"  prefix:       {tuple(prefix.shape)}  dtype={prefix.dtype}")
    print(f"  continuation: {tuple(continuation.shape)}  dtype={continuation.dtype}")

    print(f"\nPrefix   — active pitches/frame (mean): {prefix.sum(-1).mean():.2f}")
    print(f"Cont     — active pitches/frame (mean): {continuation.sum(-1).mean():.2f}")
    print(f"Prefix   — value range: [{prefix.min():.0f}, {prefix.max():.0f}]")
    print(f"Cont     — value range: [{continuation.min():.0f}, {continuation.max():.0f}]")

    print("\nDataset check passed.")


if __name__ == "__main__":
    main()
