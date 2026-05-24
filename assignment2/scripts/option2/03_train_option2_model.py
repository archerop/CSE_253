"""
Train the Option 2 symbolic Transformer.

Run from assignment2/:
    python scripts/option2/03_train_option2_model.py [--debug]

Pass --debug to train on a tiny subset (fast smoke test).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader

from app.shared.config import (
    CHECKPOINT_DIR,
    OPTION2_BATCH_SIZE,
    OPTION2_CACHE_DIR,
    OPTION2_OUTPUT_DIR,
)
from app.option2.symbolic_dataset import build_window_index, SymbolicDataset, precache_pianorolls
from app.option2.symbolic_models import SymbolicTransformer
from app.option2.symbolic_train import train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--debug",
        action="store_true",
        help="Use tiny dataset subset for a quick smoke test.",
    )
    p.add_argument("--epochs", type=int, default=None, help="Override max epochs.")
    p.add_argument("--batch-size", type=int, default=OPTION2_BATCH_SIZE)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    if args.debug:
        train_max, val_max = 200, 50
        print("DEBUG MODE: using 200 train / 50 val windows.\n")
    else:
        train_max, val_max = None, None
        print("Pre-caching piano-rolls (skips already-cached files)...")
        precache_pianorolls()

    print("Loading datasets...")
    OPTION2_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_windows = build_window_index(
        split="train",
        max_windows=train_max,
        cache_path=None if args.debug else OPTION2_CACHE_DIR / "train_windows.pkl",
    )
    val_windows = build_window_index(
        split="validation",
        max_windows=val_max,
        cache_path=None if args.debug else OPTION2_CACHE_DIR / "val_windows.pkl",
    )
    print(f"  train: {len(train_windows)} windows | val: {len(val_windows)} windows")

    train_ds = SymbolicDataset(train_windows)
    val_ds = SymbolicDataset(val_windows)

    batch_size = args.batch_size
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)

    model = SymbolicTransformer().to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    max_epochs = args.epochs or (3 if args.debug else None)
    ckpt_path = CHECKPOINT_DIR / "option2_best.pt"

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_path=ckpt_path,
        **({"max_epochs": max_epochs} if max_epochs else {}),
    )

    # Save loss history for plotting
    OPTION2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history_path = OPTION2_OUTPUT_DIR / "train_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    print(f"Best checkpoint saved to  {ckpt_path}")


if __name__ == "__main__":
    main()
