"""
Build and cache the Option 2 window index for train and validation splits.

Run from assignment2/:
    python scripts/option2/01_build_option2_index.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.shared.config import OPTION2_CACHE_DIR
from app.option2.symbolic_dataset import build_window_index


def main() -> None:
    OPTION2_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Building Option 2 Window Index ===\n")

    for split in ("train", "validation", "test"):
        cache_path = OPTION2_CACHE_DIR / f"{split}_windows.pkl"
        windows = build_window_index(split=split, cache_path=cache_path)
        total_seconds = len(windows) * 4.0  # each window = prefix+cont = 8s, stride=2s
        print(f"  [{split:>12s}] {len(windows):>6d} windows  (cached → {cache_path.name})")

    print("\nDone.")


if __name__ == "__main__":
    main()
