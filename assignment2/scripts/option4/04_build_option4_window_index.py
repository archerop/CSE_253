from __future__ import annotations

from pathlib import Path
import argparse
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.shared.config import (
    METADATA_CACHE_DIR,
    WINDOW_INDEX_CACHE_DIR,
    DEFAULT_CLIP_SECONDS,
    DEFAULT_STRIDE_SECONDS,
    DEFAULT_TRAIN_MAX_WINDOWS,
    DEFAULT_VAL_MAX_WINDOWS,
    DEFAULT_TEST_MAX_WINDOWS,
    DEFAULT_RANDOM_SEED,
)
from app.option4.window_index import (
    build_window_index_for_split,
    save_window_index,
    summarize_window_index,
)


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    if value.lower() in {"none", "null", "full"}:
        return None
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Option 4 aligned MIDI/audio window indices."
    )

    parser.add_argument("--subset-name", type=str, default="debug")
    parser.add_argument("--clip-seconds", type=float, default=DEFAULT_CLIP_SECONDS)
    parser.add_argument("--stride-seconds", type=float, default=DEFAULT_STRIDE_SECONDS)

    parser.add_argument(
        "--train-max-windows",
        type=_parse_optional_int,
        default=DEFAULT_TRAIN_MAX_WINDOWS,
        help="Max train windows. Use 'none' or 'full' for all windows.",
    )
    parser.add_argument(
        "--val-max-windows",
        type=_parse_optional_int,
        default=DEFAULT_VAL_MAX_WINDOWS,
        help="Max validation windows. Use 'none' or 'full' for all windows.",
    )
    parser.add_argument(
        "--test-max-windows",
        type=_parse_optional_int,
        default=DEFAULT_TEST_MAX_WINDOWS,
        help="Max test windows. Use 'none' or 'full' for all windows.",
    )

    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)

    args = parser.parse_args()

    metadata_path = METADATA_CACHE_DIR / "maestro_resolved_metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Resolved metadata not found: {metadata_path}\n"
            "Run Step 1 first: python scripts/shared/01_check_maestro_metadata.py"
        )

    metadata = pd.read_csv(metadata_path)

    split_specs = {
        "train": args.train_max_windows,
        "validation": args.val_max_windows,
        "test": args.test_max_windows,
    }

    print("=" * 80)
    print("Step 4: Build Option 4 window indices")
    print("=" * 80)
    print(f"metadata_path:  {metadata_path}")
    print(f"subset_name:    {args.subset_name}")
    print(f"clip_seconds:   {args.clip_seconds}")
    print(f"stride_seconds: {args.stride_seconds}")
    print(f"seed:           {args.seed}")
    print()

    WINDOW_INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for split, max_windows in split_specs.items():
        print("-" * 80)
        print(f"Building split: {split}")
        print(f"max_windows:    {max_windows}")

        index = build_window_index_for_split(
            metadata=metadata,
            split=split,
            clip_seconds=args.clip_seconds,
            stride_seconds=args.stride_seconds,
            max_windows=max_windows,
            seed=args.seed,
            drop_incomplete_last_window=True,
        )

        output_path = (
            WINDOW_INDEX_CACHE_DIR
            / f"option4_{args.subset_name}_{split}_windows.csv"
        )

        save_window_index(index, output_path)

        summary = summarize_window_index(index)

        print(f"Saved: {output_path}")
        print("Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        print()

    print("=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
