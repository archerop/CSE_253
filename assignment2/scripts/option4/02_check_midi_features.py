from __future__ import annotations

from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.shared.config import (
    FRAME_RATE,
    MIDI_LOW,
    MIDI_HIGH,
    CLIP_SECONDS,
    ONSET_WIDTH_FRAMES,
    METADATA_CACHE_DIR,
    FIGURE_DIR,
    CACHE_DIR,
)
from app.option4.midi_features import (
    CHANNEL_NAMES,
    midi_to_pianoroll_features,
    save_pianoroll_npz,
    summarize_pianoroll_features,
    validate_pianoroll_features,
)


def select_metadata_row(
    metadata: pd.DataFrame,
    split: str,
    row_index: int,
    start_sec: float,
    clip_seconds: float,
) -> pd.Series:
    """
    Select a row from the resolved metadata table.

    We filter by split and require the piece to be long enough for the requested window.
    """
    required_duration = start_sec + clip_seconds

    subset = metadata[
        (metadata["split"] == split) & (metadata["duration"] >= required_duration)
    ].copy()

    if len(subset) == 0:
        raise ValueError(
            f"No pieces found for split={split!r} with duration >= {required_duration:.2f}s"
        )

    if row_index < 0 or row_index >= len(subset):
        raise IndexError(
            f"row_index={row_index} out of range for subset of size {len(subset)}"
        )

    return subset.iloc[row_index]


def plot_pianoroll_features(
    features: np.ndarray,
    output_path: Path,
    start_sec: float,
    frame_rate: float,
    midi_low: int,
    title: str,
) -> None:
    """
    Plot active, onset, and velocity-onset channels.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    channels, num_frames, num_pitches = features.shape
    duration = num_frames / frame_rate

    extent = [
        start_sec,
        start_sec + duration,
        midi_low,
        midi_low + num_pitches - 1,
    ]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(14, 8),
        sharex=True,
        constrained_layout=True,
    )

    for i, channel_name in enumerate(CHANNEL_NAMES):
        ax = axes[i]
        # imshow expects [height, width], so transpose pitch/time.
        image = features[i].T

        ax.imshow(
            image,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=extent,
        )
        ax.set_ylabel("MIDI pitch")
        ax.set_title(channel_name)

    axes[-1].set_xlabel("time (seconds)")
    fig.suptitle(title, fontsize=14)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2 sanity check: convert one MAESTRO MIDI window to piano-roll features."
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--start-sec", type=float, default=30.0)
    parser.add_argument("--seconds", type=float, default=CLIP_SECONDS)
    parser.add_argument("--onset-width-frames", type=int, default=ONSET_WIDTH_FRAMES)
    args = parser.parse_args()

    metadata_path = METADATA_CACHE_DIR / "maestro_resolved_metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Resolved metadata not found: {metadata_path}\n"
            "Run Step 1 first: python scripts/01_check_maestro_metadata.py"
        )

    metadata = pd.read_csv(metadata_path)

    row = select_metadata_row(
        metadata=metadata,
        split=args.split,
        row_index=args.row_index,
        start_sec=args.start_sec,
        clip_seconds=args.seconds,
    )

    print("=" * 80)
    print("Step 2: MIDI preprocessing sanity check")
    print("=" * 80)
    print(f"piece_id:  {row['piece_id']}")
    print(f"split:     {row['split']}")
    print(f"composer:  {row['composer']}")
    print(f"title:     {row['title']}")
    print(f"duration:  {row['duration']:.2f} sec")
    print(f"midi_path: {row['midi_path']}")
    print()

    features, feature_metadata = midi_to_pianoroll_features(
        midi_path=row["midi_path"],
        frame_rate=FRAME_RATE,
        midi_low=MIDI_LOW,
        midi_high=MIDI_HIGH,
        start_sec=args.start_sec,
        clip_seconds=args.seconds,
        onset_width_frames=args.onset_width_frames,
    )

    validate_pianoroll_features(features)
    stats = summarize_pianoroll_features(features)

    print("Feature tensor:")
    print(f"shape: {features.shape}  # (channels, frames, pitches)")
    print(f"frame_rate: {FRAME_RATE:.4f} frames/sec")
    print(f"window: {args.start_sec:.2f}s to {args.start_sec + args.seconds:.2f}s")
    print()

    print("Feature metadata:")
    for key, value in feature_metadata.items():
        print(f"{key}: {value}")
    print()

    print("Feature statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print()

    sample_dir = CACHE_DIR / "midi_features"
    sample_npz = sample_dir / "sample_midi_features.npz"

    combined_metadata = {
        **feature_metadata,
        "piece_id": row["piece_id"],
        "split": row["split"],
        "composer": row["composer"],
        "title": row["title"],
    }

    save_pianoroll_npz(
        output_path=sample_npz,
        features=features,
        metadata=combined_metadata,
    )

    print(f"Saved sample feature tensor to: {sample_npz}")

    figure_path = FIGURE_DIR / "step2_pianoroll_example.png"
    figure_title = (
        f"Step 2 MIDI preprocessing example\n"
        f"{row['composer']} — {row['title']} "
        f"({args.start_sec:.1f}s–{args.start_sec + args.seconds:.1f}s)"
    )

    plot_pianoroll_features(
        features=features,
        output_path=figure_path,
        start_sec=args.start_sec,
        frame_rate=FRAME_RATE,
        midi_low=MIDI_LOW,
        title=figure_title,
    )

    print(f"Saved piano-roll figure to: {figure_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
