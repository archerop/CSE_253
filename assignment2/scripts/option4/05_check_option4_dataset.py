from __future__ import annotations

from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.shared.config import (
    FRAME_RATE,
    MIDI_LOW,
    WINDOW_INDEX_CACHE_DIR,
    OPTION4_OUTPUT_DIR,
)
from app.option4.option4_dataset import (
    Option4MidiToAudioDataset,
    make_option4_dataloader,
)


def plot_dataset_sample(
    piano_roll,
    log_mel,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # torch.Tensor -> numpy
    if hasattr(piano_roll, "detach"):
        piano_roll = piano_roll.detach().cpu().numpy()
    if hasattr(log_mel, "detach"):
        log_mel = log_mel.detach().cpu().numpy()

    channels, midi_frames, num_pitches = piano_roll.shape
    mel_bins, mel_frames = log_mel.shape

    duration = midi_frames / FRAME_RATE

    midi_extent = [0, duration, MIDI_LOW, MIDI_LOW + num_pitches - 1]
    mel_extent = [0, duration, 0, mel_bins - 1]

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(14, 10),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].imshow(
        piano_roll[0].T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=midi_extent,
    )
    axes[0].set_ylabel("MIDI pitch")
    axes[0].set_title("active notes")

    axes[1].imshow(
        piano_roll[1].T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=midi_extent,
    )
    axes[1].set_ylabel("MIDI pitch")
    axes[1].set_title("onsets")

    axes[2].imshow(
        piano_roll[2].T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=midi_extent,
    )
    axes[2].set_ylabel("MIDI pitch")
    axes[2].set_title("velocity-onsets")

    # Better contrast for presentation-style spectrogram.
    vmin = float(log_mel.min())
    vmax = float(torch.tensor(log_mel).quantile(0.99)) if log_mel.size else None

    axes[3].imshow(
        log_mel,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=mel_extent,
        vmin=vmin,
        vmax=vmax,
    )
    axes[3].set_ylabel("mel bin")
    axes[3].set_title("target log-mel spectrogram")
    axes[3].set_xlabel("time (seconds)")

    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4/5 sanity check: Option4Dataset and DataLoader."
    )
    parser.add_argument("--subset-name", type=str, default="smoke")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()

    index_csv = (
        WINDOW_INDEX_CACHE_DIR
        / f"option4_{args.subset_name}_{args.split}_windows.csv"
    )

    if not index_csv.exists():
        raise FileNotFoundError(
            f"Window index not found: {index_csv}\n"
            "Run Step 4 first, for example:\n"
            "python scripts/option4/04_build_option4_window_index.py "
            "--subset-name smoke --train-max-windows 128 "
            "--val-max-windows 32 --test-max-windows 32"
        )

    print("=" * 80)
    print("Step 4/5: Check Option4Dataset")
    print("=" * 80)
    print(f"index_csv:   {index_csv}")
    print(f"subset_name: {args.subset_name}")
    print(f"split:       {args.split}")
    print()

    dataset = Option4MidiToAudioDataset(index_csv=index_csv, return_audio=True)

    print(f"Dataset size: {len(dataset)}")

    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(
            f"sample_index={args.sample_index} out of range for dataset size {len(dataset)}"
        )

    sample = dataset[args.sample_index]

    print()
    print("Single sample:")
    print(f"window_id:    {sample['window_id']}")
    print(f"piece_id:     {sample['piece_id']}")
    print(f"composer:     {sample['composer']}")
    print(f"title:        {sample['title']}")
    print(f"piano_roll:   {tuple(sample['piano_roll'].shape)} {sample['piano_roll'].dtype}")
    print(f"log_mel:      {tuple(sample['log_mel'].shape)} {sample['log_mel'].dtype}")
    print(f"audio:        {tuple(sample['audio'].shape)} {sample['audio'].dtype}")
    print(f"start_sec:    {sample['start_sec'].item():.2f}")
    print(f"clip_seconds: {sample['clip_seconds'].item():.2f}")

    print()
    print("Value ranges:")
    print(
        f"piano_roll min/max: "
        f"{sample['piano_roll'].min().item():.4f} / {sample['piano_roll'].max().item():.4f}"
    )
    print(
        f"log_mel min/max: "
        f"{sample['log_mel'].min().item():.4f} / {sample['log_mel'].max().item():.4f}"
    )
    print(
        f"audio min/max: "
        f"{sample['audio'].min().item():.4f} / {sample['audio'].max().item():.4f}"
    )

    loader = make_option4_dataloader(
        index_csv=index_csv,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        return_audio=False,
    )

    batch = next(iter(loader))

    print()
    print("Batch:")
    print(f"piano_roll: {tuple(batch['piano_roll'].shape)}")
    print(f"log_mel:    {tuple(batch['log_mel'].shape)}")
    print(f"window_ids: {batch['window_id'][:min(3, len(batch['window_id']))]}")

    figure_path = (
        OPTION4_OUTPUT_DIR
        / "figures"
        / "step4_option4_dataset_example.png"
    )

    plot_dataset_sample(
        piano_roll=sample["piano_roll"],
        log_mel=sample["log_mel"],
        output_path=figure_path,
        title=(
            "Step 4 Option4Dataset example\n"
            f"{sample['composer']} — {sample['title']}"
        ),
    )

    print()
    print(f"Saved dataset sample figure to: {figure_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
