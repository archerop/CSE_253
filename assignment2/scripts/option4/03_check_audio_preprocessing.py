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
    SAMPLE_RATE,
    HOP_LENGTH,
    FRAME_RATE,
    MIDI_LOW,
    MIDI_HIGH,
    CLIP_SECONDS,
    ONSET_WIDTH_FRAMES,
    N_FFT,
    WIN_LENGTH,
    N_MELS,
    FMIN,
    FMAX,
    CENTER,
    METADATA_CACHE_DIR,
    FIGURE_DIR,
    CACHE_DIR,
)
from app.option4.midi_features import (
    CHANNEL_NAMES,
    midi_to_pianoroll_features,
    summarize_pianoroll_features,
    validate_pianoroll_features,
)
from app.option4.audio_preprocessing import (
    load_audio_window_to_logmel,
    save_logmel_npz,
    summarize_audio_and_logmel,
    validate_audio_and_logmel,
)


def select_metadata_row(
    metadata: pd.DataFrame,
    split: str,
    row_index: int,
    start_sec: float,
    clip_seconds: float,
) -> pd.Series:
    """
    Select one MAESTRO row long enough for the requested window.
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
            f"row_index={row_index} out of range for subset size {len(subset)}"
        )

    return subset.iloc[row_index]


def plot_aligned_midi_audio_example(
    piano_roll: np.ndarray,
    log_mel: np.ndarray,
    output_path: Path,
    start_sec: float,
    frame_rate: float,
    midi_low: int,
    title: str,
) -> None:
    """
    Plot MIDI piano-roll channels and corresponding log-mel spectrogram.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    channels, midi_frames, num_pitches = piano_roll.shape
    mel_bins, mel_frames = log_mel.shape

    midi_duration = midi_frames / frame_rate
    mel_duration = mel_frames / frame_rate

    midi_extent = [
        start_sec,
        start_sec + midi_duration,
        midi_low,
        midi_low + num_pitches - 1,
    ]

    mel_extent = [
        start_sec,
        start_sec + mel_duration,
        0,
        mel_bins - 1,
    ]

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(14, 10),
        sharex=True,
        constrained_layout=True,
    )

    # Active notes.
    axes[0].imshow(
        piano_roll[0].T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=midi_extent,
    )
    axes[0].set_ylabel("MIDI pitch")
    axes[0].set_title("MIDI active notes")

    # Onsets.
    axes[1].imshow(
        piano_roll[1].T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=midi_extent,
    )
    axes[1].set_ylabel("MIDI pitch")
    axes[1].set_title("MIDI onsets")

    # Velocity-onsets.
    axes[2].imshow(
        piano_roll[2].T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=midi_extent,
    )
    axes[2].set_ylabel("MIDI pitch")
    axes[2].set_title("MIDI velocity-onsets")

    # Log-mel spectrogram.
    axes[3].imshow(
        log_mel,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=mel_extent,
    )
    axes[3].set_ylabel("mel bin")
    axes[3].set_title("Audio log-mel spectrogram")
    axes[3].set_xlabel("time (seconds)")

    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3 sanity check: audio preprocessing and MIDI/audio alignment."
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
    print("Step 3: Audio preprocessing + MIDI/audio alignment check")
    print("=" * 80)
    print(f"piece_id:   {row['piece_id']}")
    print(f"split:      {row['split']}")
    print(f"composer:   {row['composer']}")
    print(f"title:      {row['title']}")
    print(f"duration:   {row['duration']:.2f} sec")
    print(f"midi_path:  {row['midi_path']}")
    print(f"audio_path: {row['audio_path']}")
    print()

    # Expected number of frames for both MIDI and log-mel.
    expected_frames = int(np.ceil(args.seconds * FRAME_RATE))

    # Step 2 representation for the same time window.
    piano_roll, midi_feature_metadata = midi_to_pianoroll_features(
        midi_path=row["midi_path"],
        frame_rate=FRAME_RATE,
        midi_low=MIDI_LOW,
        midi_high=MIDI_HIGH,
        start_sec=args.start_sec,
        clip_seconds=args.seconds,
        onset_width_frames=args.onset_width_frames,
    )

    validate_pianoroll_features(piano_roll)
    midi_stats = summarize_pianoroll_features(piano_roll)

    # Step 3 target for the same time window.
    audio, log_mel, audio_metadata = load_audio_window_to_logmel(
        audio_path=row["audio_path"],
        start_sec=args.start_sec,
        clip_seconds=args.seconds,
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        center=CENTER,
        expected_frames=expected_frames,
    )

    validate_audio_and_logmel(
        audio=audio,
        log_mel=log_mel,
        n_mels=N_MELS,
        expected_frames=expected_frames,
    )

    audio_stats = summarize_audio_and_logmel(audio, log_mel)

    print("Alignment check:")
    print(f"expected_frames:       {expected_frames}")
    print(f"piano_roll shape:      {piano_roll.shape}  # (channels, frames, pitches)")
    print(f"log_mel shape:         {log_mel.shape}     # (mel_bins, frames)")
    print(f"MIDI frames == mel frames? {piano_roll.shape[1] == log_mel.shape[1]}")
    print()

    if piano_roll.shape[1] != log_mel.shape[1]:
        raise ValueError(
            f"Frame mismatch: piano_roll has {piano_roll.shape[1]} frames, "
            f"log_mel has {log_mel.shape[1]} frames."
        )

    print("MIDI feature metadata:")
    for key, value in midi_feature_metadata.items():
        print(f"{key}: {value}")
    print()

    print("MIDI feature statistics:")
    for key, value in midi_stats.items():
        print(f"{key}: {value}")
    print()

    print("Audio/log-mel metadata:")
    for key, value in audio_metadata.items():
        print(f"{key}: {value}")
    print()

    print("Audio/log-mel statistics:")
    for key, value in audio_stats.items():
        print(f"{key}: {value}")
    print()

    # Save audio/log-mel sample.
    audio_feature_dir = CACHE_DIR / "audio_features"
    audio_feature_path = audio_feature_dir / "sample_logmel_features.npz"

    audio_metadata_combined = {
        **audio_metadata,
        "piece_id": row["piece_id"],
        "split": row["split"],
        "composer": row["composer"],
        "title": row["title"],
    }

    save_logmel_npz(
        output_path=audio_feature_path,
        audio=audio,
        log_mel=log_mel,
        metadata=audio_metadata_combined,
    )

    print(f"Saved audio/log-mel sample to: {audio_feature_path}")

    # Save aligned Option 4 pair.
    aligned_dir = CACHE_DIR / "aligned_examples"
    aligned_path = aligned_dir / "sample_option4_pair.npz"
    aligned_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        aligned_path,
        piano_roll=piano_roll.astype(np.float32),
        log_mel=log_mel.astype(np.float32),
        audio=audio.astype(np.float32),
        channel_names=np.array(CHANNEL_NAMES),
        piece_id=str(row["piece_id"]),
        split=str(row["split"]),
        composer=str(row["composer"]),
        title=str(row["title"]),
        start_sec=float(args.start_sec),
        clip_seconds=float(args.seconds),
        sample_rate=int(SAMPLE_RATE),
        hop_length=int(HOP_LENGTH),
        frame_rate=float(FRAME_RATE),
    )

    print(f"Saved aligned Option 4 pair to: {aligned_path}")

    # Save figure.
    figure_path = FIGURE_DIR / "step3_aligned_midi_audio_example.png"
    figure_title = (
        f"Step 3 aligned MIDI/audio example\n"
        f"{row['composer']} — {row['title']} "
        f"({args.start_sec:.1f}s–{args.start_sec + args.seconds:.1f}s)"
    )

    plot_aligned_midi_audio_example(
        piano_roll=piano_roll,
        log_mel=log_mel,
        output_path=figure_path,
        start_sec=args.start_sec,
        frame_rate=FRAME_RATE,
        midi_low=MIDI_LOW,
        title=figure_title,
    )

    print(f"Saved alignment figure to: {figure_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
