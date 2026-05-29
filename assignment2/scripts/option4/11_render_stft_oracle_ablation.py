from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.shared.config import (
    SAMPLE_RATE,
    HOP_LENGTH,
    N_FFT,
    WIN_LENGTH,
    N_MELS,
    FMIN,
    FMAX,
    CENTER,
    WINDOW_INDEX_CACHE_DIR,
    OPTION4_OUTPUT_DIR,
)
from app.option4.audio_preprocessing import load_audio_window_to_logmel
from app.option4.cached_dataset import CachedOption4MidiToAudioDataset, option4_cache_dir
from app.option4.option4_dataset import Option4MidiToAudioDataset
from app.option4.audio_rendering import (
    audio_to_stft_magnitude,
    stft_magnitude_to_audio_griffinlim,
    logmel_to_audio_griffinlim,
    trim_or_pad_audio,
    peak_normalize,
    save_audio_pair,
)


def get_fmax() -> float:
    if FMAX is None:
        return float(SAMPLE_RATE) / 2.0
    return float(FMAX)


def build_dataset(subset_name: str, split: str, use_cache: bool):
    if use_cache:
        return CachedOption4MidiToAudioDataset(
            cache_dir=option4_cache_dir(subset_name, split)
        )

    index_csv = WINDOW_INDEX_CACHE_DIR / f"option4_{subset_name}_{split}_windows.csv"
    return Option4MidiToAudioDataset(index_csv=index_csv, return_audio=False)


def get_dataset_metadata(dataset) -> pd.DataFrame:
    if hasattr(dataset, "metadata"):
        return dataset.metadata
    if hasattr(dataset, "index"):
        return dataset.index
    raise TypeError("Dataset does not expose metadata/index.")


def plot_oracle_comparison(
    target_log_mel: torch.Tensor,
    stft_magnitudes: dict[str, np.ndarray],
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_log_mel = target_log_mel.detach().cpu()

    n_rows = 1 + len(stft_magnitudes)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=2,
        figsize=(14, 4 * n_rows),
        constrained_layout=True,
    )

    if n_rows == 1:
        axes = np.asarray([axes])

    target_vmax = float(torch.quantile(target_log_mel, 0.99).item())
    target_vmax = max(target_vmax, 1e-6)

    im = axes[0, 0].imshow(
        target_log_mel,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=target_vmax,
    )
    axes[0, 0].set_title("Ground-truth log-mel target")
    axes[0, 0].set_xlabel("time frame")
    axes[0, 0].set_ylabel("mel bin")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    axes[0, 1].axis("off")
    axes[0, 1].text(
        0.02,
        0.8,
        "This ablation compares:\n"
        "1. 80-bin log-mel → mel_to_audio → Griffin-Lim\n"
        "2. linear STFT magnitude → Griffin-Lim\n\n"
        "If STFT oracle sounds much better, the main bottleneck is the compact log-mel renderer.",
        fontsize=11,
        va="top",
    )

    for row_idx, (label, mag) in enumerate(stft_magnitudes.items(), start=1):
        mag = np.asarray(mag, dtype=np.float32)
        log_mag = np.log1p(mag)

        vmax = float(np.quantile(log_mag, 0.99))
        vmax = max(vmax, 1e-6)

        im = axes[row_idx, 0].imshow(
            log_mag,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=vmax,
        )
        axes[row_idx, 0].set_title(f"{label}: log1p(STFT magnitude)")
        axes[row_idx, 0].set_xlabel("time frame")
        axes[row_idx, 0].set_ylabel("frequency bin")
        fig.colorbar(im, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)

        axes[row_idx, 1].plot(log_mag.mean(axis=0))
        axes[row_idx, 1].set_title(f"{label}: mean log-magnitude over frequency")
        axes[row_idx, 1].set_xlabel("time frame")
        axes[row_idx, 1].set_ylabel("mean log magnitude")
        axes[row_idx, 1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 8.2: STFT oracle rendering ablation."
    )

    parser.add_argument("--subset-name", type=str, default="debug")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--use-cache", action="store_true")

    parser.add_argument(
        "--n-iters",
        type=int,
        nargs="+",
        default=[64, 128, 256],
        help="Griffin-Lim iteration counts to test.",
    )

    parser.add_argument(
        "--stft-n-ffts",
        type=int,
        nargs="+",
        default=[N_FFT],
        help="STFT n_fft values to test for oracle reconstruction.",
    )

    parser.add_argument(
        "--stft-win-lengths",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional win_length values. If omitted, each win_length equals its n_fft. "
            "If provided, must have the same number of values as --stft-n-ffts."
        ),
    )

    parser.add_argument("--make-mp3", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")

    args = parser.parse_args()

    normalize = not args.no_normalize

    if args.stft_win_lengths is None or len(args.stft_win_lengths) == 0:
        stft_win_lengths = list(args.stft_n_ffts)
    else:
        stft_win_lengths = args.stft_win_lengths

    if len(stft_win_lengths) != len(args.stft_n_ffts):
        raise ValueError(
            "--stft-win-lengths must be omitted or have same length as --stft-n-ffts"
        )

    dataset = build_dataset(
        subset_name=args.subset_name,
        split=args.split,
        use_cache=args.use_cache,
    )

    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(
            f"sample_index={args.sample_index} out of range for dataset size {len(dataset)}"
        )

    sample = dataset[args.sample_index]
    metadata = get_dataset_metadata(dataset)
    row = metadata.iloc[args.sample_index]

    target_log_mel = sample["log_mel"].float()

    clip_seconds = float(sample["clip_seconds"].item())
    start_sec = float(sample["start_sec"].item())
    target_num_samples = int(round(clip_seconds * SAMPLE_RATE))

    output_dir = (
        OPTION4_OUTPUT_DIR
        / "audio"
        / "stft_oracle_ablation"
        / f"{args.subset_name}_{args.split}_sample{args.sample_index}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Step 8.2: STFT oracle rendering ablation")
    print("=" * 80)
    print(f"subset/split:      {args.subset_name}/{args.split}")
    print(f"sample_index:      {args.sample_index}")
    print(f"use_cache:         {args.use_cache}")
    print(f"output_dir:        {output_dir}")
    print(f"window_id:         {sample['window_id']}")
    print(f"piece:             {sample['composer']} — {sample['title']}")
    print(f"start_sec:         {start_sec}")
    print(f"clip_seconds:      {clip_seconds}")
    print(f"n_iters:           {args.n_iters}")
    print(f"stft_n_ffts:       {args.stft_n_ffts}")
    print(f"stft_win_lengths:  {stft_win_lengths}")
    print(f"normalize:         {normalize}")
    print()

    # ------------------------------------------------------------------
    # Load original audio window
    # ------------------------------------------------------------------
    audio_original, _, _ = load_audio_window_to_logmel(
        audio_path=row["audio_path"],
        start_sec=start_sec,
        clip_seconds=clip_seconds,
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=get_fmax(),
        center=CENTER,
        expected_frames=target_log_mel.shape[-1],
    )

    audio_original = np.asarray(audio_original, dtype=np.float32)
    audio_original = trim_or_pad_audio(audio_original, target_num_samples)

    if normalize:
        audio_original_save = peak_normalize(audio_original)
    else:
        audio_original_save = audio_original

    audio_outputs: dict[str, dict[str, str | None]] = {}

    audio_outputs["ground_truth_original"] = save_audio_pair(
        wav_path=output_dir / "ground_truth_original.wav",
        audio=audio_original_save,
        sample_rate=SAMPLE_RATE,
        make_mp3=args.make_mp3,
    )

    # ------------------------------------------------------------------
    # Ground-truth log-mel reconstruction
    # ------------------------------------------------------------------
    for n_iter in args.n_iters:
        gt_logmel_recon = logmel_to_audio_griffinlim(
            log_mel=target_log_mel,
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=get_fmax(),
            n_iter=n_iter,
            target_num_samples=target_num_samples,
            normalize=normalize,
        )

        key = f"gt_logmel_reconstruction_iter{n_iter}"

        audio_outputs[key] = save_audio_pair(
            wav_path=output_dir / f"{key}.wav",
            audio=gt_logmel_recon,
            sample_rate=SAMPLE_RATE,
            make_mp3=args.make_mp3,
        )

    # ------------------------------------------------------------------
    # STFT oracle reconstruction
    # ------------------------------------------------------------------
    stft_magnitudes_for_plot: dict[str, np.ndarray] = {}

    for n_fft, win_length in zip(args.stft_n_ffts, stft_win_lengths):
        mag = audio_to_stft_magnitude(
            audio=audio_original,
            n_fft=n_fft,
            hop_length=HOP_LENGTH,
            win_length=win_length,
            center=CENTER,
        )

        stft_magnitudes_for_plot[f"nfft{n_fft}_win{win_length}"] = mag

        for n_iter in args.n_iters:
            stft_recon = stft_magnitude_to_audio_griffinlim(
                magnitude=mag,
                sample_rate=SAMPLE_RATE,
                n_fft=n_fft,
                hop_length=HOP_LENGTH,
                win_length=win_length,
                center=CENTER,
                n_iter=n_iter,
                target_num_samples=target_num_samples,
                normalize=normalize,
            )

            key = f"gt_stft_reconstruction_nfft{n_fft}_win{win_length}_iter{n_iter}"

            audio_outputs[key] = save_audio_pair(
                wav_path=output_dir / f"{key}.wav",
                audio=stft_recon,
                sample_rate=SAMPLE_RATE,
                make_mp3=args.make_mp3,
            )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    figure_path = output_dir / "stft_oracle_comparison.png"

    plot_oracle_comparison(
        target_log_mel=target_log_mel,
        stft_magnitudes=stft_magnitudes_for_plot,
        output_path=figure_path,
        title=(
            "STFT oracle rendering ablation\n"
            f"{sample['composer']} — {sample['title']}\n"
            f"window_id={sample['window_id']}"
        ),
    )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    metadata_json: dict[str, Any] = {
        "subset_name": args.subset_name,
        "split": args.split,
        "sample_index": args.sample_index,
        "window_id": sample["window_id"],
        "piece_id": sample["piece_id"],
        "composer": sample["composer"],
        "title": sample["title"],
        "start_sec": start_sec,
        "clip_seconds": clip_seconds,
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "current_logmel_n_fft": N_FFT,
        "current_logmel_win_length": WIN_LENGTH,
        "current_logmel_n_mels": N_MELS,
        "current_logmel_fmin": FMIN,
        "current_logmel_fmax": get_fmax(),
        "center": CENTER,
        "n_iters": args.n_iters,
        "stft_n_ffts": args.stft_n_ffts,
        "stft_win_lengths": stft_win_lengths,
        "normalize": normalize,
        "audio_outputs": audio_outputs,
        "figure": str(figure_path),
    }

    metadata_path = output_dir / "stft_oracle_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata_json, f, indent=2)

    print()
    print("Saved audio outputs:")
    for key, paths in audio_outputs.items():
        print(f"  {key}: {paths['wav']}")
        if paths["mp3"] is not None:
            print(f"       mp3: {paths['mp3']}")

    print()
    print(f"Saved figure:   {figure_path}")
    print(f"Saved metadata: {metadata_path}")
    print("=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
