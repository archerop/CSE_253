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
    MIDI_LOW,
    WINDOW_INDEX_CACHE_DIR,
    OPTION4_OUTPUT_DIR,
)
from app.option4.audio_models import ContourNetLiteUNet
from app.option4.audio_preprocessing import load_audio_window_to_logmel
from app.option4.cached_dataset import CachedOption4MidiToAudioDataset, option4_cache_dir
from app.option4.option4_dataset import Option4MidiToAudioDataset
from app.option4.metrics import compute_batch_metrics
from app.option4.audio_rendering import (
    peak_normalize,
    save_audio_pair,
    logmel_to_audio_griffinlim,
)


def safe_torch_load(path: str | Path, map_location):
    """
    Load a checkpoint across PyTorch versions.
    """
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_fmax() -> float:
    if FMAX is None:
        return float(SAMPLE_RATE) / 2.0
    return float(FMAX)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataset(subset_name: str, split: str, use_cache: bool):
    if use_cache:
        cache_dir = option4_cache_dir(subset_name, split)
        return CachedOption4MidiToAudioDataset(cache_dir=cache_dir)

    index_csv = WINDOW_INDEX_CACHE_DIR / f"option4_{subset_name}_{split}_windows.csv"
    return Option4MidiToAudioDataset(index_csv=index_csv, return_audio=False)


def get_dataset_metadata(dataset) -> pd.DataFrame:
    if hasattr(dataset, "metadata"):
        return dataset.metadata
    if hasattr(dataset, "index"):
        return dataset.index
    raise TypeError("Dataset does not expose metadata/index.")


def load_contour_unet_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> ContourNetLiteUNet:
    ckpt = safe_torch_load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})

    model = ContourNetLiteUNet(
        input_channels=3,
        n_mels=N_MELS,
        midi_low=MIDI_LOW,
        fmin=FMIN,
        fmax=get_fmax(),
        base_channels=int(ckpt_args.get("base_channels", 32)),
        dropout=float(ckpt_args.get("dropout", 0.0)),
        condition_strength=float(ckpt_args.get("condition_strength", 1.0)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model


@torch.no_grad()
def predict_logmel(
    model: torch.nn.Module,
    piano_roll: torch.Tensor,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    model.eval()

    x = piano_roll.unsqueeze(0).to(device)

    with torch.cuda.amp.autocast(enabled=use_amp):
        pred = model(x)

    return pred[0].detach().cpu()


def render_one_logmel(
    log_mel: torch.Tensor,
    clip_seconds: float,
    n_iter: int,
    normalize: bool,
) -> np.ndarray:
    target_num_samples = int(round(float(clip_seconds) * SAMPLE_RATE))

    audio = logmel_to_audio_griffinlim(
        log_mel=log_mel,
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

    return audio


def plot_rendering_comparison(
    piano_roll: torch.Tensor,
    target: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    piano_roll = piano_roll.detach().cpu()
    target = target.detach().cpu()

    n_models = len(predictions)

    fig, axes = plt.subplots(
        nrows=2 + n_models,
        ncols=3,
        figsize=(16, 4 + 3 * n_models),
        constrained_layout=True,
    )

    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    target_vmax = float(torch.quantile(target, 0.99).item())
    target_vmax = max(target_vmax, 1e-6)

    # Row 0: inputs
    input_images = [
        (piano_roll[0].T, "Input active notes", None, None),
        (piano_roll[1].T, "Input onsets", None, None),
        (piano_roll[2].T, "Input velocity-onsets", None, None),
    ]

    for ax, (image, subtitle, vmin, vmax) in zip(axes[0], input_images):
        im = ax.imshow(
            image,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(subtitle)
        ax.set_xlabel("time frame")
        ax.set_ylabel("pitch bin")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 1: target
    target_images = [
        (target, "Target log-mel", 0.0, target_vmax),
        (target, "Target log-mel again", 0.0, target_vmax),
        (torch.zeros_like(target), "Zero reference", 0.0, target_vmax),
    ]

    for ax, (image, subtitle, vmin, vmax) in zip(axes[1], target_images):
        im = ax.imshow(
            image,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(subtitle)
        ax.set_xlabel("time frame")
        ax.set_ylabel("mel bin")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Rows for models
    for row_idx, (label, pred) in enumerate(predictions.items(), start=2):
        pred = pred.detach().cpu()
        pred_show = pred.clamp_min(0.0)
        error = torch.abs(pred - target)

        err_vmax = float(torch.quantile(error, 0.99).item())
        err_vmax = max(err_vmax, 1e-6)

        images = [
            (pred_show, f"{label}: predicted log-mel", 0.0, target_vmax),
            (error, f"{label}: |prediction - target|", 0.0, err_vmax),
            (target - pred, f"{label}: target - raw prediction", -err_vmax, err_vmax),
        ]

        for ax, (image, subtitle, vmin, vmax) in zip(axes[row_idx], images):
            im = ax.imshow(
                image,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(subtitle)
            ax.set_xlabel("time frame")
            ax.set_ylabel("mel bin")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 8: Render Option 4 predicted log-mel spectrograms to audio."
    )

    parser.add_argument("--subset-name", type=str, default="debug")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--use-cache", action="store_true")

    parser.add_argument(
        "--l1-checkpoint",
        type=str,
        default="outputs/option4/checkpoints/contour_unet_debug_l1_best.pt",
    )
    parser.add_argument(
        "--weighted-checkpoint",
        type=str,
        default="outputs/option4/checkpoints/contour_unet_debug_weighted_energy_onset_best.pt",
    )

    parser.add_argument("--n-iter", type=int, default=64)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--make-mp3", action="store_true")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable peak normalization for rendered audio examples.",
    )

    args = parser.parse_args()

    device = get_device()
    use_amp = bool(args.amp and device.type == "cuda")
    normalize = not args.no_normalize

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

    piano_roll = sample["piano_roll"].float()
    target_log_mel = sample["log_mel"].float()

    clip_seconds = float(sample["clip_seconds"].item())
    start_sec = float(sample["start_sec"].item())
    target_num_samples = int(round(clip_seconds * SAMPLE_RATE))

    output_dir = (
        OPTION4_OUTPUT_DIR
        / "audio"
        / "render_examples"
        / f"{args.subset_name}_{args.split}_sample{args.sample_index}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Step 8: Render Option 4 audio examples")
    print("=" * 80)
    print(f"subset/split:       {args.subset_name}/{args.split}")
    print(f"sample_index:       {args.sample_index}")
    print(f"use_cache:          {args.use_cache}")
    print(f"device:             {device}")
    print(f"use_amp:            {use_amp}")
    print(f"n_iter:             {args.n_iter}")
    print(f"normalize:          {normalize}")
    print(f"output_dir:         {output_dir}")
    print(f"window_id:          {sample['window_id']}")
    print(f"piece:              {sample['composer']} — {sample['title']}")
    print(f"start_sec:          {start_sec}")
    print(f"clip_seconds:       {clip_seconds}")
    print()

    # ------------------------------------------------------------------
    # 1. Original ground-truth audio window
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

    if len(audio_original) != target_num_samples:
        from app.option4.audio_rendering import trim_or_pad_audio
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
    # 2. Ground-truth log-mel reconstruction through same renderer
    # ------------------------------------------------------------------
    gt_reconstruction = render_one_logmel(
        log_mel=target_log_mel,
        clip_seconds=clip_seconds,
        n_iter=args.n_iter,
        normalize=normalize,
    )

    audio_outputs["ground_truth_logmel_reconstruction"] = save_audio_pair(
        wav_path=output_dir / "ground_truth_logmel_reconstruction.wav",
        audio=gt_reconstruction,
        sample_rate=SAMPLE_RATE,
        make_mp3=args.make_mp3,
    )

    # ------------------------------------------------------------------
    # 3. Model predictions and generated audio
    # ------------------------------------------------------------------
    predictions: dict[str, torch.Tensor] = {}
    metrics: dict[str, dict[str, float]] = {}

    checkpoints = {
        "contour_unet_l1": args.l1_checkpoint,
        "contour_unet_weighted": args.weighted_checkpoint,
    }

    for label, ckpt_path in checkpoints.items():
        ckpt_path = Path(ckpt_path)

        if not ckpt_path.exists():
            print(f"[warn] checkpoint not found, skip {label}: {ckpt_path}")
            continue

        print(f"Loading checkpoint for {label}: {ckpt_path}")
        model = load_contour_unet_from_checkpoint(ckpt_path, device=device)

        pred_log_mel = predict_logmel(
            model=model,
            piano_roll=piano_roll,
            device=device,
            use_amp=use_amp,
        )

        predictions[label] = pred_log_mel

        metric_dict = compute_batch_metrics(
            pred=pred_log_mel.unsqueeze(0),
            target=target_log_mel.unsqueeze(0),
        )
        metrics[label] = metric_dict

        generated_audio = render_one_logmel(
            log_mel=pred_log_mel,
            clip_seconds=clip_seconds,
            n_iter=args.n_iter,
            normalize=normalize,
        )

        audio_outputs[label] = save_audio_pair(
            wav_path=output_dir / f"{label}_generated.wav",
            audio=generated_audio,
            sample_rate=SAMPLE_RATE,
            make_mp3=args.make_mp3,
        )

        print(
            f"{label}: "
            f"logmel_l1={metric_dict['logmel_l1']:.6f}, "
            f"logmel_mse={metric_dict['logmel_mse']:.6f}, "
            f"energy_l1={metric_dict['energy_l1']:.6f}, "
            f"onset_l1={metric_dict['onset_l1']:.6f}"
        )

    # ------------------------------------------------------------------
    # 4. Save comparison figure
    # ------------------------------------------------------------------
    figure_path = output_dir / "rendering_comparison.png"

    plot_rendering_comparison(
        piano_roll=piano_roll,
        target=target_log_mel,
        predictions=predictions,
        output_path=figure_path,
        title=(
            "Step 8 rendering comparison\n"
            f"{sample['composer']} — {sample['title']}\n"
            f"window_id={sample['window_id']}"
        ),
    )

    # ------------------------------------------------------------------
    # 5. Save metadata
    # ------------------------------------------------------------------
    metadata_json = {
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
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "win_length": WIN_LENGTH,
        "n_mels": N_MELS,
        "fmin": FMIN,
        "fmax": get_fmax(),
        "center": CENTER,
        "griffinlim_n_iter": args.n_iter,
        "normalize": normalize,
        "audio_outputs": audio_outputs,
        "metrics": metrics,
        "figure": str(figure_path),
    }

    metadata_path = output_dir / "rendering_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata_json, f, indent=2)

    print()
    print("Saved audio outputs:")
    for name, paths in audio_outputs.items():
        print(f"  {name}:")
        print(f"    wav: {paths['wav']}")
        if paths["mp3"] is not None:
            print(f"    mp3: {paths['mp3']}")

    print()
    print(f"Saved figure:   {figure_path}")
    print(f"Saved metadata: {metadata_path}")
    print("=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
