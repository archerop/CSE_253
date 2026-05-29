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
    WIN_LENGTH,
    N_MELS,
    FMIN,
    FMAX,
    CENTER,
    MIDI_LOW,
    OPTION4_OUTPUT_DIR,
)
from app.option4.audio_preprocessing import load_audio_window_to_logmel
from app.option4.audio_models import (
    LinearProjectedStftResidualCNN,
    LinearProjectedStftUNet,
)
from app.option4.stft_cached_dataset import (
    CachedOption4StftDataset,
    option4_stft_cache_dir,
)
from app.option4.stft_preprocessing import log_stft_magnitude_to_magnitude
from app.option4.stft_metrics import compute_stft_batch_metrics
from app.option4.audio_rendering import (
    trim_or_pad_audio,
    peak_normalize,
    save_audio_pair,
    stft_magnitude_to_audio_griffinlim,
)


def safe_torch_load(path: str | Path, map_location):
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


def audio_rms(audio: np.ndarray, eps: float = 1e-8) -> float:
    audio = np.asarray(audio, dtype=np.float32)
    return float(np.sqrt(np.mean(audio ** 2) + eps))


def limit_peak(audio: np.ndarray, peak: float = 0.95, eps: float = 1e-8) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs > peak:
        audio = audio / max(max_abs, eps) * peak
    return audio.astype(np.float32)


def match_rms_to_reference(
    audio: np.ndarray,
    reference_audio: np.ndarray,
    max_gain: float = 8.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Match generated/reconstructed audio RMS to original audio, with max gain
    cap to avoid amplifying low-energy Griffin-Lim noise too much.
    """
    audio = np.asarray(audio, dtype=np.float32)
    ref_rms = audio_rms(reference_audio, eps=eps)
    cur_rms = audio_rms(audio, eps=eps)

    if cur_rms < eps:
        return audio

    gain = min(ref_rms / cur_rms, max_gain)
    return (audio * gain).astype(np.float32)


def normalize_for_export(
    audio: np.ndarray,
    reference_audio: np.ndarray,
    mode: str,
    peak: float = 0.95,
) -> np.ndarray:
    if mode == "none":
        return audio.astype(np.float32)

    if mode == "peak":
        return peak_normalize(audio, peak=peak)

    if mode == "match-rms":
        matched = match_rms_to_reference(audio, reference_audio)
        return limit_peak(matched, peak=peak)

    raise ValueError(f"Unknown normalization mode: {mode}")


def infer_model_type_from_checkpoint(ckpt: dict[str, Any], path: Path) -> str:
    model_type = ckpt.get("model_type")
    if model_type in {"residual_cnn", "unet"}:
        return model_type

    name = ckpt.get("model_name", "")
    if "Residual" in name or "residual" in path.name:
        return "residual_cnn"
    if "UNet" in name or "unet" in path.name:
        return "unet"

    raise ValueError(f"Cannot infer model type from checkpoint: {path}")


def load_stft_model_from_checkpoint(
    checkpoint_path: str | Path,
    n_freq_bins: int,
    device: torch.device,
):
    checkpoint_path = Path(checkpoint_path)
    ckpt = safe_torch_load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model_type = infer_model_type_from_checkpoint(ckpt, checkpoint_path)

    n_fft = int(ckpt_args.get("n_fft", 1024))

    if model_type == "residual_cnn":
        model = LinearProjectedStftResidualCNN(
            input_channels=3,
            n_freq_bins=n_freq_bins,
            sample_rate=SAMPLE_RATE,
            n_fft=n_fft,
            midi_low=MIDI_LOW,
            hidden_channels=int(ckpt_args.get("hidden_channels", 48)),
            num_blocks=int(ckpt_args.get("num_blocks", 8)),
            dropout=float(ckpt_args.get("dropout", 0.0)),
            condition_strength=float(ckpt_args.get("condition_strength", 1.0)),
        ).to(device)

    elif model_type == "unet":
        model = LinearProjectedStftUNet(
            input_channels=3,
            n_freq_bins=n_freq_bins,
            sample_rate=SAMPLE_RATE,
            n_fft=n_fft,
            midi_low=MIDI_LOW,
            base_channels=int(ckpt_args.get("base_channels", 24)),
            blocks_per_level=int(ckpt_args.get("blocks_per_level", 2)),
            refinement_blocks=int(ckpt_args.get("refinement_blocks", 2)),
            dropout=float(ckpt_args.get("dropout", 0.0)),
            condition_strength=float(ckpt_args.get("condition_strength", 1.0)),
        ).to(device)

    else:
        raise ValueError(f"Unknown model_type={model_type}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, model_type, ckpt_args


@torch.no_grad()
def predict_log_stft(
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


def render_log_stft_to_audio(
    log_stft_mag: torch.Tensor | np.ndarray,
    n_fft: int,
    win_length: int,
    n_iter: int,
    target_num_samples: int,
) -> np.ndarray:
    mag = log_stft_magnitude_to_magnitude(log_stft_mag)

    audio = stft_magnitude_to_audio_griffinlim(
        magnitude=mag,
        sample_rate=SAMPLE_RATE,
        n_fft=n_fft,
        hop_length=HOP_LENGTH,
        win_length=win_length,
        center=CENTER,
        n_iter=n_iter,
        target_num_samples=target_num_samples,
        normalize=False,
    )

    return audio.astype(np.float32)


def plot_stft_rendering_comparison(
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
        (target, "Target log-STFT magnitude", 0.0, target_vmax),
        (target, "Target log-STFT again", 0.0, target_vmax),
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
        ax.set_ylabel("frequency bin")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Rows for models
    for row_idx, (label, pred) in enumerate(predictions.items(), start=2):
        pred = pred.detach().cpu()
        pred_show = pred.clamp_min(0.0)
        error = torch.abs(pred - target)

        err_vmax = float(torch.quantile(error, 0.99).item())
        err_vmax = max(err_vmax, 1e-6)

        images = [
            (pred_show, f"{label}: predicted log-STFT", 0.0, target_vmax),
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
            ax.set_ylabel("frequency bin")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render 513-bin STFT model predictions to audio."
    )

    parser.add_argument("--subset-name", type=str, default="debug")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--win-length", type=int, default=1024)

    parser.add_argument(
        "--residual-checkpoint",
        type=str,
        default="outputs/option4/checkpoints/stft_residual_cnn_debug_weighted_energy_onset_nfft1024_best.pt",
    )
    parser.add_argument(
        "--unet-checkpoint",
        type=str,
        default="outputs/option4/checkpoints/stft_unet_debug_weighted_energy_onset_nfft1024_best.pt",
    )

    parser.add_argument("--n-iter", type=int, default=256)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--make-mp3", action="store_true")
    parser.add_argument(
        "--normalization",
        type=str,
        default="match-rms",
        choices=["match-rms", "peak", "none"],
    )

    args = parser.parse_args()

    device = get_device()
    use_amp = bool(args.amp and device.type == "cuda")

    cache_dir = option4_stft_cache_dir(args.subset_name, args.split, args.n_fft)
    dataset = CachedOption4StftDataset(cache_dir=cache_dir)

    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(
            f"sample_index={args.sample_index} out of range for dataset size {len(dataset)}"
        )

    sample = dataset[args.sample_index]
    metadata = dataset.metadata
    row = metadata.iloc[args.sample_index]

    piano_roll = sample["piano_roll"].float()
    target_log_stft = sample["target"].float()

    n_freq_bins = int(target_log_stft.shape[0])
    clip_seconds = float(sample["clip_seconds"].item())
    start_sec = float(sample["start_sec"].item())
    target_num_samples = int(round(clip_seconds * SAMPLE_RATE))

    output_dir = (
        OPTION4_OUTPUT_DIR
        / "audio"
        / "stft_model_render_examples"
        / f"{args.subset_name}_{args.split}_sample{args.sample_index}_nfft{args.n_fft}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Step 11B: Render STFT model audio examples")
    print("=" * 80)
    print(f"subset/split:       {args.subset_name}/{args.split}")
    print(f"sample_index:       {args.sample_index}")
    print(f"n_fft:              {args.n_fft}")
    print(f"n_freq_bins:        {n_freq_bins}")
    print(f"n_iter:             {args.n_iter}")
    print(f"normalization:      {args.normalization}")
    print(f"device:             {device}")
    print(f"use_amp:            {use_amp}")
    print(f"output_dir:         {output_dir}")
    print(f"window_id:          {sample['window_id']}")
    print(f"piece:              {sample['composer']} — {sample['title']}")
    print(f"start_sec:          {start_sec}")
    print(f"clip_seconds:       {clip_seconds}")
    print()

    # ------------------------------------------------------------------
    # 1. Load original audio window.
    # ------------------------------------------------------------------
    audio_original, _, _ = load_audio_window_to_logmel(
        audio_path=row["audio_path"],
        start_sec=start_sec,
        clip_seconds=clip_seconds,
        sample_rate=SAMPLE_RATE,
        n_fft=args.n_fft,
        hop_length=HOP_LENGTH,
        win_length=args.win_length,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=get_fmax(),
        center=CENTER,
        expected_frames=target_log_stft.shape[-1],
    )

    audio_original = np.asarray(audio_original, dtype=np.float32)
    audio_original = trim_or_pad_audio(audio_original, target_num_samples)
    audio_original_save = limit_peak(audio_original, peak=0.95)

    audio_outputs: dict[str, dict[str, str | None]] = {}

    audio_outputs["ground_truth_original"] = save_audio_pair(
        wav_path=output_dir / "ground_truth_original.wav",
        audio=audio_original_save,
        sample_rate=SAMPLE_RATE,
        make_mp3=args.make_mp3,
    )

    # ------------------------------------------------------------------
    # 2. Ground-truth STFT reconstruction.
    # ------------------------------------------------------------------
    gt_recon_raw = render_log_stft_to_audio(
        log_stft_mag=target_log_stft,
        n_fft=args.n_fft,
        win_length=args.win_length,
        n_iter=args.n_iter,
        target_num_samples=target_num_samples,
    )

    gt_recon_save = normalize_for_export(
        gt_recon_raw,
        reference_audio=audio_original,
        mode=args.normalization,
    )

    audio_outputs["ground_truth_stft_reconstruction"] = save_audio_pair(
        wav_path=output_dir / f"ground_truth_stft_reconstruction_iter{args.n_iter}.wav",
        audio=gt_recon_save,
        sample_rate=SAMPLE_RATE,
        make_mp3=args.make_mp3,
    )

    # ------------------------------------------------------------------
    # 3. Model predictions.
    # ------------------------------------------------------------------
    predictions: dict[str, torch.Tensor] = {}
    metrics: dict[str, dict[str, float]] = {}

    checkpoints = {
        "stft_residual_cnn": args.residual_checkpoint,
        "stft_unet": args.unet_checkpoint,
    }

    for label, ckpt_path_str in checkpoints.items():
        ckpt_path = Path(ckpt_path_str)

        if not ckpt_path.exists():
            print(f"[warn] checkpoint not found, skip {label}: {ckpt_path}")
            continue

        print(f"Loading {label}: {ckpt_path}")

        model, model_type, ckpt_args = load_stft_model_from_checkpoint(
            checkpoint_path=ckpt_path,
            n_freq_bins=n_freq_bins,
            device=device,
        )

        pred_log_stft = predict_log_stft(
            model=model,
            piano_roll=piano_roll,
            device=device,
            use_amp=use_amp,
        )

        predictions[label] = pred_log_stft

        metric_dict = compute_stft_batch_metrics(
            pred=pred_log_stft.unsqueeze(0),
            target=target_log_stft.unsqueeze(0),
        )
        metrics[label] = metric_dict

        generated_raw = render_log_stft_to_audio(
            log_stft_mag=pred_log_stft,
            n_fft=args.n_fft,
            win_length=args.win_length,
            n_iter=args.n_iter,
            target_num_samples=target_num_samples,
        )

        generated_save = normalize_for_export(
            generated_raw,
            reference_audio=audio_original,
            mode=args.normalization,
        )

        audio_outputs[label] = save_audio_pair(
            wav_path=output_dir / f"{label}_generated_iter{args.n_iter}.wav",
            audio=generated_save,
            sample_rate=SAMPLE_RATE,
            make_mp3=args.make_mp3,
        )

        print(
            f"{label}: "
            f"stft_l1={metric_dict['stft_l1']:.6f}, "
            f"stft_mse={metric_dict['stft_mse']:.6f}, "
            f"energy_l1={metric_dict['energy_l1']:.6f}, "
            f"onset_l1={metric_dict['onset_l1']:.6f}"
        )

    # ------------------------------------------------------------------
    # 4. Save comparison figure.
    # ------------------------------------------------------------------
    figure_path = output_dir / "stft_rendering_comparison.png"

    plot_stft_rendering_comparison(
        piano_roll=piano_roll,
        target=target_log_stft,
        predictions=predictions,
        output_path=figure_path,
        title=(
            "STFT model rendering comparison\n"
            f"{sample['composer']} — {sample['title']}\n"
            f"window_id={sample['window_id']}"
        ),
    )

    # ------------------------------------------------------------------
    # 5. Save metadata.
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
        "n_fft": args.n_fft,
        "n_freq_bins": n_freq_bins,
        "hop_length": HOP_LENGTH,
        "win_length": args.win_length,
        "center": CENTER,
        "griffinlim_n_iter": args.n_iter,
        "normalization": args.normalization,
        "audio_outputs": audio_outputs,
        "metrics": metrics,
        "figure": str(figure_path),
    }

    metadata_path = output_dir / "stft_rendering_metadata.json"

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
