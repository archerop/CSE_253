from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.shared.config import (
    SAMPLE_RATE,
    HOP_LENGTH,
    N_MELS,
    FMIN,
    FMAX,
    CENTER,
    MIDI_LOW,
    OPTION4_OUTPUT_DIR,
)
from app.option4.audio_preprocessing import load_audio_window_to_logmel
from app.option4.audio_rendering import (
    trim_or_pad_audio,
    peak_normalize,
    save_audio_pair,
    stft_magnitude_to_audio_griffinlim,
)
from app.option4.stft_preprocessing import log_stft_magnitude_to_magnitude
from app.option4.stft_cached_dataset import option4_stft_cache_dir
from app.option4.stft_refinement_dataset import (
    CachedStftRefinementDataset,
    option4_stft_prediction_cache_dir,
)
from app.option4.texture_refiner import MultiBandTextureRefiner
from app.option4.stft_metrics import compute_stft_batch_metrics, spectral_convergence_loss


def safe_load(path: str | Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_fmax() -> float:
    return float(SAMPLE_RATE) / 2.0 if FMAX is None else float(FMAX)


def rms(audio: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float32) ** 2) + eps))


def limit_peak(audio: np.ndarray, peak: float = 0.95, eps: float = 1e-8) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    p = float(np.max(np.abs(audio))) if audio.size else 0.0
    if p > peak:
        audio = audio / max(p, eps) * peak
    return audio.astype(np.float32)


def match_rms(audio: np.ndarray, ref: np.ndarray, max_gain: float = 8.0) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    gain = min(rms(ref) / max(rms(audio), 1e-8), max_gain)
    return limit_peak(audio * gain)


def normalize_audio(audio: np.ndarray, ref: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return audio.astype(np.float32)
    if mode == "peak":
        return peak_normalize(audio)
    if mode == "match-rms":
        return match_rms(audio, ref)
    raise ValueError(mode)


def build_refiner(checkpoint_path: Path, n_freq_bins: int, n_fft: int, device: torch.device):
    ckpt = safe_load(checkpoint_path, map_location=device)
    a = ckpt["args"]

    model = MultiBandTextureRefiner(
        n_freq_bins=n_freq_bins,
        sample_rate=SAMPLE_RATE,
        n_fft=n_fft,
        midi_low=MIDI_LOW,
        n_bands=int(a.get("n_bands", 8)),
        hidden_channels=int(a.get("hidden_channels", 32)),
        num_blocks_per_band=int(a.get("num_blocks_per_band", 3)),
        dropout=float(a.get("dropout", 0.0)),
        residual_scale=float(a.get("residual_scale", 0.2)),
        condition_strength=float(a.get("condition_strength", 1.0)),
        use_condition=bool(a.get("use_condition", True)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def render_log_stft(log_stft: torch.Tensor | np.ndarray, n_fft: int, win_length: int, n_iter: int, target_num_samples: int):
    if isinstance(log_stft, torch.Tensor):
        log_stft = log_stft.detach().cpu().numpy()

    mag = log_stft_magnitude_to_magnitude(log_stft)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-name", default="small")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--prediction-cache-name", default="stft_unet_small_best")
    parser.add_argument(
        "--refiner-checkpoint",
        default="outputs/option4/checkpoints/texture_refiner_small_weighted_energy_onset_sc0.05_bands8_nfft1024_best.pt",
    )
    parser.add_argument("--n-iter", type=int, default=256)
    parser.add_argument("--normalization", choices=["match-rms", "peak", "none"], default="match-rms")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--make-mp3", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    stft_cache = option4_stft_cache_dir(args.subset_name, args.split, args.n_fft)
    pred_cache = option4_stft_prediction_cache_dir(
        args.subset_name,
        args.split,
        args.n_fft,
        args.prediction_cache_name,
    )

    dataset = CachedStftRefinementDataset(stft_cache, pred_cache)
    sample = dataset[args.sample_index]
    row = dataset.metadata.iloc[args.sample_index]

    piano_roll = sample["piano_roll"].float()
    target = sample["target"].float()
    base_pred = sample["initial_pred"].float()

    n_freq_bins = int(target.shape[0])
    clip_seconds = float(sample["clip_seconds"].item())
    start_sec = float(sample["start_sec"].item())
    target_num_samples = int(round(clip_seconds * SAMPLE_RATE))

    refiner = build_refiner(Path(args.refiner_checkpoint), n_freq_bins, args.n_fft, device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_amp):
            refined = refiner(
                piano_roll.unsqueeze(0).to(device),
                base_pred.unsqueeze(0).to(device),
            )[0].detach().cpu()

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
        expected_frames=target.shape[-1],
    )
    audio_original = trim_or_pad_audio(np.asarray(audio_original, dtype=np.float32), target_num_samples)

    audio_gt_stft = render_log_stft(target, args.n_fft, args.win_length, args.n_iter, target_num_samples)
    audio_base = render_log_stft(base_pred, args.n_fft, args.win_length, args.n_iter, target_num_samples)
    audio_refined = render_log_stft(refined, args.n_fft, args.win_length, args.n_iter, target_num_samples)

    out_dir = (
        OPTION4_OUTPUT_DIR
        / "audio"
        / "texture_refiner_render_examples"
        / f"{args.subset_name}_{args.split}_sample{args.sample_index}_nfft{args.n_fft}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_outputs = {
        "ground_truth_original": save_audio_pair(
            out_dir / "ground_truth_original.wav",
            limit_peak(audio_original),
            SAMPLE_RATE,
            make_mp3=args.make_mp3,
        ),
        "ground_truth_stft_reconstruction": save_audio_pair(
            out_dir / f"ground_truth_stft_reconstruction_iter{args.n_iter}.wav",
            normalize_audio(audio_gt_stft, audio_original, args.normalization),
            SAMPLE_RATE,
            make_mp3=args.make_mp3,
        ),
        "base_stft_unet_generated": save_audio_pair(
            out_dir / f"base_stft_unet_generated_iter{args.n_iter}.wav",
            normalize_audio(audio_base, audio_original, args.normalization),
            SAMPLE_RATE,
            make_mp3=args.make_mp3,
        ),
        "texture_refined_generated": save_audio_pair(
            out_dir / f"texture_refined_generated_iter{args.n_iter}.wav",
            normalize_audio(audio_refined, audio_original, args.normalization),
            SAMPLE_RATE,
            make_mp3=args.make_mp3,
        ),
    }

    base_metrics = compute_stft_batch_metrics(base_pred.unsqueeze(0), target.unsqueeze(0))
    refined_metrics = compute_stft_batch_metrics(refined.unsqueeze(0), target.unsqueeze(0))
    base_metrics["spectral_convergence"] = float(spectral_convergence_loss(base_pred.unsqueeze(0), target.unsqueeze(0)).item())
    refined_metrics["spectral_convergence"] = float(spectral_convergence_loss(refined.unsqueeze(0), target.unsqueeze(0)).item())

    metadata: dict[str, Any] = {
        "subset_name": args.subset_name,
        "split": args.split,
        "sample_index": args.sample_index,
        "window_id": sample["window_id"],
        "piece_id": sample["piece_id"],
        "composer": sample["composer"],
        "title": sample["title"],
        "start_sec": start_sec,
        "clip_seconds": clip_seconds,
        "n_fft": args.n_fft,
        "n_freq_bins": n_freq_bins,
        "n_iter": args.n_iter,
        "normalization": args.normalization,
        "prediction_cache_name": args.prediction_cache_name,
        "refiner_checkpoint": args.refiner_checkpoint,
        "base_metrics": base_metrics,
        "refined_metrics": refined_metrics,
        "audio_outputs": audio_outputs,
    }

    with (out_dir / "texture_refiner_render_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 80)
    print("Rendered TextureNet-lite audio")
    print("=" * 80)
    print("output_dir:", out_dir)
    print("window_id:", sample["window_id"])
    print("piece:", sample["composer"], "-", sample["title"])
    print()
    print("Base metrics:", base_metrics)
    print("Refined metrics:", refined_metrics)
    print()
    for k, v in audio_outputs.items():
        print(k, "->", v["wav"])
    print("=" * 80)


if __name__ == "__main__":
    main()
