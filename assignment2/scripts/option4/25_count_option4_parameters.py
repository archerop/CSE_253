from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.shared.config import SAMPLE_RATE, MIDI_LOW
from app.option4.audio_models import (
    LinearProjectedStftResidualCNN,
    LinearProjectedStftUNet,
)
from app.option4.texture_refiner import MultiBandTextureRefiner


def safe_load(path: str | Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def top_level_breakdown(model: torch.nn.Module) -> dict[str, int]:
    result = {}
    for name, module in model.named_children():
        result[name] = sum(p.numel() for p in module.parameters())
    return result


def build_enhanced_unet(checkpoint_path: Path):
    ckpt = safe_load(checkpoint_path)
    args = ckpt.get("args", {})

    model = LinearProjectedStftUNet(
        input_channels=6,
        n_freq_bins=int(args.get("n_freq_bins", 513)),
        sample_rate=SAMPLE_RATE,
        n_fft=int(args.get("n_fft", 1024)),
        midi_low=MIDI_LOW,
        base_channels=int(args.get("base_channels", 24)),
        blocks_per_level=int(args.get("blocks_per_level", 2)),
        refinement_blocks=int(args.get("refinement_blocks", 2)),
        dropout=float(args.get("dropout", 0.05)),
        condition_strength=float(args.get("condition_strength", 1.0)),
    )

    model.load_state_dict(ckpt["model_state_dict"])
    return model, args


def build_basic_unet(checkpoint_path: Path):
    ckpt = safe_load(checkpoint_path)
    args = ckpt.get("args", {})

    model = LinearProjectedStftUNet(
        input_channels=3,
        n_freq_bins=int(args.get("n_freq_bins", 513)),
        sample_rate=SAMPLE_RATE,
        n_fft=int(args.get("n_fft", 1024)),
        midi_low=MIDI_LOW,
        base_channels=int(args.get("base_channels", 24)),
        blocks_per_level=int(args.get("blocks_per_level", 2)),
        refinement_blocks=int(args.get("refinement_blocks", 2)),
        dropout=float(args.get("dropout", 0.05)),
        condition_strength=float(args.get("condition_strength", 1.0)),
    )

    model.load_state_dict(ckpt["model_state_dict"])
    return model, args


def build_enhanced_texture_refiner(checkpoint_path: Path):
    ckpt = safe_load(checkpoint_path)
    args = ckpt.get("args", {})

    feature_mode = args.get("feature_mode", "enhanced")
    condition_channels = 6 if feature_mode == "enhanced" else 3

    model = MultiBandTextureRefiner(
        n_freq_bins=513,
        sample_rate=SAMPLE_RATE,
        n_fft=int(args.get("n_fft", 1024)),
        midi_low=MIDI_LOW,
        n_bands=int(args.get("n_bands", 8)),
        hidden_channels=int(args.get("hidden_channels", 32)),
        num_blocks_per_band=int(args.get("num_blocks_per_band", 3)),
        dropout=float(args.get("dropout", 0.05)),
        residual_scale=float(args.get("residual_scale", 0.2)),
        condition_strength=float(args.get("condition_strength", 1.0)),
        condition_channels=condition_channels,
        use_condition=not bool(args.get("no_condition", False)),
    )

    model.load_state_dict(ckpt["model_state_dict"])
    return model, args


def main() -> None:
    paths = {
        "basic_stft_unet": Path(
            "outputs/option4/checkpoints/stft_unet_small_weighted_energy_onset_nfft1024_best.pt"
        ),
        "enhanced_stft_unet": Path(
            "outputs/option4/checkpoints/enhanced_stft_unet_small_weighted_energy_onset_sc0.05_nfft1024_best.pt"
        ),
        "enhanced_texture_refiner": Path(
            "outputs/option4/checkpoints/enhanced_texture_refiner_small_weighted_energy_onset_sc0.05_bands8_nfft1024_best.pt"
        ),
    }

    rows = []

    if paths["basic_stft_unet"].exists():
        model, args = build_basic_unet(paths["basic_stft_unet"])
        total, trainable = count_params(model)
        rows.append(
            {
                "model": "basic_stft_unet",
                "checkpoint": str(paths["basic_stft_unet"]),
                "input_channels": 3,
                "total_params": total,
                "trainable_params": trainable,
            }
        )

    if paths["enhanced_stft_unet"].exists():
        model, args = build_enhanced_unet(paths["enhanced_stft_unet"])
        total, trainable = count_params(model)
        rows.append(
            {
                "model": "enhanced_stft_unet",
                "checkpoint": str(paths["enhanced_stft_unet"]),
                "input_channels": 6,
                "total_params": total,
                "trainable_params": trainable,
            }
        )

        print("\n=== enhanced_stft_unet top-level parameter breakdown ===")
        for name, value in top_level_breakdown(model).items():
            print(f"{name:24s} {value:,}")

    if paths["enhanced_texture_refiner"].exists():
        model, args = build_enhanced_texture_refiner(paths["enhanced_texture_refiner"])
        total, trainable = count_params(model)
        rows.append(
            {
                "model": "enhanced_texture_refiner",
                "checkpoint": str(paths["enhanced_texture_refiner"]),
                "input_channels": "1 initial + 6 condition",
                "total_params": total,
                "trainable_params": trainable,
            }
        )

        print("\n=== enhanced_texture_refiner top-level parameter breakdown ===")
        for name, value in top_level_breakdown(model).items():
            print(f"{name:24s} {value:,}")

    df = pd.DataFrame(rows)
    out = Path("outputs/option4/metrics/option4_parameter_counts.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("\n=== parameter counts ===")
    print(df.to_string(index=False))
    print(f"\nSaved: {out}")

    if {"basic_stft_unet", "enhanced_stft_unet"}.issubset(set(df["model"])):
        basic = int(df.loc[df["model"] == "basic_stft_unet", "total_params"].iloc[0])
        enhanced = int(df.loc[df["model"] == "enhanced_stft_unet", "total_params"].iloc[0])
        print()
        print("Enhanced U-Net extra params over basic U-Net:", enhanced - basic)


if __name__ == "__main__":
    main()
