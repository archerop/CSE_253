from __future__ import annotations

from pathlib import Path
import argparse
import random
import sys
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.shared.config import (
    SAMPLE_RATE,
    MIDI_LOW,
    OPTION4_OUTPUT_DIR,
)
from app.option4.audio_models import LinearProjectedStftUNet
from app.option4.texture_refiner import MultiBandTextureRefiner
from app.option4.stft_cached_dataset import (
    CachedOption4StftDataset,
    option4_stft_cache_dir,
)
from app.option4.stft_metrics import (
    MetricAverager,
    compute_stft_batch_metrics,
    composite_stft_loss,
    spectral_convergence_loss,
)


def safe_load(path: str | Path, map_location):
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def build_datasets(args: argparse.Namespace):
    train_dataset = CachedOption4StftDataset(
        option4_stft_cache_dir(args.subset_name, "train", args.n_fft)
    )
    val_dataset = CachedOption4StftDataset(
        option4_stft_cache_dir(args.subset_name, "validation", args.n_fft)
    )
    return train_dataset, val_dataset


def build_unet_from_checkpoint(
    checkpoint_path: str | Path,
    n_freq_bins: int,
    n_fft: int,
    device: torch.device,
    unet_dropout: float,
) -> Tuple[LinearProjectedStftUNet, dict]:
    ckpt = safe_load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})

    model = LinearProjectedStftUNet(
        input_channels=3,
        n_freq_bins=n_freq_bins,
        sample_rate=SAMPLE_RATE,
        n_fft=n_fft,
        midi_low=MIDI_LOW,
        base_channels=int(ckpt_args.get("base_channels", 24)),
        blocks_per_level=int(ckpt_args.get("blocks_per_level", 2)),
        refinement_blocks=int(ckpt_args.get("refinement_blocks", 2)),
        dropout=unet_dropout,
        condition_strength=float(ckpt_args.get("condition_strength", 1.0)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt_args


def build_refiner_from_checkpoint(
    checkpoint_path: str | Path,
    n_freq_bins: int,
    n_fft: int,
    device: torch.device,
) -> Tuple[MultiBandTextureRefiner, dict]:
    ckpt = safe_load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})

    model = MultiBandTextureRefiner(
        n_freq_bins=n_freq_bins,
        sample_rate=SAMPLE_RATE,
        n_fft=n_fft,
        midi_low=MIDI_LOW,
        n_bands=int(ckpt_args.get("n_bands", 8)),
        hidden_channels=int(ckpt_args.get("hidden_channels", 32)),
        num_blocks_per_band=int(ckpt_args.get("num_blocks_per_band", 3)),
        dropout=float(ckpt_args.get("dropout", 0.05)),
        residual_scale=float(ckpt_args.get("residual_scale", 0.2)),
        condition_strength=float(ckpt_args.get("condition_strength", 1.0)),
        use_condition=bool(ckpt_args.get("use_condition", True)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt_args


def freeze_all(module: torch.nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: torch.nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def configure_unet_trainable(unet: LinearProjectedStftUNet, scope: str) -> None:
    """
    Fine-tune only the late part of U-Net by default.

    scope:
      head:
        output_head only
      decoder:
        up3/up2/up1/refinement/output_head
      decoder_bottleneck:
        bottleneck_dilated + decoder + refinement + output_head
      all:
        all U-Net parameters
    """
    freeze_all(unet)

    if scope == "head":
        unfreeze_module(unet.output_head)
        return

    if scope == "decoder":
        for name in ["up3", "up2", "up1", "refinement", "output_head"]:
            unfreeze_module(getattr(unet, name))
        return

    if scope == "decoder_bottleneck":
        for name in [
            "bottleneck_dilated",
            "up3",
            "up2",
            "up1",
            "refinement",
            "output_head",
        ]:
            unfreeze_module(getattr(unet, name))
        return

    if scope == "all":
        unfreeze_module(unet)
        return

    raise ValueError(f"Unknown unfreeze scope: {scope}")


def count_trainable(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_total(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def compute_loss(
    refined: torch.Tensor,
    target: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    base = composite_stft_loss(
        pred=refined,
        target=target,
        loss_mode=args.loss_mode,
        weighted_alpha=args.weighted_alpha,
        energy_weight=args.energy_weight,
        onset_weight=args.onset_weight,
    )

    if args.spectral_convergence_weight > 0:
        sc = spectral_convergence_loss(refined, target)
        return base + args.spectral_convergence_weight * sc

    return base


@torch.no_grad()
def compute_metrics_with_sc(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    metrics = compute_stft_batch_metrics(pred, target)
    metrics["spectral_convergence"] = float(spectral_convergence_loss(pred, target).item())
    return metrics


def train_one_epoch(
    unet: torch.nn.Module,
    refiner: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    use_amp: bool,
) -> Dict[str, float]:
    unet.train()
    refiner.train()

    avg = MetricAverager()
    initial_avg = MetricAverager()

    loss_total = 0.0
    initial_loss_total = 0.0
    count = 0

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)

    for batch_idx, batch in enumerate(progress):
        if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
            break

        piano_roll = batch["piano_roll"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            initial = unet(piano_roll)
            refined = refiner(piano_roll, initial)
            loss = compute_loss(refined, target, args)
            initial_loss = compute_loss(initial, target, args)

        if use_amp:
            scaler.scale(loss).backward()

            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"]],
                    args.grad_clip_norm,
                )

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"]],
                    args.grad_clip_norm,
                )

            optimizer.step()

        b = piano_roll.shape[0]
        count += b
        loss_total += float(loss.item()) * b
        initial_loss_total += float(initial_loss.item()) * b

        refined_metrics = compute_metrics_with_sc(refined.detach(), target.detach())
        initial_metrics = compute_metrics_with_sc(initial.detach(), target.detach())

        avg.update(refined_metrics, n=b)
        initial_avg.update(initial_metrics, n=b)

        progress.set_postfix({"loss": f"{loss.item():.5f}"})

    metrics = avg.compute()
    metrics["loss"] = loss_total / max(1, count)

    initial_metrics = initial_avg.compute()
    for k, v in initial_metrics.items():
        metrics[f"initial_{k}"] = v
    metrics["initial_loss"] = initial_loss_total / max(1, count)

    return metrics


@torch.no_grad()
def evaluate(
    unet: torch.nn.Module,
    refiner: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    use_amp: bool,
    split_name: str,
) -> Dict[str, float]:
    unet.eval()
    refiner.eval()

    avg = MetricAverager()
    initial_avg = MetricAverager()

    loss_total = 0.0
    initial_loss_total = 0.0
    count = 0

    progress = tqdm(loader, desc=f"eval {split_name}", leave=False)

    for batch_idx, batch in enumerate(progress):
        if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
            break

        piano_roll = batch["piano_roll"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            initial = unet(piano_roll)
            refined = refiner(piano_roll, initial)
            loss = compute_loss(refined, target, args)
            initial_loss = compute_loss(initial, target, args)

        b = piano_roll.shape[0]
        count += b
        loss_total += float(loss.item()) * b
        initial_loss_total += float(initial_loss.item()) * b

        refined_metrics = compute_metrics_with_sc(refined.detach(), target.detach())
        initial_metrics = compute_metrics_with_sc(initial.detach(), target.detach())

        avg.update(refined_metrics, n=b)
        initial_avg.update(initial_metrics, n=b)

        progress.set_postfix({"loss": f"{loss.item():.5f}"})

    metrics = avg.compute()
    metrics["loss"] = loss_total / max(1, count)

    initial_metrics = initial_avg.compute()
    for k, v in initial_metrics.items():
        metrics[f"initial_{k}"] = v
    metrics["initial_loss"] = initial_loss_total / max(1, count)

    return metrics


def get_monitor_value(train_metrics: Dict[str, float], val_metrics: Dict[str, float], monitor: str) -> float:
    if monitor.startswith("train_"):
        return float(train_metrics[monitor.removeprefix("train_")])
    if monitor.startswith("val_"):
        return float(val_metrics[monitor.removeprefix("val_")])
    raise ValueError(f"Invalid monitor: {monitor}")


def is_improvement(current: float, best: float, min_delta: float) -> bool:
    return current < (best - min_delta)


def save_checkpoint(
    output_path: Path,
    unet: torch.nn.Module,
    refiner: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_monitor_value: float,
    args: argparse.Namespace,
    history: list[dict],
    unet_arch_args: dict,
    refiner_arch_args: dict,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_name": "StftUNetTextureRefinerFineTune",
            "unet_state_dict": unet.state_dict(),
            "refiner_state_dict": refiner.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_monitor_value": best_monitor_value,
            "args": vars(args),
            "history": history,
            "unet_arch_args": unet_arch_args,
            "refiner_arch_args": refiner_arch_args,
        },
        output_path,
    )


def plot_loss_curve(history_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_df["epoch"], history_df["val_loss"], label="fine-tuned refined val loss")
    ax.plot(history_df["epoch"], history_df["val_initial_loss"], label="fine-tuned initial val loss")

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Decoder-only U-Net + TextureNet-lite fine-tuning")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_prediction_figure(
    unet: torch.nn.Module,
    refiner: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
    args: argparse.Namespace,
    use_amp: bool,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    unet.eval()
    refiner.eval()

    batch = next(iter(loader))

    piano_roll = batch["piano_roll"].to(device)
    target = batch["target"].to(device)

    with torch.cuda.amp.autocast(enabled=use_amp):
        initial = unet(piano_roll)
        refined = refiner(piano_roll, initial)

    piano_roll_0 = piano_roll[0].detach().cpu()
    target_0 = target[0].detach().cpu()
    initial_0 = initial[0].detach().cpu()
    refined_0 = refined[0].detach().cpu()

    err_initial = torch.abs(initial_0 - target_0)
    err_refined = torch.abs(refined_0 - target_0)

    vmax = float(torch.quantile(target_0, 0.99).item())
    vmax = max(vmax, 1e-6)

    err_vmax = float(torch.quantile(torch.cat([err_initial.flatten(), err_refined.flatten()]), 0.99).item())
    err_vmax = max(err_vmax, 1e-6)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(16, 11),
        constrained_layout=True,
    )

    images = [
        (piano_roll_0[0].T, "Input active notes", None, None),
        (target_0, "Target log-STFT", 0.0, vmax),
        (initial_0.clamp_min(0.0), "Fine-tuned U-Net initial", 0.0, vmax),
        (piano_roll_0[1].T, "Input onsets", None, None),
        (refined_0.clamp_min(0.0), "Fine-tuned refined", 0.0, vmax),
        (refined_0 - initial_0, "Texture residual", -err_vmax, err_vmax),
        (err_initial, "|Initial - target|", 0.0, err_vmax),
        (err_refined, "|Refined - target|", 0.0, err_vmax),
        (target_0 - refined_0, "Target - refined", -err_vmax, err_vmax),
    ]

    for ax, (image, subtitle, vmin, vmax_i) in zip(axes.ravel(), images):
        im = ax.imshow(
            image,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax_i,
        )
        ax.set_title(subtitle)
        ax.set_xlabel("time frame")
        ax.set_ylabel("pitch / frequency bin")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    window_id = batch["window_id"][0]
    composer = batch["composer"][0]
    piece_title = batch["title"][0]

    fig.suptitle(
        f"{title}\n{composer} — {piece_title}\nwindow_id={window_id}",
        fontsize=13,
    )

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decoder-only fine-tuning for STFT U-Net + TextureNet-lite."
    )

    parser.add_argument("--subset-name", type=str, default="small")
    parser.add_argument("--n-fft", type=int, default=1024)

    parser.add_argument(
        "--unet-checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--refiner-checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--unfreeze-scope",
        type=str,
        default="decoder",
        choices=["head", "decoder", "decoder_bottleneck", "all"],
    )

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr-unet", type=float, default=3e-4)
    parser.add_argument("--lr-refiner", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--unet-dropout", type=float, default=0.0)

    parser.add_argument(
        "--loss-mode",
        type=str,
        default="weighted_energy_onset",
        choices=["l1", "weighted_l1", "weighted_energy_onset"],
    )
    parser.add_argument("--weighted-alpha", type=float, default=4.0)
    parser.add_argument("--energy-weight", type=float, default=0.05)
    parser.add_argument("--onset-weight", type=float, default=0.05)
    parser.add_argument("--spectral-convergence-weight", type=float, default=0.05)

    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)

    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--early-stop-min-delta", type=float, default=2e-5)
    parser.add_argument(
        "--early-stop-monitor",
        type=str,
        default="val_loss",
        choices=[
            "val_loss",
            "val_stft_l1",
            "val_stft_mse",
            "val_energy_l1",
            "val_onset_l1",
            "val_spectral_convergence",
            "train_loss",
            "train_stft_l1",
        ],
    )

    args = parser.parse_args()

    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = get_device()
    use_amp = bool(args.amp and device.type == "cuda")

    train_dataset, val_dataset = build_datasets(args)
    first = train_dataset[0]
    n_freq_bins = int(first["target"].shape[0])

    unet, unet_arch_args = build_unet_from_checkpoint(
        checkpoint_path=args.unet_checkpoint,
        n_freq_bins=n_freq_bins,
        n_fft=args.n_fft,
        device=device,
        unet_dropout=args.unet_dropout,
    )

    refiner, refiner_arch_args = build_refiner_from_checkpoint(
        checkpoint_path=args.refiner_checkpoint,
        n_freq_bins=n_freq_bins,
        n_fft=args.n_fft,
        device=device,
    )

    configure_unet_trainable(unet, args.unfreeze_scope)
    for p in refiner.parameters():
        p.requires_grad = True

    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in unet.parameters() if p.requires_grad],
                "lr": args.lr_unet,
            },
            {
                "params": [p for p in refiner.parameters() if p.requires_grad],
                "lr": args.lr_refiner,
            },
        ],
        weight_decay=args.weight_decay,
    )

    tag = (
        f"finetune_unet_texture_{args.subset_name}_"
        f"{args.loss_mode}_sc{args.spectral_convergence_weight:g}_"
        f"{args.unfreeze_scope}_nfft{args.n_fft}"
    )

    checkpoint_dir = OPTION4_OUTPUT_DIR / "checkpoints"
    metrics_dir = OPTION4_OUTPUT_DIR / "metrics"
    figure_dir = OPTION4_OUTPUT_DIR / "figures"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint = checkpoint_dir / f"{tag}_best.pt"
    last_checkpoint = checkpoint_dir / f"{tag}_last.pt"
    history_csv = metrics_dir / f"{tag}_history.csv"
    eval_csv = metrics_dir / f"{tag}_eval.csv"
    loss_curve_path = figure_dir / f"{tag}_loss_curve.png"
    prediction_fig_path = figure_dir / f"{tag}_prediction_example.png"

    print("=" * 80)
    print("Step 15: Decoder-only fine-tuning")
    print("=" * 80)
    print(f"subset_name:             {args.subset_name}")
    print(f"n_fft:                   {args.n_fft}")
    print(f"n_freq_bins:             {n_freq_bins}")
    print(f"train size:              {len(train_dataset)}")
    print(f"val size:                {len(val_dataset)}")
    print(f"device:                  {device}")
    print(f"use_amp:                 {use_amp}")
    print(f"unfreeze_scope:          {args.unfreeze_scope}")
    print(f"batch_size:              {args.batch_size}")
    print(f"epochs:                  {args.epochs}")
    print(f"lr_unet:                 {args.lr_unet}")
    print(f"lr_refiner:              {args.lr_refiner}")
    print(f"loss_mode:               {args.loss_mode}")
    print(f"sc_weight:               {args.spectral_convergence_weight}")
    print(f"unet total params:       {count_total(unet):,}")
    print(f"unet trainable params:   {count_trainable(unet):,}")
    print(f"refiner total params:    {count_total(refiner):,}")
    print(f"refiner trainable params:{count_trainable(refiner):,}")
    print(f"best_checkpoint:         {best_checkpoint}")
    print("=" * 80)

    history: list[dict] = []
    best_monitor_value = float("inf")
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            unet=unet,
            refiner=refiner,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            use_amp=use_amp,
        )

        val_metrics = evaluate(
            unet=unet,
            refiner=refiner,
            loader=val_loader,
            device=device,
            args=args,
            use_amp=use_amp,
            split_name="validation",
        )

        row = {"epoch": epoch}
        for k, v in train_metrics.items():
            row[f"train_{k}"] = v
        for k, v in val_metrics.items():
            row[f"val_{k}"] = v
        history.append(row)

        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_initial_loss={val_metrics['initial_loss']:.6f} | "
            f"val_l1={val_metrics['stft_l1']:.6f} | "
            f"initial_l1={val_metrics['initial_stft_l1']:.6f} | "
            f"val_sc={val_metrics['spectral_convergence']:.6f} | "
            f"initial_sc={val_metrics['initial_spectral_convergence']:.6f}"
        )

        history_df = pd.DataFrame(history)
        history_df.to_csv(history_csv, index=False)
        plot_loss_curve(history_df, loss_curve_path)

        current_monitor = get_monitor_value(train_metrics, val_metrics, args.early_stop_monitor)

        if is_improvement(current_monitor, best_monitor_value, args.early_stop_min_delta):
            best_monitor_value = current_monitor
            best_epoch = epoch
            bad_epochs = 0

            save_checkpoint(
                output_path=best_checkpoint,
                unet=unet,
                refiner=refiner,
                optimizer=optimizer,
                epoch=epoch,
                best_monitor_value=best_monitor_value,
                args=args,
                history=history,
                unet_arch_args=unet_arch_args,
                refiner_arch_args=refiner_arch_args,
            )

            print(
                f"  saved best checkpoint: {best_checkpoint} "
                f"({args.early_stop_monitor}={best_monitor_value:.6f})"
            )
        else:
            bad_epochs += 1
            print(
                f"  no improvement: current={current_monitor:.6f}, "
                f"best={best_monitor_value:.6f}, "
                f"bad_epochs={bad_epochs}/{args.early_stop_patience}"
            )

        save_checkpoint(
            output_path=last_checkpoint,
            unet=unet,
            refiner=refiner,
            optimizer=optimizer,
            epoch=epoch,
            best_monitor_value=best_monitor_value,
            args=args,
            history=history,
            unet_arch_args=unet_arch_args,
            refiner_arch_args=refiner_arch_args,
        )

        if args.early_stop_patience > 0 and bad_epochs >= args.early_stop_patience:
            print()
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch was {best_epoch} with "
                f"{args.early_stop_monitor}={best_monitor_value:.6f}."
            )
            break

    best_state = safe_load(best_checkpoint, map_location=device)
    unet.load_state_dict(best_state["unet_state_dict"])
    refiner.load_state_dict(best_state["refiner_state_dict"])

    final_val_metrics = evaluate(
        unet=unet,
        refiner=refiner,
        loader=val_loader,
        device=device,
        args=args,
        use_amp=use_amp,
        split_name="validation-best",
    )

    eval_row = {
        "model": "finetuned_unet_texture_refiner",
        "subset": args.subset_name,
        "split": "validation",
        "checkpoint": str(best_checkpoint),
        "n_fft": args.n_fft,
        "n_freq_bins": n_freq_bins,
        "unfreeze_scope": args.unfreeze_scope,
        "loss_mode": args.loss_mode,
        "weighted_alpha": args.weighted_alpha,
        "energy_weight": args.energy_weight,
        "onset_weight": args.onset_weight,
        "spectral_convergence_weight": args.spectral_convergence_weight,
        "lr_unet": args.lr_unet,
        "lr_refiner": args.lr_refiner,
        **final_val_metrics,
    }

    pd.DataFrame([eval_row]).to_csv(eval_csv, index=False)

    save_prediction_figure(
        unet=unet,
        refiner=refiner,
        loader=val_loader,
        device=device,
        output_path=prediction_fig_path,
        args=args,
        use_amp=use_amp,
        title="Fine-tuned STFT U-Net + TextureNet-lite",
    )

    print()
    print("Saved outputs:")
    print(f"  best checkpoint: {best_checkpoint}")
    print(f"  last checkpoint: {last_checkpoint}")
    print(f"  history csv:     {history_csv}")
    print(f"  eval csv:        {eval_csv}")
    print(f"  loss curve:      {loss_curve_path}")
    print(f"  prediction fig:  {prediction_fig_path}")
    print("=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
