from __future__ import annotations

from pathlib import Path
import argparse
import random
import sys
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.shared.config import (
    N_MELS,
    MIDI_LOW,
    FMIN,
    FMAX,
    WINDOW_INDEX_CACHE_DIR,
    OPTION4_OUTPUT_DIR,
)
from app.option4.audio_models import (
    StrongContourNetLiteUNet,
    count_parameters,
)
from app.option4.cached_dataset import (
    CachedOption4MidiToAudioDataset,
    option4_cache_dir,
)
from app.option4.option4_dataset import Option4MidiToAudioDataset
from app.option4.metrics import (
    MetricAverager,
    compute_batch_metrics,
    composite_spectrogram_loss,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_fmax() -> float:
    if FMAX is None:
        return 11025.0
    return float(FMAX)


def get_index_csv(subset_name: str, split: str) -> Path:
    return WINDOW_INDEX_CACHE_DIR / f"option4_{subset_name}_{split}_windows.csv"


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
    if args.use_cache:
        train_dataset = CachedOption4MidiToAudioDataset(
            cache_dir=option4_cache_dir(args.subset_name, "train")
        )

        if args.overfit_n_samples is not None:
            n = min(args.overfit_n_samples, len(train_dataset))
            small = Subset(train_dataset, list(range(n)))
            return small, small

        val_dataset = CachedOption4MidiToAudioDataset(
            cache_dir=option4_cache_dir(args.subset_name, "validation")
        )
        return train_dataset, val_dataset

    train_csv = get_index_csv(args.subset_name, "train")
    val_csv = get_index_csv(args.subset_name, "validation")

    train_dataset = Option4MidiToAudioDataset(index_csv=train_csv, return_audio=False)

    if args.overfit_n_samples is not None:
        n = min(args.overfit_n_samples, len(train_dataset))
        small = Subset(train_dataset, list(range(n)))
        return small, small

    val_dataset = Option4MidiToAudioDataset(index_csv=val_csv, return_audio=False)
    return train_dataset, val_dataset


def compute_loss(pred: torch.Tensor, target: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    return composite_spectrogram_loss(
        pred=pred,
        target=target,
        loss_mode=args.loss_mode,
        weighted_alpha=args.weighted_alpha,
        energy_weight=args.energy_weight,
        onset_weight=args.onset_weight,
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()

    avg = MetricAverager()
    loss_total = 0.0
    count = 0

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)

    for batch_idx, batch in enumerate(progress):
        if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
            break

        piano_roll = batch["piano_roll"].to(device, non_blocking=True)
        target = batch["log_mel"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(piano_roll)
            loss = compute_loss(pred, target, args)

        if use_amp:
            scaler.scale(loss).backward()

            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

            optimizer.step()

        batch_size = piano_roll.shape[0]
        count += batch_size
        loss_total += float(loss.item()) * batch_size

        batch_metrics = compute_batch_metrics(pred.detach(), target.detach())
        avg.update(batch_metrics, n=batch_size)

        progress.set_postfix({"loss": f"{loss.item():.5f}"})

    metrics = avg.compute()
    metrics["loss"] = loss_total / max(1, count)
    return metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    use_amp: bool,
    split_name: str,
) -> Dict[str, float]:
    model.eval()

    avg = MetricAverager()
    loss_total = 0.0
    count = 0

    progress = tqdm(loader, desc=f"eval {split_name}", leave=False)

    for batch_idx, batch in enumerate(progress):
        if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
            break

        piano_roll = batch["piano_roll"].to(device, non_blocking=True)
        target = batch["log_mel"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(piano_roll)
            loss = compute_loss(pred, target, args)

        batch_size = piano_roll.shape[0]
        count += batch_size
        loss_total += float(loss.item()) * batch_size

        batch_metrics = compute_batch_metrics(pred.detach(), target.detach())
        avg.update(batch_metrics, n=batch_size)

        progress.set_postfix({"loss": f"{loss.item():.5f}"})

    metrics = avg.compute()
    metrics["loss"] = loss_total / max(1, count)
    return metrics


def get_monitor_value(
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    monitor: str,
) -> float:
    if monitor.startswith("train_"):
        key = monitor.removeprefix("train_")
        return float(train_metrics[key])

    if monitor.startswith("val_"):
        key = monitor.removeprefix("val_")
        return float(val_metrics[key])

    raise ValueError(f"Invalid monitor: {monitor}")


def is_improvement(current: float, best: float, min_delta: float) -> bool:
    return current < (best - min_delta)


def save_checkpoint(
    output_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_monitor_value: float,
    args: argparse.Namespace,
    history: list[dict],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_name": "StrongContourNetLiteUNet",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_monitor_value": best_monitor_value,
            "args": vars(args),
            "history": history,
        },
        output_path,
    )


def plot_loss_curve(history_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_df["epoch"], history_df["train_logmel_l1"], label="train logmel L1")
    ax.plot(history_df["epoch"], history_df["val_logmel_l1"], label="val logmel L1")

    ax.set_xlabel("epoch")
    ax.set_ylabel("log-mel L1")
    ax.set_title("Strong ContourNet-lite U-Net training curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_prediction_figure(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
    args: argparse.Namespace,
    use_amp: bool,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    batch = next(iter(loader))

    piano_roll = batch["piano_roll"].to(device)
    target = batch["log_mel"].to(device)

    with torch.cuda.amp.autocast(enabled=use_amp):
        pred = model(piano_roll)

    piano_roll_0 = piano_roll[0].detach().cpu()
    target_0 = target[0].detach().cpu()
    pred_raw = pred[0].detach().cpu()
    pred_show = pred_raw.clamp_min(0.0)
    error = torch.abs(pred_raw - target_0)

    vmax = float(torch.quantile(target_0, 0.99).item())
    vmax = max(vmax, 1e-6)

    err_vmax = float(torch.quantile(error, 0.99).item())
    err_vmax = max(err_vmax, 1e-6)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(16, 8),
        constrained_layout=True,
    )

    images = [
        (piano_roll_0[0].T, "Input active notes", None, None),
        (target_0, "Target log-mel", 0.0, vmax),
        (pred_show, "Predicted log-mel, clamped", 0.0, vmax),
        (piano_roll_0[1].T, "Input onsets", None, None),
        (error, "|Prediction - target|", 0.0, err_vmax),
        (target_0 - pred_raw, "Target - raw prediction", -err_vmax, err_vmax),
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
        ax.set_ylabel("pitch / mel bin")
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


def maybe_print_comparison(subset_name: str, loss_mode: str, val_metrics: Dict[str, float]) -> None:
    baseline_path = OPTION4_OUTPUT_DIR / "metrics" / f"baseline_metrics_{subset_name}_validation.csv"
    residual_path = OPTION4_OUTPUT_DIR / "metrics" / f"mel_residual_cnn_{subset_name}_{loss_mode}_eval.csv"
    simple_unet_path = OPTION4_OUTPUT_DIR / "metrics" / f"contour_unet_{subset_name}_{loss_mode}_eval.csv"

    print()
    print("Strong ContourNet-lite U-Net validation:")
    print(
        f"logmel_l1={val_metrics['logmel_l1']:.6f}, "
        f"logmel_mse={val_metrics['logmel_mse']:.6f}, "
        f"energy_l1={val_metrics['energy_l1']:.6f}, "
        f"onset_l1={val_metrics['onset_l1']:.6f}, "
        f"loss={val_metrics['loss']:.6f}"
    )

    if baseline_path.exists():
        print()
        print("Non-learned baselines:")
        df = pd.read_csv(baseline_path)
        print(df[["baseline", "logmel_l1", "logmel_mse", "energy_l1", "onset_l1"]].to_string(index=False))

    if residual_path.exists():
        print()
        print("Mel-projected Residual CNN on same subset/loss:")
        df = pd.read_csv(residual_path)
        cols = ["model", "logmel_l1", "logmel_mse", "energy_l1", "onset_l1", "loss"]
        print(df[cols].to_string(index=False))

    if simple_unet_path.exists():
        print()
        print("Earlier ContourNet-lite U-Net on same subset/loss:")
        df = pd.read_csv(simple_unet_path)
        cols = ["model", "logmel_l1", "logmel_mse", "energy_l1", "onset_l1", "loss"]
        print(df[cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 10: Train Strong ContourNet-lite U-Net."
    )

    parser.add_argument("--subset-name", type=str, default="debug")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--blocks-per-level", type=int, default=2)
    parser.add_argument("--refinement-blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--condition-strength", type=float, default=1.0)

    parser.add_argument(
        "--loss-mode",
        type=str,
        default="weighted_energy_onset",
        choices=["l1", "weighted_l1", "weighted_energy_onset"],
    )
    parser.add_argument("--weighted-alpha", type=float, default=4.0)
    parser.add_argument("--energy-weight", type=float, default=0.005)
    parser.add_argument("--onset-weight", type=float, default=0.02)

    parser.add_argument("--overfit-n-samples", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)

    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--early-stop-monitor",
        type=str,
        default="val_loss",
        choices=[
            "val_loss",
            "val_logmel_l1",
            "val_energy_l1",
            "val_onset_l1",
            "train_loss",
            "train_logmel_l1",
        ],
    )

    args = parser.parse_args()

    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = get_device()
    use_amp = bool(args.amp and device.type == "cuda")

    train_dataset, val_dataset = build_datasets(args)

    tag = f"strong_contour_unet_{args.subset_name}_{args.loss_mode}"
    if args.overfit_n_samples is not None:
        tag += f"_overfit{args.overfit_n_samples}"

    print("=" * 80)
    print("Step 10: Strong ContourNet-lite U-Net")
    print("=" * 80)
    print(f"subset_name:         {args.subset_name}")
    print(f"use_cache:           {args.use_cache}")
    print(f"train size:          {len(train_dataset)}")
    print(f"val size:            {len(val_dataset)}")
    print(f"overfit_n_samples:   {args.overfit_n_samples}")
    print(f"device:              {device}")
    print(f"use_amp:             {use_amp}")
    print(f"batch_size:          {args.batch_size}")
    print(f"epochs:              {args.epochs}")
    print(f"lr:                  {args.lr}")
    print(f"loss_mode:           {args.loss_mode}")
    print(f"base_channels:       {args.base_channels}")
    print(f"blocks_per_level:    {args.blocks_per_level}")
    print(f"refinement_blocks:   {args.refinement_blocks}")
    print(f"condition_strength:  {args.condition_strength}")
    print(f"early_stop_monitor:  {args.early_stop_monitor}")
    print(f"early_stop_patience: {args.early_stop_patience}")
    print()

    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = make_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = StrongContourNetLiteUNet(
        input_channels=3,
        n_mels=N_MELS,
        midi_low=MIDI_LOW,
        fmin=FMIN,
        fmax=get_fmax(),
        base_channels=args.base_channels,
        blocks_per_level=args.blocks_per_level,
        refinement_blocks=args.refinement_blocks,
        dropout=args.dropout,
        condition_strength=args.condition_strength,
    ).to(device)

    summary = count_parameters(model)
    print("Model summary:")
    print(f"  name:       {summary.name}")
    print(f"  parameters: {summary.num_parameters:,}")
    print(f"  trainable:  {summary.num_trainable_parameters:,}")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
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

    history: list[dict] = []
    best_monitor_value = float("inf")
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            use_amp=use_amp,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            args=args,
            use_amp=use_amp,
            split_name="validation",
        )

        row = {"epoch": epoch}
        for key, value in train_metrics.items():
            row[f"train_{key}"] = value
        for key, value in val_metrics.items():
            row[f"val_{key}"] = value
        history.append(row)

        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.6f} | "
            f"train_l1={train_metrics['logmel_l1']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_l1={val_metrics['logmel_l1']:.6f} | "
            f"val_mse={val_metrics['logmel_mse']:.6f} | "
            f"val_energy={val_metrics['energy_l1']:.6f} | "
            f"val_onset={val_metrics['onset_l1']:.6f}"
        )

        history_df = pd.DataFrame(history)
        history_df.to_csv(history_csv, index=False)
        plot_loss_curve(history_df, loss_curve_path)

        current_monitor = get_monitor_value(
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            monitor=args.early_stop_monitor,
        )

        if is_improvement(current_monitor, best_monitor_value, args.early_stop_min_delta):
            best_monitor_value = current_monitor
            best_epoch = epoch
            bad_epochs = 0

            save_checkpoint(
                output_path=best_checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_monitor_value=best_monitor_value,
                args=args,
                history=history,
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
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_monitor_value=best_monitor_value,
            args=args,
            history=history,
        )

        if args.early_stop_patience > 0 and bad_epochs >= args.early_stop_patience:
            print()
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch was {best_epoch} with "
                f"{args.early_stop_monitor}={best_monitor_value:.6f}."
            )
            break

    best_state = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(best_state["model_state_dict"])

    final_val_metrics = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        args=args,
        use_amp=use_amp,
        split_name="validation-best",
    )

    eval_row = {
        "model": "strong_contour_unet",
        "subset": args.subset_name,
        "split": "validation" if args.overfit_n_samples is None else "overfit_train_subset",
        "checkpoint": str(best_checkpoint),
        "loss_mode": args.loss_mode,
        "weighted_alpha": args.weighted_alpha,
        "energy_weight": args.energy_weight,
        "onset_weight": args.onset_weight,
        "condition_strength": args.condition_strength,
        "base_channels": args.base_channels,
        "blocks_per_level": args.blocks_per_level,
        "refinement_blocks": args.refinement_blocks,
        **final_val_metrics,
    }

    pd.DataFrame([eval_row]).to_csv(eval_csv, index=False)

    save_prediction_figure(
        model=model,
        loader=val_loader,
        device=device,
        output_path=prediction_fig_path,
        args=args,
        use_amp=use_amp,
        title=f"Strong ContourNet-lite U-Net prediction example ({args.subset_name}, {args.loss_mode})",
    )

    if args.overfit_n_samples is None:
        maybe_print_comparison(args.subset_name, args.loss_mode, final_val_metrics)

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
