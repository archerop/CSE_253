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
    SAMPLE_RATE,
    MIDI_LOW,
    OPTION4_OUTPUT_DIR,
)
from app.option4.stft_cached_dataset import option4_stft_cache_dir
from app.option4.stft_refinement_dataset import (
    CachedStftRefinementDataset,
    option4_stft_prediction_cache_dir,
)
from app.option4.texture_refiner import MultiBandTextureRefiner
from app.option4.stft_metrics import (
    MetricAverager,
    compute_stft_batch_metrics,
    composite_stft_loss,
    spectral_convergence_loss,
)


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
    train_stft_cache = option4_stft_cache_dir(args.subset_name, "train", args.n_fft)
    val_stft_cache = option4_stft_cache_dir(args.subset_name, "validation", args.n_fft)

    train_pred_cache = option4_stft_prediction_cache_dir(
        subset_name=args.subset_name,
        split="train",
        n_fft=args.n_fft,
        prediction_cache_name=args.prediction_cache_name,
    )
    val_pred_cache = option4_stft_prediction_cache_dir(
        subset_name=args.subset_name,
        split="validation",
        n_fft=args.n_fft,
        prediction_cache_name=args.prediction_cache_name,
    )

    train_dataset = CachedStftRefinementDataset(
        stft_cache_dir=train_stft_cache,
        prediction_cache_dir=train_pred_cache,
    )

    if args.overfit_n_samples is not None:
        n = min(args.overfit_n_samples, len(train_dataset))
        small = Subset(train_dataset, list(range(n)))
        return small, small

    val_dataset = CachedStftRefinementDataset(
        stft_cache_dir=val_stft_cache,
        prediction_cache_dir=val_pred_cache,
    )

    return train_dataset, val_dataset


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
def compute_metrics_with_sc(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    metrics = compute_stft_batch_metrics(pred, target)
    metrics["spectral_convergence"] = float(spectral_convergence_loss(pred, target).item())
    return metrics


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
        target = batch["target"].to(device, non_blocking=True)
        initial_pred = batch["initial_pred"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            refined = model(piano_roll, initial_pred)
            loss = compute_loss(refined, target, args)

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

        batch_metrics = compute_metrics_with_sc(refined.detach(), target.detach())
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
    base_avg = MetricAverager()

    loss_total = 0.0
    base_loss_total = 0.0
    count = 0

    progress = tqdm(loader, desc=f"eval {split_name}", leave=False)

    for batch_idx, batch in enumerate(progress):
        if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
            break

        piano_roll = batch["piano_roll"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        initial_pred = batch["initial_pred"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            refined = model(piano_roll, initial_pred)
            loss = compute_loss(refined, target, args)
            base_loss = compute_loss(initial_pred, target, args)

        batch_size = piano_roll.shape[0]
        count += batch_size
        loss_total += float(loss.item()) * batch_size
        base_loss_total += float(base_loss.item()) * batch_size

        refined_metrics = compute_metrics_with_sc(refined.detach(), target.detach())
        base_metrics = compute_metrics_with_sc(initial_pred.detach(), target.detach())

        avg.update(refined_metrics, n=batch_size)
        base_avg.update(base_metrics, n=batch_size)

        progress.set_postfix({"loss": f"{loss.item():.5f}"})

    metrics = avg.compute()
    metrics["loss"] = loss_total / max(1, count)

    base_metrics = base_avg.compute()
    for key, value in base_metrics.items():
        metrics[f"base_{key}"] = value
    metrics["base_loss"] = base_loss_total / max(1, count)

    return metrics


def get_monitor_value(
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    monitor: str,
) -> float:
    if monitor.startswith("train_"):
        return float(train_metrics[monitor.removeprefix("train_")])
    if monitor.startswith("val_"):
        return float(val_metrics[monitor.removeprefix("val_")])
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
            "model_name": "MultiBandTextureRefiner",
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
    ax.plot(history_df["epoch"], history_df["val_loss"], label="refined val loss")
    ax.plot(history_df["epoch"], history_df["val_base_loss"], label="base U-Net val loss")

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("TextureNet-lite refinement curve")
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
    target = batch["target"].to(device)
    initial_pred = batch["initial_pred"].to(device)

    with torch.cuda.amp.autocast(enabled=use_amp):
        refined = model(piano_roll, initial_pred)

    piano_roll_0 = piano_roll[0].detach().cpu()
    target_0 = target[0].detach().cpu()
    base_0 = initial_pred[0].detach().cpu()
    refined_0 = refined[0].detach().cpu()

    err_base = torch.abs(base_0 - target_0)
    err_refined = torch.abs(refined_0 - target_0)

    vmax = float(torch.quantile(target_0, 0.99).item())
    vmax = max(vmax, 1e-6)

    err_vmax = float(torch.quantile(torch.cat([err_base.flatten(), err_refined.flatten()]), 0.99).item())
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
        (base_0.clamp_min(0.0), "Base U-Net prediction", 0.0, vmax),
        (piano_roll_0[1].T, "Input onsets", None, None),
        (refined_0.clamp_min(0.0), "Refined prediction", 0.0, vmax),
        (refined_0 - base_0, "Refinement residual", -err_vmax, err_vmax),
        (err_base, "|Base - target|", 0.0, err_vmax),
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
        description="Train TextureNet-lite refinement on precomputed STFT U-Net predictions."
    )

    parser.add_argument("--subset-name", type=str, default="small")
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--prediction-cache-name", type=str, default="stft_unet_small_best")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n-bands", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--num-blocks-per-band", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--residual-scale", type=float, default=0.2)
    parser.add_argument("--condition-strength", type=float, default=1.0)
    parser.add_argument("--no-condition", action="store_true")

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

    parser.add_argument("--overfit-n-samples", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)

    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--early-stop-min-delta", type=float, default=5e-5)
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

    model = MultiBandTextureRefiner(
        n_freq_bins=n_freq_bins,
        sample_rate=SAMPLE_RATE,
        n_fft=args.n_fft,
        midi_low=MIDI_LOW,
        n_bands=args.n_bands,
        hidden_channels=args.hidden_channels,
        num_blocks_per_band=args.num_blocks_per_band,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
        condition_strength=args.condition_strength,
        use_condition=not args.no_condition,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    tag = (
        f"texture_refiner_{args.subset_name}_"
        f"{args.loss_mode}_sc{args.spectral_convergence_weight:g}_"
        f"bands{args.n_bands}_nfft{args.n_fft}"
    )

    if args.overfit_n_samples is not None:
        tag += f"_overfit{args.overfit_n_samples}"

    print("=" * 80)
    print("Step 14A: TextureNet-lite refinement")
    print("=" * 80)
    print(f"subset_name:                  {args.subset_name}")
    print(f"prediction_cache_name:        {args.prediction_cache_name}")
    print(f"n_fft:                        {args.n_fft}")
    print(f"n_freq_bins:                  {n_freq_bins}")
    print(f"train size:                   {len(train_dataset)}")
    print(f"val size:                     {len(val_dataset)}")
    print(f"device:                       {device}")
    print(f"use_amp:                      {use_amp}")
    print(f"batch_size:                   {args.batch_size}")
    print(f"epochs:                       {args.epochs}")
    print(f"n_bands:                      {args.n_bands}")
    print(f"hidden_channels:              {args.hidden_channels}")
    print(f"num_blocks_per_band:          {args.num_blocks_per_band}")
    print(f"residual_scale:               {args.residual_scale}")
    print(f"use_condition:                {not args.no_condition}")
    print(f"loss_mode:                    {args.loss_mode}")
    print(f"spectral_convergence_weight:  {args.spectral_convergence_weight}")
    print(f"parameters:                   {num_params:,}")
    print(f"trainable:                    {num_trainable:,}")
    print()

    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers)

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
            f"train_l1={train_metrics['stft_l1']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_base_loss={val_metrics['base_loss']:.6f} | "
            f"val_l1={val_metrics['stft_l1']:.6f} | "
            f"base_l1={val_metrics['base_stft_l1']:.6f} | "
            f"val_sc={val_metrics['spectral_convergence']:.6f} | "
            f"base_sc={val_metrics['base_spectral_convergence']:.6f}"
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

    best_state = torch.load(best_checkpoint, map_location=device, weights_only=False)
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
        "model": "texture_refiner",
        "subset": args.subset_name,
        "split": "validation" if args.overfit_n_samples is None else "overfit_train_subset",
        "checkpoint": str(best_checkpoint),
        "n_fft": args.n_fft,
        "n_freq_bins": n_freq_bins,
        "prediction_cache_name": args.prediction_cache_name,
        "loss_mode": args.loss_mode,
        "weighted_alpha": args.weighted_alpha,
        "energy_weight": args.energy_weight,
        "onset_weight": args.onset_weight,
        "spectral_convergence_weight": args.spectral_convergence_weight,
        "n_bands": args.n_bands,
        "hidden_channels": args.hidden_channels,
        "num_blocks_per_band": args.num_blocks_per_band,
        "residual_scale": args.residual_scale,
        "use_condition": not args.no_condition,
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
        title=f"TextureNet-lite refinement ({args.subset_name})",
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
