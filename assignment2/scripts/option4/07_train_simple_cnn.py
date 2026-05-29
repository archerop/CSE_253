from __future__ import annotations

from pathlib import Path
import argparse
import random
import sys
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.shared.config import (
    N_MELS,
    WINDOW_INDEX_CACHE_DIR,
    OPTION4_OUTPUT_DIR,
)
from app.option4.audio_models import (
    SimpleMidiToLogMelCNN,
    count_parameters,
)
from app.option4.metrics import (
    MetricAverager,
    compute_batch_metrics,
)
from app.option4.option4_dataset import make_option4_dataloader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_index_csv(subset_name: str, split: str) -> Path:
    return WINDOW_INDEX_CACHE_DIR / f"option4_{subset_name}_{split}_windows.csv"


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_batches: Optional[int],
    use_amp: bool,
    grad_clip_norm: Optional[float],
) -> Dict[str, float]:
    model.train()

    avg = MetricAverager()
    loss_total = 0.0
    count = 0

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)

    for batch_idx, batch in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break

        piano_roll = batch["piano_roll"].to(device, non_blocking=True)
        target = batch["log_mel"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(piano_roll)
            loss = F.l1_loss(pred, target)

        if use_amp:
            scaler.scale(loss).backward()

            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

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
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int],
    use_amp: bool,
    split_name: str,
) -> Dict[str, float]:
    model.eval()

    avg = MetricAverager()
    loss_total = 0.0
    count = 0

    progress = tqdm(loader, desc=f"eval {split_name}", leave=False)

    for batch_idx, batch in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break

        piano_roll = batch["piano_roll"].to(device, non_blocking=True)
        target = batch["log_mel"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(piano_roll)
            loss = F.l1_loss(pred, target)

        batch_size = piano_roll.shape[0]
        count += batch_size
        loss_total += float(loss.item()) * batch_size

        batch_metrics = compute_batch_metrics(pred.detach(), target.detach())
        avg.update(batch_metrics, n=batch_size)

        progress.set_postfix({"loss": f"{loss.item():.5f}"})

    metrics = avg.compute()
    metrics["loss"] = loss_total / max(1, count)
    return metrics


def save_checkpoint(
    output_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_l1: float,
    args: argparse.Namespace,
    history: list[dict],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_name": "SimpleMidiToLogMelCNN",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_l1": best_val_l1,
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
    ax.set_title("Simple CNN training curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_prediction_figure(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_path: Path,
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

    # Use the first example.
    piano_roll_0 = piano_roll[0].detach().cpu()
    target_0 = target[0].detach().cpu()
    pred_0_raw = pred[0].detach().cpu()
    pred_0_show = pred_0_raw.clamp_min(0.0)
    error_0 = torch.abs(pred_0_raw - target_0)

    vmax = float(torch.quantile(target_0, 0.99).item())
    vmax = max(vmax, 1e-6)

    err_vmax = float(torch.quantile(error_0, 0.99).item())
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
        (pred_0_show, "Predicted log-mel, clamped for display", 0.0, vmax),
        (piano_roll_0[1].T, "Input onsets", None, None),
        (error_0, "|Prediction - target|", 0.0, err_vmax),
        (target_0 - pred_0_raw, "Target - raw prediction", -err_vmax, err_vmax),
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


def maybe_print_baseline_comparison(
    subset_name: str,
    val_metrics: Dict[str, float],
) -> None:
    baseline_path = OPTION4_OUTPUT_DIR / "metrics" / f"baseline_metrics_{subset_name}_validation.csv"

    if not baseline_path.exists():
        print()
        print(f"No baseline metrics found at: {baseline_path}")
        print("Run Step 5 first if you want direct comparison.")
        return

    baseline_df = pd.read_csv(baseline_path)

    print()
    print("Comparison with Step 5 baselines:")
    print(baseline_df[["baseline", "logmel_l1", "logmel_mse", "energy_l1", "onset_l1"]].to_string(index=False))

    print()
    print("Simple CNN validation:")
    print(
        f"logmel_l1={val_metrics['logmel_l1']:.6f}, "
        f"logmel_mse={val_metrics['logmel_mse']:.6f}, "
        f"energy_l1={val_metrics['energy_l1']:.6f}, "
        f"onset_l1={val_metrics['onset_l1']:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 6: Train Simple CNN learned baseline for Option 4."
    )

    parser.add_argument("--subset-name", type=str, default="debug")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)

    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)

    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")

    args = parser.parse_args()

    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = get_device()
    use_amp = bool(args.amp and device.type == "cuda")

    train_csv = get_index_csv(args.subset_name, "train")
    val_csv = get_index_csv(args.subset_name, "validation")

    if not train_csv.exists():
        raise FileNotFoundError(f"Train index not found: {train_csv}")

    if not val_csv.exists():
        raise FileNotFoundError(f"Validation index not found: {val_csv}")

    print("=" * 80)
    print("Step 6: Train Simple CNN learned baseline")
    print("=" * 80)
    print(f"project root:      {PROJECT_ROOT}")
    print(f"subset_name:       {args.subset_name}")
    print(f"train_csv:         {train_csv}")
    print(f"val_csv:           {val_csv}")
    print(f"device:            {device}")
    print(f"use_amp:           {use_amp}")
    print(f"batch_size:        {args.batch_size}")
    print(f"epochs:            {args.epochs}")
    print(f"lr:                {args.lr}")
    print(f"weight_decay:      {args.weight_decay}")
    print(f"num_workers:       {args.num_workers}")
    print(f"max_train_batches: {args.max_train_batches}")
    print(f"max_val_batches:   {args.max_val_batches}")
    print()

    train_loader = make_option4_dataloader(
        index_csv=train_csv,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        return_audio=False,
    )

    val_loader = make_option4_dataloader(
        index_csv=val_csv,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        return_audio=False,
    )

    model = SimpleMidiToLogMelCNN(
        input_channels=3,
        n_mels=N_MELS,
        base_channels=args.base_channels,
        dropout=args.dropout,
    ).to(device)

    summary = count_parameters(model)
    print("Model summary:")
    print(f"  name: {summary.name}")
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

    best_checkpoint = checkpoint_dir / f"simple_cnn_{args.subset_name}_best.pt"
    last_checkpoint = checkpoint_dir / f"simple_cnn_{args.subset_name}_last.pt"
    history_csv = metrics_dir / f"simple_cnn_{args.subset_name}_history.csv"
    eval_csv = metrics_dir / f"simple_cnn_{args.subset_name}_eval.csv"
    loss_curve_path = figure_dir / f"simple_cnn_{args.subset_name}_loss_curve.png"
    prediction_fig_path = figure_dir / f"simple_cnn_{args.subset_name}_prediction_example.png"

    history: list[dict] = []
    best_val_l1 = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_batches=args.max_train_batches,
            use_amp=use_amp,
            grad_clip_norm=args.grad_clip_norm,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            max_batches=args.max_val_batches,
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
            f"train_l1={train_metrics['logmel_l1']:.6f} | "
            f"val_l1={val_metrics['logmel_l1']:.6f} | "
            f"val_mse={val_metrics['logmel_mse']:.6f} | "
            f"val_energy={val_metrics['energy_l1']:.6f} | "
            f"val_onset={val_metrics['onset_l1']:.6f}"
        )

        history_df = pd.DataFrame(history)
        history_df.to_csv(history_csv, index=False)
        plot_loss_curve(history_df, loss_curve_path)

        current_val_l1 = val_metrics["logmel_l1"]

        if current_val_l1 < best_val_l1:
            best_val_l1 = current_val_l1
            save_checkpoint(
                output_path=best_checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_l1=best_val_l1,
                args=args,
                history=history,
            )
            print(f"  saved best checkpoint: {best_checkpoint}")

        save_checkpoint(
            output_path=last_checkpoint,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_l1=best_val_l1,
            args=args,
            history=history,
        )

    # Load best checkpoint for final figure / eval.
    best_state = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(best_state["model_state_dict"])

    final_val_metrics = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        max_batches=args.max_val_batches,
        use_amp=use_amp,
        split_name="validation-best",
    )

    eval_row = {
        "model": "simple_cnn",
        "subset": args.subset_name,
        "split": "validation",
        "checkpoint": str(best_checkpoint),
        **final_val_metrics,
    }

    pd.DataFrame([eval_row]).to_csv(eval_csv, index=False)

    save_prediction_figure(
        model=model,
        loader=val_loader,
        device=device,
        output_path=prediction_fig_path,
        use_amp=use_amp,
        title=f"Simple CNN prediction example ({args.subset_name})",
    )

    maybe_print_baseline_comparison(
        subset_name=args.subset_name,
        val_metrics=final_val_metrics,
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
