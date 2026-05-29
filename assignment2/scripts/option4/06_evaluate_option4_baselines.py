from __future__ import annotations

from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.shared.config import (
    MIDI_LOW,
    N_MELS,
    FMIN,
    FMAX,
    WINDOW_INDEX_CACHE_DIR,
    OPTION4_OUTPUT_DIR,
)
from app.option4.option4_dataset import make_option4_dataloader
from app.option4.baselines import (
    silence_baseline_like,
    heuristic_note_to_logmel_baseline,
)
from app.option4.metrics import (
    MetricAverager,
    compute_batch_metrics,
)


def plot_baseline_comparison(
    target: torch.Tensor,
    silence_pred: torch.Tensor,
    heuristic_pred: torch.Tensor,
    output_path: Path,
    title: str,
) -> None:
    """
    Plot target and baseline spectrograms for one example.

    Inputs are single examples with shape [M, T].
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_np = target.detach().cpu()
    silence_np = silence_pred.detach().cpu()
    heuristic_np = heuristic_pred.detach().cpu()

    heuristic_error = torch.abs(heuristic_np - target_np)
    silence_error = torch.abs(silence_np - target_np)

    vmax = float(torch.quantile(target_np, 0.99).item())
    vmax = max(vmax, 1e-6)

    err_vmax = float(torch.quantile(heuristic_error, 0.99).item())
    err_vmax = max(err_vmax, 1e-6)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(15, 7),
        constrained_layout=True,
    )

    images = [
        (target_np, "Target log-mel", 0.0, vmax),
        (silence_np, "Silence baseline", 0.0, vmax),
        (heuristic_np, "Heuristic MIDI baseline", 0.0, vmax),
        (silence_error, "|Silence - target|", 0.0, err_vmax),
        (heuristic_error, "|Heuristic - target|", 0.0, err_vmax),
        (target_np - heuristic_np, "Target - heuristic", -err_vmax, err_vmax),
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
        ax.set_ylabel("mel bin")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_baselines(
    subset_name: str,
    split: str,
    batch_size: int,
    num_workers: int,
    max_batches: int | None,
    heuristic_strength: float,
) -> pd.DataFrame:
    index_csv = (
        WINDOW_INDEX_CACHE_DIR
        / f"option4_{subset_name}_{split}_windows.csv"
    )

    if not index_csv.exists():
        raise FileNotFoundError(
            f"Window index not found: {index_csv}\n"
            "Build it first with scripts/option4/04_build_option4_window_index.py"
        )

    loader = make_option4_dataloader(
        index_csv=index_csv,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        return_audio=False,
    )

    silence_avg = MetricAverager()
    heuristic_avg = MetricAverager()

    first_batch = None
    total_seen = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Evaluating {split}")):
        if max_batches is not None and batch_idx >= max_batches:
            break

        piano_roll = batch["piano_roll"]
        target = batch["log_mel"]

        silence_pred = silence_baseline_like(target)

        heuristic_pred = heuristic_note_to_logmel_baseline(
            piano_roll=piano_roll,
            n_mels=N_MELS,
            midi_low=MIDI_LOW,
            fmin=FMIN,
            fmax=FMAX,
            strength=heuristic_strength,
        )

        n = target.shape[0]
        total_seen += n

        silence_metrics = compute_batch_metrics(silence_pred, target)
        heuristic_metrics = compute_batch_metrics(heuristic_pred, target)

        silence_avg.update(silence_metrics, n=n)
        heuristic_avg.update(heuristic_metrics, n=n)

        if first_batch is None:
            first_batch = {
                "target": target[0].detach().clone(),
                "silence": silence_pred[0].detach().clone(),
                "heuristic": heuristic_pred[0].detach().clone(),
                "window_id": batch["window_id"][0],
                "piece_id": batch["piece_id"][0],
                "composer": batch["composer"][0],
                "title": batch["title"][0],
            }

    rows = []

    for name, avg in [
        ("silence", silence_avg),
        ("heuristic_note_to_mel", heuristic_avg),
    ]:
        metrics = avg.compute()
        row = {
            "baseline": name,
            "subset": subset_name,
            "split": split,
            "num_examples": total_seen,
            "batch_size": batch_size,
            "max_batches": max_batches if max_batches is not None else "all",
            "heuristic_strength": heuristic_strength if name.startswith("heuristic") else "",
        }
        row.update(metrics)
        rows.append(row)

    results = pd.DataFrame(rows)

    metrics_dir = OPTION4_OUTPUT_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    output_csv = metrics_dir / f"baseline_metrics_{subset_name}_{split}.csv"
    results.to_csv(output_csv, index=False)

    print()
    print(f"Saved baseline metrics to: {output_csv}")
    print(results.to_string(index=False))

    if first_batch is not None:
        figure_dir = OPTION4_OUTPUT_DIR / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)

        figure_path = figure_dir / f"baseline_comparison_{subset_name}_{split}.png"

        title = (
            f"Option 4 baseline comparison\n"
            f"{first_batch['composer']} — {first_batch['title']}\n"
            f"window_id={first_batch['window_id']}"
        )

        plot_baseline_comparison(
            target=first_batch["target"],
            silence_pred=first_batch["silence"],
            heuristic_pred=first_batch["heuristic"],
            output_path=figure_path,
            title=title,
        )

        print(f"Saved baseline comparison figure to: {figure_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 5: Evaluate Option 4 non-learned baselines."
    )

    parser.add_argument("--subset-name", type=str, default="smoke")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional limit for fast debugging.",
    )
    parser.add_argument(
        "--heuristic-strength",
        type=float,
        default=0.25,
        help="Global scale factor for heuristic note-to-mel baseline.",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Step 5: Option 4 baseline evaluation")
    print("=" * 80)
    print(f"subset_name:        {args.subset_name}")
    print(f"split:              {args.split}")
    print(f"batch_size:         {args.batch_size}")
    print(f"num_workers:        {args.num_workers}")
    print(f"max_batches:        {args.max_batches}")
    print(f"heuristic_strength: {args.heuristic_strength}")
    print()

    evaluate_baselines(
        subset_name=args.subset_name,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        heuristic_strength=args.heuristic_strength,
    )

    print("=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
