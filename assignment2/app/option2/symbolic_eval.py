"""Evaluation metrics for Option 2 symbolic generation."""

from typing import Dict, List

import numpy as np
import torch
from scipy.stats import entropy as scipy_entropy

from app.shared.config import OPTION2_CONTINUATION_SECONDS, OPTION2_FRAME_RATE


def note_density(roll: np.ndarray) -> float:
    """Average number of simultaneously active pitches per frame."""
    return float(roll.sum(axis=1).mean())


def pitch_entropy(roll: np.ndarray) -> float:
    """Shannon entropy over the marginal pitch distribution (nats)."""
    pitch_counts = roll.sum(axis=0) + 1e-8
    probs = pitch_counts / pitch_counts.sum()
    return float(scipy_entropy(probs))


def polyphony_ratio(roll: np.ndarray, min_notes: int = 2) -> float:
    """Fraction of frames that contain at least min_notes simultaneous pitches."""
    return float((roll.sum(axis=1) >= min_notes).mean())


def frame_f1(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Frame-level binary F1 between predicted and ground-truth piano-rolls.
    Both arrays should be (T, 88) binary float arrays.
    """
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(f1)


def empty_bar_ratio(roll: np.ndarray) -> float:
    """Fraction of frames with no active pitches (silence fraction)."""
    return float((roll.sum(axis=1) == 0).mean())


def evaluate_generation(
    pred_roll: np.ndarray,
    gt_roll: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics comparing predicted vs ground-truth continuation.

    Args:
        pred_roll: (T, 88) predicted binary piano-roll
        gt_roll:   (T, 88) ground-truth binary piano-roll

    Returns:
        dict of metric_name → value
    """
    # Align lengths in case of minor frame count differences
    min_T = min(len(pred_roll), len(gt_roll))
    pred = pred_roll[:min_T]
    gt = gt_roll[:min_T]

    return {
        "note_density_pred": note_density(pred),
        "note_density_gt": note_density(gt),
        "pitch_entropy_pred": pitch_entropy(pred),
        "pitch_entropy_gt": pitch_entropy(gt),
        "polyphony_ratio_pred": polyphony_ratio(pred),
        "polyphony_ratio_gt": polyphony_ratio(gt),
        "empty_bar_ratio_pred": empty_bar_ratio(pred),
        "empty_bar_ratio_gt": empty_bar_ratio(gt),
        "frame_f1": frame_f1(pred, gt),
    }


def print_metrics(metrics: Dict[str, float]) -> None:
    """Pretty-print a metrics dict."""
    print("\n--- Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"  {k:<30s} {v:.4f}")
    print()


def evaluate_token_generation(
    pred_ids: List[int],
    gt_ids: List[int],
    tokenizer,
    frame_rate: float = OPTION2_FRAME_RATE,
    duration_seconds: float = OPTION2_CONTINUATION_SECONDS,
) -> Dict[str, float]:
    """Decode token sequences to piano-rolls and evaluate."""
    from app.option2.symbolic_generate import tokens_to_pianoroll
    pred_roll = tokens_to_pianoroll(pred_ids, tokenizer, frame_rate, duration_seconds)
    gt_roll   = tokens_to_pianoroll(gt_ids,   tokenizer, frame_rate, duration_seconds)
    return evaluate_generation(pred_roll, gt_roll)
