"""Evaluation metrics for Option 2 symbolic generation."""

from typing import Dict, List

import numpy as np
import torch
from scipy.linalg import sqrtm
from scipy.stats import entropy as scipy_entropy

from app.shared.config import MIDI_LOW, OPTION2_CONTINUATION_SECONDS, OPTION2_FRAME_RATE


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


def mgeval_features(roll: np.ndarray) -> np.ndarray:
    """
    Compute a 28-dim MGEval feature vector from a (T, 88) binary piano-roll.

    Features (in order):
      [0:12]  pitch class histogram (L1-normalised)
      [12:20] note length histogram (L1-normalised, 8 buckets)
      [20]    note density (mean active pitches/frame)
      [21]    pitch range (max_pitch - min_pitch, 0 if silent)
      [22]    average pitch interval (mean |semitone step| between consecutive onsets)
      [23]    polyphony ratio (fraction of frames with >=2 notes)
      [24]    empty bar ratio (fraction of silent frames)
      [25]    pitch entropy (Shannon entropy of pitch class histogram)
      [26]    pitch count (number of distinct pitches)
      [27]    note count (total number of notes)
    """
    T, P = roll.shape  # P == 88

    # Extract notes as (pitch_idx, start_frame, end_frame) via run-length encoding
    notes = []
    for p in range(P):
        col = roll[:, p].astype(bool)
        in_note = False
        start = 0
        for t in range(T):
            if col[t] and not in_note:
                in_note = True
                start = t
            elif not col[t] and in_note:
                notes.append((p, start, t))
                in_note = False
        if in_note:
            notes.append((p, start, T))

    feat = np.zeros(28, dtype=np.float64)

    # Pitch class histogram [0:12]
    pc_hist = np.zeros(12, dtype=np.float64)
    for p, s, e in notes:
        midi_pitch = p + MIDI_LOW
        pc_hist[midi_pitch % 12] += 1
    if pc_hist.sum() > 0:
        pc_hist /= pc_hist.sum()
    feat[0:12] = pc_hist

    # Note length histogram [12:20] — 8 buckets in frames
    buckets = [1, 2, 4, 8, 16, 32, 64]  # upper edges (exclusive) for first 7
    len_hist = np.zeros(8, dtype=np.float64)
    for p, s, e in notes:
        length = e - s
        if length <= 1:
            b = 0
        elif length == 2:
            b = 1
        elif length <= 4:
            b = 2
        elif length <= 8:
            b = 3
        elif length <= 16:
            b = 4
        elif length <= 32:
            b = 5
        elif length <= 64:
            b = 6
        else:
            b = 7
        len_hist[b] += 1
    if len_hist.sum() > 0:
        len_hist /= len_hist.sum()
    feat[12:20] = len_hist

    # Note density [20]
    feat[20] = roll.sum(axis=1).mean()

    # Pitch range [21]
    if notes:
        pitches = [p for p, s, e in notes]
        feat[21] = float(max(pitches) - min(pitches))

    # Average pitch interval [22]
    if len(notes) >= 2:
        sorted_notes = sorted(notes, key=lambda x: x[1])
        intervals = [abs((sorted_notes[i+1][0] + MIDI_LOW) - (sorted_notes[i][0] + MIDI_LOW))
                     for i in range(len(sorted_notes) - 1)]
        feat[22] = float(np.mean(intervals))

    # Polyphony ratio [23]
    feat[23] = float((roll.sum(axis=1) >= 2).mean())

    # Empty bar ratio [24]
    feat[24] = float((roll.sum(axis=1) == 0).mean())

    # Pitch entropy [25]
    if pc_hist.sum() > 0:
        nonzero = pc_hist[pc_hist > 0]
        feat[25] = float(-np.sum(nonzero * np.log(nonzero + 1e-12)))

    # Pitch count [26]
    if notes:
        feat[26] = float(len(set(p for p, s, e in notes)))

    # Note count [27]
    feat[27] = float(len(notes))

    return feat


def fmd(gen_rolls: List[np.ndarray], gt_rolls: List[np.ndarray]) -> float:
    """
    Compute Fréchet Music Distance between generated and ground-truth piano-roll sets.

    Args:
        gen_rolls: list of (T, 88) arrays (generated)
        gt_rolls:  list of (T, 88) arrays (ground truth)

    Returns:
        FMD scalar (lower is better; 0 means identical distributions)
    """
    eps = 1e-6

    gen_feats = np.stack([mgeval_features(r) for r in gen_rolls])  # (N, 28)
    gt_feats  = np.stack([mgeval_features(r) for r in gt_rolls])   # (N, 28)

    mu1, mu2 = gen_feats.mean(0), gt_feats.mean(0)
    sigma1 = np.cov(gen_feats, rowvar=False) + eps * np.eye(gen_feats.shape[1])
    sigma2 = np.cov(gt_feats,  rowvar=False) + eps * np.eye(gt_feats.shape[1])

    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2).real

    return float(np.dot(diff, diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def evaluate_dataset(
    gen_rolls: List[np.ndarray],
    gt_rolls: List[np.ndarray],
) -> Dict[str, float]:
    """
    Compute dataset-level metrics: per-sample averages + FMD.

    Returns a flat dict with the same keys as evaluate_generation plus 'fmd'.
    """
    per_sample = [evaluate_generation(g, t) for g, t in zip(gen_rolls, gt_rolls)]
    keys = per_sample[0].keys()
    averaged = {k: float(np.mean([m[k] for m in per_sample])) for k in keys}
    averaged["fmd"] = fmd(gen_rolls, gt_rolls)
    return averaged


def print_dataset_metrics(metrics: Dict[str, float]) -> None:
    """Pretty-print aggregated dataset-level metrics."""
    print("\n--- Dataset Metrics ---")
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
