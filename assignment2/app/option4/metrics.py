from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn.functional as F


def logmel_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean absolute error over log-mel spectrograms.

    Shapes:
        pred:   [B, M, T]
        target: [B, M, T]
    """
    return F.l1_loss(pred, target)


def logmel_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error over log-mel spectrograms.

    Shapes:
        pred:   [B, M, T]
        target: [B, M, T]
    """
    return F.mse_loss(pred, target)


def energy_contour(spec: torch.Tensor) -> torch.Tensor:
    """
    Compute frame-level energy contour from log-mel spectrogram.

    Shape:
        spec: [B, M, T]

    Returns:
        energy: [B, T]
    """
    return spec.sum(dim=1)


def positive_diff(x: torch.Tensor) -> torch.Tensor:
    """
    Positive temporal difference.

    Shape:
        x: [B, T]

    Returns:
        positive differences: [B, T-1]
    """
    return torch.relu(x[:, 1:] - x[:, :-1])


def energy_contour_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    L1 error between frame-level energy contours.

    This metric is a proxy for whether the prediction follows the
    overall loudness / density pattern of the target.
    """
    pred_energy = energy_contour(pred)
    target_energy = energy_contour(target)
    return F.l1_loss(pred_energy, target_energy)


def onset_envelope_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    L1 error between simple onset-envelope proxies.

    We estimate onset envelope using positive differences of the
    energy contour. This is not a full perceptual onset metric, but it
    is useful for checking whether note attacks create energy changes.
    """
    pred_onset = positive_diff(energy_contour(pred))
    target_onset = positive_diff(energy_contour(target))
    return F.l1_loss(pred_onset, target_onset)


def compute_batch_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute all baseline/model metrics for a batch.

    Returns Python floats for easy aggregation/logging.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")

    with torch.no_grad():
        metrics = {
            "logmel_l1": float(logmel_l1(pred, target).item()),
            "logmel_mse": float(logmel_mse(pred, target).item()),
            "energy_l1": float(energy_contour_l1(pred, target).item()),
            "onset_l1": float(onset_envelope_l1(pred, target).item()),
        }

    return metrics


@dataclass
class MetricAverager:
    """
    Weighted average of batch metrics.

    Each update should pass the batch size as weight.
    """
    totals: Dict[str, float] = field(default_factory=dict)
    count: int = 0

    def update(self, metrics: Dict[str, float], n: int) -> None:
        if n <= 0:
            return

        for key, value in metrics.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value) * n

        self.count += int(n)

    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            return {}

        return {key: value / self.count for key, value in self.totals.items()}


def weighted_logmel_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 4.0,
    quantile: float = 0.95,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Energy-weighted L1 loss for sparse log-mel spectrograms.

    Plain L1 can be dominated by near-zero regions because log-mel targets are
    sparse. This loss gives higher weight to active spectrogram regions while
    normalizing by the total weight so the scale remains stable.

    Shapes:
        pred:   [B, M, T]
        target: [B, M, T]
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")

    with torch.no_grad():
        scale = torch.quantile(target.detach().flatten(), quantile).clamp_min(eps)
        normalized = (target.detach() / scale).clamp(0.0, 1.0)
        weight = 1.0 + alpha * normalized

    error = torch.abs(pred - target)
    return (weight * error).sum() / weight.sum().clamp_min(eps)


def composite_spectrogram_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mode: str = "weighted_energy_onset",
    weighted_alpha: float = 4.0,
    energy_weight: float = 0.005,
    onset_weight: float = 0.02,
) -> torch.Tensor:
    """
    Training loss used for the formal CNN baseline and later U-Net.

    Modes:
        l1:
            plain log-mel L1

        weighted_l1:
            energy-weighted log-mel L1

        weighted_energy_onset:
            weighted log-mel L1 + small energy/onset auxiliary terms

    The auxiliary weights are intentionally small because energy_l1 has a much
    larger raw scale than logmel_l1.
    """
    if loss_mode == "l1":
        return logmel_l1(pred, target)

    if loss_mode == "weighted_l1":
        return weighted_logmel_l1(pred, target, alpha=weighted_alpha)

    if loss_mode == "weighted_energy_onset":
        main = weighted_logmel_l1(pred, target, alpha=weighted_alpha)
        energy = energy_contour_l1(pred, target)
        onset = onset_envelope_l1(pred, target)
        return main + energy_weight * energy + onset_weight * onset

    raise ValueError(
        f"Unknown loss_mode={loss_mode!r}. "
        "Expected one of: l1, weighted_l1, weighted_energy_onset."
    )
