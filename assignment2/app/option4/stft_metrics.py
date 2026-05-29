from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch


def stft_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def stft_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def energy_contour_l1_stft(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Energy contour over time. Uses mean over frequency bins to keep scale stable
    across different frequency resolutions.
    """
    pred_energy = pred.clamp_min(0.0).mean(dim=1)
    target_energy = target.clamp_min(0.0).mean(dim=1)
    return torch.mean(torch.abs(pred_energy - target_energy))


def onset_envelope_l1_stft(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_energy = pred.clamp_min(0.0).mean(dim=1)
    target_energy = target.clamp_min(0.0).mean(dim=1)

    pred_onset = torch.relu(pred_energy[:, 1:] - pred_energy[:, :-1])
    target_onset = torch.relu(target_energy[:, 1:] - target_energy[:, :-1])

    return torch.mean(torch.abs(pred_onset - target_onset))


def weighted_logstft_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 4.0,
    quantile: float = 0.95,
    eps: float = 1e-6,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")

    with torch.no_grad():
        scale = torch.quantile(target.detach().flatten(), quantile).clamp_min(eps)
        normalized = (target.detach() / scale).clamp(0.0, 1.0)
        weight = 1.0 + alpha * normalized

    error = torch.abs(pred - target)
    return (weight * error).sum() / weight.sum().clamp_min(eps)


def composite_stft_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mode: str = "weighted_energy_onset",
    weighted_alpha: float = 4.0,
    energy_weight: float = 0.05,
    onset_weight: float = 0.05,
) -> torch.Tensor:
    if loss_mode == "l1":
        return stft_l1(pred, target)

    if loss_mode == "weighted_l1":
        return weighted_logstft_l1(pred, target, alpha=weighted_alpha)

    if loss_mode == "weighted_energy_onset":
        main = weighted_logstft_l1(pred, target, alpha=weighted_alpha)
        energy = energy_contour_l1_stft(pred, target)
        onset = onset_envelope_l1_stft(pred, target)
        return main + energy_weight * energy + onset_weight * onset

    raise ValueError(
        f"Unknown loss_mode={loss_mode!r}. "
        "Expected l1, weighted_l1, or weighted_energy_onset."
    )


@torch.no_grad()
def compute_stft_batch_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    return {
        "stft_l1": float(stft_l1(pred, target).item()),
        "stft_mse": float(stft_mse(pred, target).item()),
        "energy_l1": float(energy_contour_l1_stft(pred, target).item()),
        "onset_l1": float(onset_envelope_l1_stft(pred, target).item()),
    }


@dataclass
class MetricAverager:
    sums: Dict[str, float] = field(default_factory=dict)
    count: int = 0

    def update(self, metrics: Dict[str, float], n: int) -> None:
        for key, value in metrics.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value) * n
        self.count += int(n)

    def compute(self) -> Dict[str, float]:
        if self.count <= 0:
            return {}
        return {key: value / self.count for key, value in self.sums.items()}



def log_stft_to_magnitude_tensor(
    x: torch.Tensor,
    clamp_min: float = 0.0,
) -> torch.Tensor:
    """
    Convert log1p(STFT magnitude) tensor back to magnitude.
    """
    return torch.expm1(x.clamp_min(clamp_min)).clamp_min(0.0)


def spectral_convergence_loss(
    pred_log_stft: torch.Tensor,
    target_log_stft: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Single-resolution spectral convergence loss.

    || |S_pred| - |S_target| ||_F / || |S_target| ||_F

    Inputs are log1p magnitudes, so we convert back to magnitude first.
    """
    pred_mag = log_stft_to_magnitude_tensor(pred_log_stft)
    target_mag = log_stft_to_magnitude_tensor(target_log_stft)

    numerator = torch.linalg.vector_norm(pred_mag - target_mag)
    denominator = torch.linalg.vector_norm(target_mag).clamp_min(eps)

    return numerator / denominator
