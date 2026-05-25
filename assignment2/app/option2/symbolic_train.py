"""Training loop for Option 2 GPT-2 symbolic MIDI generation."""

from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel

from app.shared.config import (
    CHECKPOINT_DIR,
    OPTION2_BATCH_SIZE,
    OPTION2_LEARNING_RATE,
    OPTION2_MAX_EPOCHS,
    OPTION2_PATIENCE,
    OPTION2_PREFIX_MAX_LEN,
    OPTION2_WEIGHT_DECAY,
)


def _step(
    model: GPT2LMHeadModel,
    prefix: torch.Tensor,
    continuation: torch.Tensor,
) -> torch.Tensor:
    """
    One forward + loss step (teacher forcing, GPT-2 labels API).

    prefix:       (B, P) LongTensor
    continuation: (B, C) LongTensor

    Input  = [prefix | cont[:-1]]  shape (B, P+C-1)
    Labels = [-100 * P | cont[:-1]] shape (B, P+C-1)
    GPT-2 computes cross-entropy only where labels != -100,
    so only the continuation positions contribute to the loss.
    """
    inp = torch.cat([prefix, continuation[:, :-1]], dim=1)        # (B, P+C-1)
    labels = torch.cat([
        torch.full_like(prefix, -100),
        continuation[:, :-1],
    ], dim=1)                                                      # (B, P+C-1)
    # Mask PAD tokens (id=0) from attention so padding is ignored
    attention_mask = (inp != 0).long()
    return model(input_ids=inp, labels=labels, attention_mask=attention_mask).loss


def train_one_epoch(
    model: GPT2LMHeadModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for prefix, continuation in loader:
        prefix       = prefix.to(device)
        continuation = continuation.to(device)
        loss = _step(model, prefix, continuation)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: GPT2LMHeadModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for prefix, continuation in loader:
        prefix       = prefix.to(device)
        continuation = continuation.to(device)
        total_loss += _step(model, prefix, continuation).item()
    return total_loss / len(loader)


def train(
    model: GPT2LMHeadModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    checkpoint_path: Path = CHECKPOINT_DIR / "option2_best.pt",
    max_epochs: int = OPTION2_MAX_EPOCHS,
    patience: int = OPTION2_PATIENCE,
    lr: float = OPTION2_LEARNING_RATE,
    weight_decay: float = OPTION2_WEIGHT_DECAY,
) -> Dict[str, List[float]]:
    """Train with early stopping; save best checkpoint by val loss."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience // 2, factor=0.5
    )

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss   = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            tag = "✓"
        else:
            epochs_without_improvement += 1
            tag = ""

        print(
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"best={best_val_loss:.4f} {tag}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch} epochs (patience={patience}).")
            break

    return history


def load_best_checkpoint(
    model: GPT2LMHeadModel,
    checkpoint_path: Path = CHECKPOINT_DIR / "option2_best.pt",
    device: torch.device = torch.device("cpu"),
) -> GPT2LMHeadModel:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")
    return model.to(device)
