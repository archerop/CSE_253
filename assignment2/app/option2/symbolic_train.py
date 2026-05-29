"""Training loop for Option 2 symbolic MIDI generation (model-agnostic)."""

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from app.shared.config import (
    CHECKPOINT_DIR,
    OPTION2_BATCH_SIZE,
    OPTION2_LEARNING_RATE,
    OPTION2_MAX_EPOCHS,
    OPTION2_PATIENCE,
    OPTION2_WEIGHT_DECAY,
)


def _step(
    model: nn.Module,
    prefix: torch.Tensor,
    continuation: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    One forward + loss step (teacher forcing, model-agnostic).

    prefix:       (B, P) LongTensor
    continuation: (B, C) LongTensor
    weight:       (vocab_size,) float tensor — per-token loss weights, or None

    Input  = [prefix | cont[:-1]]  shape (B, P+C-1)
    Target = continuation           shape (B, C)
    Loss   = weighted cross-entropy over continuation positions (PAD id=0 ignored).
    """
    P   = prefix.size(1)
    inp = torch.cat([prefix, continuation[:, :-1]], dim=1)  # (B, P+C-1)

    logits = model(inp)                   # (B, P+C-1, vocab_size)
    pred   = logits[:, P - 1:, :]        # (B, C, vocab_size)

    return F.cross_entropy(
        pred.reshape(-1, pred.size(-1)),
        continuation.reshape(-1),
        weight=weight,
        ignore_index=0,  # PAD token
    )


def build_note_weight(tokenizer, note_token_weight: float = 2.0) -> torch.Tensor:
    """
    Return a (vocab_size,) float tensor where Pitch/Duration/Velocity tokens
    have weight `note_token_weight` and all other tokens have weight 1.0.

    Upweighting note tokens counteracts the natural skew of REMI sequences
    toward structural tokens (BAR, Position, Rest, Tempo), which otherwise
    causes the model to generate sparse, mostly-silent continuations.
    """
    w = torch.ones(tokenizer.vocab_size)
    for tok_str, tok_id in tokenizer.vocab.items():
        if any(tok_str.startswith(p) for p in ("Pitch_", "Duration_", "Velocity_")):
            w[tok_id] = note_token_weight
    n_note = int((w > 1.0).sum().item())
    print(f"Note-token weight={note_token_weight:.1f} applied to {n_note} tokens "
          f"(Pitch/Duration/Velocity) out of {tokenizer.vocab_size}")
    return w


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weight: Optional[torch.Tensor] = None,
) -> float:
    model.train()
    total_loss = 0.0
    for prefix, continuation in loader:
        prefix       = prefix.to(device)
        continuation = continuation.to(device)
        loss = _step(model, prefix, continuation, weight=weight)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    weight: Optional[torch.Tensor] = None,
) -> float:
    model.eval()
    total_loss = 0.0
    for prefix, continuation in loader:
        prefix       = prefix.to(device)
        continuation = continuation.to(device)
        total_loss  += _step(model, prefix, continuation, weight=weight).item()
    return total_loss / len(loader)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    checkpoint_path: Path = CHECKPOINT_DIR / "option2_best.pt",
    max_epochs: int = OPTION2_MAX_EPOCHS,
    patience: int = OPTION2_PATIENCE,
    lr: float = OPTION2_LEARNING_RATE,
    weight_decay: float = OPTION2_WEIGHT_DECAY,
    note_token_weight: float = 2.0,
    tokenizer=None,
    resume_from: Optional[Path] = None,
) -> Dict[str, List[float]]:
    """
    Train with early stopping; save best checkpoint by val loss.

    Args:
        note_token_weight: Loss multiplier for Pitch/Duration/Velocity tokens.
            Set to 1.0 to use uniform cross-entropy. Values 1.5–3.0 encourage
            denser note generation by counteracting the structural-token bias.
        tokenizer: Required when note_token_weight != 1.0.
        resume_from: Path to an existing checkpoint to resume training from.
            Loads model weights, optimizer state, and continues epoch numbering.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Build per-token loss weight vector
    token_weight: Optional[torch.Tensor] = None
    if note_token_weight != 1.0 and tokenizer is not None:
        token_weight = build_note_weight(tokenizer, note_token_weight).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience // 2, factor=0.5
    )

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    start_epoch = 1

    # Resume from checkpoint if provided
    if resume_from is not None and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_val_loss = ckpt["val_loss"]
        start_epoch   = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']} (val_loss={best_val_loss:.4f})")
    elif resume_from is not None:
        print(f"Checkpoint not found at {resume_from} — training from scratch.")

    for epoch in range(start_epoch, start_epoch + max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, weight=token_weight)
        val_loss   = evaluate(model, val_loader, device, weight=token_weight)
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
            f"Epoch {epoch:3d} | "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"best={best_val_loss:.4f} {tag}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch} epochs (patience={patience}).")
            break

    return history


def load_best_checkpoint(
    model: nn.Module,
    checkpoint_path: Path = CHECKPOINT_DIR / "option2_best.pt",
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")
    return model.to(device)
