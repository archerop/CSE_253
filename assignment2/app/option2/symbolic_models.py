"""Model architectures for Option 2 symbolic conditioned generation."""

import math

import torch
import torch.nn as nn

from app.shared.config import (
    N_PITCHES,
    OPTION2_D_MODEL,
    OPTION2_DIM_FEEDFORWARD,
    OPTION2_DROPOUT,
    OPTION2_NHEAD,
    OPTION2_NUM_LAYERS,
)


class CopyLastFrameBaseline(nn.Module):
    """Baseline: repeat the last prefix frame for the full continuation."""

    def forward(self, prefix: torch.Tensor, cont_len: int) -> torch.Tensor:
        # prefix: (B, P, 88) → returns (B, cont_len, 88)
        last = prefix[:, -1:, :]
        return last.expand(-1, cont_len, -1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class SymbolicTransformer(nn.Module):
    """
    Causal Transformer decoder for autoregressive piano-roll prediction.

    During training (teacher forcing):
      - input = [prefix | continuation[:-1]]  shape (B, P+C-1, 88)
      - loss computed on output[:, P-1:, :] vs continuation  (B, C, 88)

    During generation:
      - prefix fed as context, continuation grown autoregressively
    """

    def __init__(
        self,
        n_pitches: int = N_PITCHES,
        d_model: int = OPTION2_D_MODEL,
        nhead: int = OPTION2_NHEAD,
        num_layers: int = OPTION2_NUM_LAYERS,
        dim_feedforward: int = OPTION2_DIM_FEEDFORWARD,
        dropout: float = OPTION2_DROPOUT,
        max_seq_len: int = 1000,
    ):
        super().__init__()
        self.n_pitches = n_pitches
        self.d_model = d_model

        self.input_proj = nn.Linear(n_pitches, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, n_pitches)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask (True = ignore) for causal attention."""
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 88) → logits (B, T, 88)
        T = x.size(1)
        h = self.pos_enc(self.input_proj(x))
        mask = self._causal_mask(T, x.device)
        h = self.transformer(h, mask=mask, is_causal=True)
        return self.output_proj(h)

    @torch.no_grad()
    def generate(
        self,
        prefix: torch.Tensor,
        cont_len: int,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Autoregressively generate cont_len frames given a prefix.

        Args:
            prefix: (B, P, 88) or (P, 88) float tensor
            cont_len: number of frames to generate
            threshold: sigmoid threshold for binarizing output

        Returns:
            generated: (B, cont_len, 88) binary float tensor
        """
        self.eval()
        if prefix.dim() == 2:
            prefix = prefix.unsqueeze(0)

        context = prefix.clone()
        generated = []

        for _ in range(cont_len):
            logits = self.forward(context)           # (B, T, 88)
            next_frame = (torch.sigmoid(logits[:, -1:, :]) > threshold).float()
            generated.append(next_frame)
            context = torch.cat([context, next_frame], dim=1)

        return torch.cat(generated, dim=1)           # (B, cont_len, 88)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
