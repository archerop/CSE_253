"""Model architectures for Option 2 symbolic conditioned MIDI generation.

All models implement the unified interface:
    forward(x: LongTensor (B, T)) -> Tensor (B, T, vocab_size)

Available via build_model(model_type, vocab_size):
    'lstm'        - SymbolicLSTM        (~1.1M params)
    'gru'         - SymbolicGRU         (~0.9M params)
    'transformer' - SymbolicTransformerTokens (~0.9M params)
    'gpt2'        - GPT2Wrapper         (~11M params)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from app.shared.config import (
    OPTION2_D_MODEL,
    OPTION2_DIM_FEEDFORWARD,
    OPTION2_DROPOUT,
    OPTION2_GPT2_N_EMBD,
    OPTION2_GPT2_N_HEAD,
    OPTION2_GPT2_N_LAYER,
    OPTION2_HIDDEN_SIZE,
    OPTION2_MAX_SEQ_LEN,
    OPTION2_NHEAD,
    OPTION2_NUM_LAYERS,
    OPTION2_RNN_LAYERS,
)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

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
        return self.dropout(x + self.pe[:, : x.size(1)])


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

class SymbolicLSTM(nn.Module):
    """Embedding → multi-layer LSTM → linear projection."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = OPTION2_HIDDEN_SIZE,
        num_layers: int = OPTION2_RNN_LAYERS,
        dropout: float = OPTION2_DROPOUT,
    ):
        super().__init__()
        self.embedding   = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.lstm        = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) LongTensor → (B, T, vocab_size)
        h, _ = self.lstm(self.embedding(x))
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# GRU
# ---------------------------------------------------------------------------

class SymbolicGRU(nn.Module):
    """Embedding → multi-layer GRU → linear projection."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = OPTION2_HIDDEN_SIZE,
        num_layers: int = OPTION2_RNN_LAYERS,
        dropout: float = OPTION2_DROPOUT,
    ):
        super().__init__()
        self.embedding   = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.gru         = nn.GRU(
            hidden_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(self.embedding(x))
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# Simple causal Transformer (token-based)
# ---------------------------------------------------------------------------

class SymbolicTransformerTokens(nn.Module):
    """Lightweight causal Transformer with token embeddings."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = OPTION2_D_MODEL,
        nhead: int = OPTION2_NHEAD,
        num_layers: int = OPTION2_NUM_LAYERS,
        dim_feedforward: int = OPTION2_DIM_FEEDFORWARD,
        dropout: float = OPTION2_DROPOUT,
        max_seq_len: int = OPTION2_MAX_SEQ_LEN,
    ):
        super().__init__()
        self.embedding   = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc     = PositionalEncoding(d_model, max_seq_len, dropout)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T    = x.size(1)
        h    = self.pos_enc(self.embedding(x))
        mask = self._causal_mask(T, x.device)
        h    = self.transformer(h, mask=mask, is_causal=True)
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# GPT-2 wrapper (unified interface)
# ---------------------------------------------------------------------------

class GPT2Wrapper(nn.Module):
    """Wraps HuggingFace GPT2LMHeadModel to match forward(x) → (B,T,vocab) interface."""

    def __init__(self, gpt2_model: GPT2LMHeadModel):
        super().__init__()
        self.model = gpt2_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=x).logits  # (B, T, vocab_size)


def build_gpt2_model(vocab_size: int, pad_token_id: int = 0) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=OPTION2_MAX_SEQ_LEN,
        n_embd=OPTION2_GPT2_N_EMBD,
        n_layer=OPTION2_GPT2_N_LAYER,
        n_head=OPTION2_GPT2_N_HEAD,
        pad_token_id=pad_token_id,
        bos_token_id=1,
        eos_token_id=2,
        resid_pdrop=OPTION2_DROPOUT,
        embd_pdrop=OPTION2_DROPOUT,
        attn_pdrop=OPTION2_DROPOUT,
    )
    return GPT2LMHeadModel(config)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(model_type: str, vocab_size: int) -> nn.Module:
    """Return a model for the given type. All share forward(x) → (B,T,vocab_size)."""
    if model_type == "lstm":
        return SymbolicLSTM(vocab_size)
    if model_type == "gru":
        return SymbolicGRU(vocab_size)
    if model_type == "transformer":
        return SymbolicTransformerTokens(vocab_size)
    if model_type == "gpt2":
        return GPT2Wrapper(build_gpt2_model(vocab_size))
    raise ValueError(f"Unknown model_type '{model_type}'. Choose: lstm | gru | transformer | gpt2")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_tokens(
    model: nn.Module,
    prefix_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 50,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Autoregressively generate max_new_tokens tokens after prefix_ids.

    Works with any model that implements forward(x: LongTensor) → (B, T, vocab_size).

    Args:
        prefix_ids: (1, P) LongTensor
    Returns:
        (1, max_new_tokens) LongTensor — generated tokens only
    """
    model.eval()
    model = model.to(device)
    context = prefix_ids.to(device)

    for _ in range(max_new_tokens):
        ctx    = context[:, -OPTION2_MAX_SEQ_LEN:]
        logits = model(ctx)[:, -1, :]           # (1, vocab_size)

        logits = logits / max(temperature, 1e-8)
        if top_k > 0:
            top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_vals[:, -1:]] = float("-inf")

        probs    = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)  # (1, 1)
        context  = torch.cat([context, next_tok], dim=1)

    return context[:, prefix_ids.size(1):]  # generated part only


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

class CopyLastPatternBaseline:
    """Baseline: cycle the last CONT_MAX_LEN tokens from the prefix."""

    def generate(self, prefix_ids: torch.Tensor, cont_len: int) -> torch.Tensor:
        B, P = prefix_ids.shape
        indices = torch.arange(cont_len, device=prefix_ids.device) % P
        return prefix_ids[:, indices]
