"""GPT-2 model for Option 2 symbolic conditioned MIDI generation."""

import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from app.shared.config import (
    OPTION2_DROPOUT,
    OPTION2_GPT2_N_EMBD,
    OPTION2_GPT2_N_HEAD,
    OPTION2_GPT2_N_LAYER,
    OPTION2_MAX_SEQ_LEN,
)


def build_gpt2_model(vocab_size: int, pad_token_id: int = 0) -> GPT2LMHeadModel:
    """Small GPT-2 (~15-20M params) for token-level piano generation."""
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


@torch.no_grad()
def generate_tokens(
    model: GPT2LMHeadModel,
    prefix_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 50,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Autoregressively generate max_new_tokens tokens after prefix_ids.

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
        logits = model(input_ids=ctx).logits[:, -1, :]  # (1, vocab_size)

        logits = logits / max(temperature, 1e-8)
        if top_k > 0:
            top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_vals[:, -1:]] = float("-inf")

        probs    = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)  # (1, 1)
        context  = torch.cat([context, next_tok], dim=1)

    return context[:, prefix_ids.size(1):]  # generated part only


class CopyLastPatternBaseline:
    """Baseline: cycle the last CONT_MAX_LEN tokens from the prefix."""

    def generate(self, prefix_ids: torch.Tensor, cont_len: int) -> torch.Tensor:
        """
        Args:
            prefix_ids: (B, P) LongTensor
        Returns:
            (B, cont_len) LongTensor
        """
        B, P = prefix_ids.shape
        indices = torch.arange(cont_len, device=prefix_ids.device) % P
        return prefix_ids[:, indices]
