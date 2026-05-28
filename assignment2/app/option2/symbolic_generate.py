"""Token-based generation and MIDI conversion for Option 2 (GPT-2 + REMI)."""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pretty_midi
import symusic
import torch

from miditok import REMI
from miditok.classes import TokSequence

from app.shared.config import (
    MIDI_LOW,
    MIDI_HIGH,
    N_PITCHES,
    OPTION2_CONTINUATION_SECONDS,
    OPTION2_CONT_MAX_LEN,
    OPTION2_FRAME_RATE,
    OPTION2_OUTPUT_DIR,
    OPTION2_PREFIX_MAX_LEN,
    OPTION2_PREFIX_SECONDS,
)
from transformers import GPT2LMHeadModel
from app.option2.symbolic_models import CopyLastPatternBaseline, generate_tokens


def _pad_or_truncate(seq: List[int], max_len: int, pad_id: int = 0) -> List[int]:
    seq = seq[:max_len]
    return seq + [pad_id] * (max_len - len(seq))


def _trim_score(score: symusic.Score, start_s: float, end_s: float) -> symusic.Score:
    """Clip a symusic.Score to [start_s, end_s) and shift time to 0."""
    tpq = score.ticks_per_quarter
    qpm = score.tempos[0].qpm if score.tempos else 120.0
    ticks_per_sec = qpm * tpq / 60.0
    start_tick = int(start_s * ticks_per_sec)
    end_tick   = int(end_s   * ticks_per_sec)
    return score.clip(start_tick, end_tick).shift_time(-start_tick)


def _score_to_pretty_midi(score: symusic.Score) -> pretty_midi.PrettyMIDI:
    """Convert symusic.Score to pretty_midi.PrettyMIDI via a temp MIDI write."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        tmp_path = f.name
    try:
        score.dump_midi(tmp_path)
        pm = pretty_midi.PrettyMIDI(tmp_path)
    finally:
        os.unlink(tmp_path)
    return pm


def extract_prefix(
    midi_path: str,
    tokenizer: REMI,
    prefix_seconds: float = OPTION2_PREFIX_SECONDS,
    prefix_max_len: int = OPTION2_PREFIX_MAX_LEN,
) -> torch.Tensor:
    """Return (prefix_max_len,) LongTensor for the first prefix_seconds of a MIDI."""
    score    = symusic.Score(midi_path)
    trimmed  = _trim_score(score, 0.0, prefix_seconds)
    seqs     = tokenizer.encode(trimmed)
    ids      = seqs[0].ids if seqs else []
    padded   = _pad_or_truncate(ids, prefix_max_len, pad_id=tokenizer["PAD_None"])
    return torch.tensor(padded, dtype=torch.long)


def generate_conditioned(
    model: GPT2LMHeadModel,
    prefix_ids: torch.Tensor,
    cont_max_len: int,
    device: torch.device,
    temperature: float = 0.8,
    top_k: int = 10,
    eos_id: int = 2,
    pad_id: int = 0,
) -> torch.Tensor:
    """
    Run autoregressive token generation.

    Args:
        prefix_ids: (prefix_max_len,) LongTensor (1D)
    Returns:
        (cont_max_len,) LongTensor — generated continuation tokens (padded after EOS)
    """
    generated = generate_tokens(
        model,
        prefix_ids.unsqueeze(0),   # (1, P)
        max_new_tokens=cont_max_len,
        temperature=temperature,
        top_k=top_k,
        device=str(device),
        eos_id=eos_id,
        pad_id=pad_id,
    )
    return generated.squeeze(0)  # (cont_max_len,)


def tokens_to_pianoroll(
    token_ids: List[int],
    tokenizer: REMI,
    frame_rate: float = OPTION2_FRAME_RATE,
    duration_seconds: float = OPTION2_CONTINUATION_SECONDS,
) -> np.ndarray:
    """
    Decode a REMI token sequence → piano-roll (T, 88) float32.

    Returns an array of zeros if decoding produces no notes.
    """
    # Strip padding and EOS — neither carries musical information
    pad_id  = tokenizer["PAD_None"]
    eos_id  = tokenizer["EOS_None"]
    ids     = [t for t in token_ids if t not in (pad_id, eos_id)]

    n_frames = int(np.ceil(duration_seconds * frame_rate))
    roll     = np.zeros((n_frames, N_PITCHES), dtype=np.float32)

    if not ids:
        return roll

    try:
        tok_seq = TokSequence(ids=ids)
        score   = tokenizer.decode([tok_seq])
        pm      = _score_to_pretty_midi(score)
        for inst in pm.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                pitch_idx = note.pitch - MIDI_LOW
                if not (0 <= pitch_idx < N_PITCHES):
                    continue
                start_f = int(note.start * frame_rate)
                end_f   = min(int(note.end * frame_rate), n_frames)
                if start_f < n_frames:
                    roll[start_f:end_f, pitch_idx] = 1.0
    except Exception as e:
        import warnings
        warnings.warn(f"REMI decode failed ({len(ids)} tokens): {e}")

    return roll


def save_symbolic_conditioned(
    prefix_midi_path: str,
    model: GPT2LMHeadModel,
    tokenizer: REMI,
    device: torch.device,
    output_path: Optional[Path] = None,
    prefix_seconds: float = OPTION2_PREFIX_SECONDS,
    continuation_seconds: float = OPTION2_CONTINUATION_SECONDS,
    temperature: float = 0.8,
    top_k: int = 10,
) -> Path:
    """
    Full pipeline: tokenize prefix → generate continuation tokens → decode → save .mid.

    Returns the path of the saved symbolic_conditioned.mid.
    """
    if output_path is None:
        OPTION2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OPTION2_OUTPUT_DIR / "symbolic_conditioned.mid"

    # 1. Tokenize prefix
    prefix_ids = extract_prefix(prefix_midi_path, tokenizer, prefix_seconds)

    # 2. Generate continuation tokens
    cont_ids = generate_conditioned(
        model, prefix_ids, OPTION2_CONT_MAX_LEN, device, temperature, top_k,
        eos_id=tokenizer["EOS_None"],
        pad_id=tokenizer["PAD_None"],
    )

    # 3. Build prefix MIDI from original file (preserves original velocity/timing)
    original_pm   = pretty_midi.PrettyMIDI(prefix_midi_path)
    prefix_pm     = pretty_midi.PrettyMIDI()
    prefix_inst   = pretty_midi.Instrument(program=0, name="Piano")
    for inst in original_pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            if note.start < prefix_seconds:
                prefix_inst.notes.append(pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start,
                    end=min(note.end, prefix_seconds),
                ))
    prefix_pm.instruments.append(prefix_inst)

    # 4. Decode continuation tokens → pretty_midi → shift by prefix_seconds
    pad_id   = tokenizer["PAD_None"]
    eos_id   = tokenizer["EOS_None"]
    ids      = [t for t in cont_ids.tolist() if t not in (pad_id, eos_id)]
    cont_pm  = pretty_midi.PrettyMIDI()
    if ids:
        try:
            tok_seq   = TokSequence(ids=ids)
            score     = tokenizer.decode([tok_seq])
            cont_pm   = _score_to_pretty_midi(score)
        except Exception:
            pass

    # 5. Merge prefix + continuation into one MIDI
    combined_pm   = pretty_midi.PrettyMIDI()
    combined_inst = pretty_midi.Instrument(program=0, name="Piano")

    for note in prefix_inst.notes:
        combined_inst.notes.append(note)

    for inst in cont_pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            combined_inst.notes.append(pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start  + prefix_seconds,
                end=note.end      + prefix_seconds,
            ))

    combined_inst.notes.sort(key=lambda n: n.start)
    combined_pm.instruments.append(combined_inst)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_pm.write(str(output_path))
    print(f"Saved: {output_path}")
    return output_path
