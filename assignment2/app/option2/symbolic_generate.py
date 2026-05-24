"""
Autoregressive generation and piano-roll → MIDI conversion for Option 2.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pretty_midi
import torch

from app.shared.config import (
    MIDI_LOW,
    MIDI_HIGH,
    N_PITCHES,
    OPTION2_CONTINUATION_SECONDS,
    OPTION2_FRAME_RATE,
    OPTION2_OUTPUT_DIR,
    OPTION2_PREFIX_SECONDS,
)
from app.option2.symbolic_dataset import _midi_to_pianoroll
from app.option2.symbolic_models import SymbolicTransformer


def extract_prefix(
    midi_path: str,
    prefix_seconds: float = OPTION2_PREFIX_SECONDS,
    frame_rate: float = OPTION2_FRAME_RATE,
) -> torch.Tensor:
    """Return a (P, 88) float tensor for the first prefix_seconds of a MIDI file."""
    roll = _midi_to_pianoroll(midi_path, frame_rate)
    prefix_len = int(prefix_seconds * frame_rate)
    prefix = np.zeros((prefix_len, N_PITCHES), dtype=np.float32)
    actual = min(prefix_len, len(roll))
    prefix[:actual] = roll[:actual]
    return torch.from_numpy(prefix)


def pianoroll_to_midi(
    roll: np.ndarray,
    frame_rate: float = OPTION2_FRAME_RATE,
    velocity: int = 80,
    min_note_frames: int = 1,
) -> pretty_midi.PrettyMIDI:
    """
    Convert a binary piano-roll array (T, 88) to a PrettyMIDI object.

    Consecutive active frames for the same pitch become a single note.
    """
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0, name="Piano")

    T, _ = roll.shape
    for pitch_idx in range(N_PITCHES):
        pitch = pitch_idx + MIDI_LOW
        in_note = False
        note_start = 0

        for t in range(T):
            active = roll[t, pitch_idx] > 0.5
            if active and not in_note:
                note_start = t
                in_note = True
            elif not active and in_note:
                if t - note_start >= min_note_frames:
                    piano.notes.append(
                        pretty_midi.Note(
                            velocity=velocity,
                            pitch=pitch,
                            start=note_start / frame_rate,
                            end=t / frame_rate,
                        )
                    )
                in_note = False

        if in_note and T - note_start >= min_note_frames:
            piano.notes.append(
                pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=note_start / frame_rate,
                    end=T / frame_rate,
                )
            )

    piano.notes.sort(key=lambda n: n.start)
    pm.instruments.append(piano)
    return pm


def generate_conditioned(
    model: SymbolicTransformer,
    prefix_tensor: torch.Tensor,
    cont_len: int,
    device: torch.device,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Run autoregressive generation.

    Args:
        model: trained SymbolicTransformer
        prefix_tensor: (P, 88) float tensor
        cont_len: number of frames to generate
        device: torch device
        threshold: binarization threshold

    Returns:
        generated: (cont_len, 88) numpy array (binary float32)
    """
    model = model.to(device)
    prefix = prefix_tensor.unsqueeze(0).to(device)  # (1, P, 88)
    generated = model.generate(prefix, cont_len, threshold=threshold)  # (1, C, 88)
    return generated.squeeze(0).cpu().numpy()


def save_symbolic_conditioned(
    prefix_midi_path: str,
    model: SymbolicTransformer,
    device: torch.device,
    output_path: Optional[Path] = None,
    prefix_seconds: float = OPTION2_PREFIX_SECONDS,
    continuation_seconds: float = OPTION2_CONTINUATION_SECONDS,
    frame_rate: float = OPTION2_FRAME_RATE,
    threshold: float = 0.5,
) -> Path:
    """
    Full pipeline: load MIDI prefix → generate continuation → save combined MIDI.

    Returns the path of the saved .mid file.
    """
    if output_path is None:
        OPTION2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OPTION2_OUTPUT_DIR / "symbolic_conditioned.mid"

    # 1. Extract prefix piano-roll
    prefix_tensor = extract_prefix(prefix_midi_path, prefix_seconds, frame_rate)
    prefix_roll = prefix_tensor.numpy()  # (P, 88)

    # 2. Generate continuation
    cont_len = int(continuation_seconds * frame_rate)
    continuation_roll = generate_conditioned(model, prefix_tensor, cont_len, device, threshold)

    # 3. Build prefix MIDI (from original file, trimmed to prefix_seconds)
    original_pm = pretty_midi.PrettyMIDI(prefix_midi_path)
    prefix_pm = pretty_midi.PrettyMIDI()
    prefix_instrument = pretty_midi.Instrument(program=0, name="Piano (prefix)")
    for inst in original_pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            if note.start < prefix_seconds:
                clipped = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start,
                    end=min(note.end, prefix_seconds),
                )
                prefix_instrument.notes.append(clipped)
    prefix_pm.instruments.append(prefix_instrument)

    # 4. Build continuation MIDI from generated piano-roll, time-shifted
    cont_pm = pianoroll_to_midi(continuation_roll, frame_rate)
    combined_pm = pretty_midi.PrettyMIDI()
    combined_instrument = pretty_midi.Instrument(program=0, name="Piano")

    for note in prefix_instrument.notes:
        combined_instrument.notes.append(note)

    offset = prefix_seconds
    for inst in cont_pm.instruments:
        for note in inst.notes:
            combined_instrument.notes.append(
                pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start + offset,
                    end=note.end + offset,
                )
            )

    combined_instrument.notes.sort(key=lambda n: n.start)
    combined_pm.instruments.append(combined_instrument)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_pm.write(str(output_path))
    print(f"Saved: {output_path}")
    return output_path
