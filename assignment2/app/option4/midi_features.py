from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pretty_midi


CHANNEL_NAMES = ["active", "onset", "velocity_onset"]


def _pitch_to_index(pitch: int, midi_low: int) -> int:
    return int(pitch) - int(midi_low)


def midi_to_pianoroll_features(
    midi_path: str | Path,
    frame_rate: float,
    midi_low: int = 21,
    midi_high: int = 108,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    clip_seconds: Optional[float] = None,
    onset_width_frames: int = 2,
) -> Tuple[np.ndarray, Dict[str, float | int | str]]:
    """
    Convert a MIDI file into 3-channel piano-roll conditioning features.

    Channels:
    0. active:
       active[t, p] = 1 if pitch p is sounding at frame t.

    1. onset:
       onset[t, p] = 1 if pitch p starts near frame t.

    2. velocity_onset:
       velocity_onset[t, p] = velocity / 127 at note onset frames.

    Output shape:
        features: float32 array with shape (3, T, 88)

    Notes:
    - The time grid is controlled by frame_rate.
    - This function can extract only a local window, which avoids creating
      large full-piece tensors when we only need a training example.
    """
    midi_path = Path(midi_path)

    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    if clip_seconds is not None and end_sec is not None:
        raise ValueError("Specify either clip_seconds or end_sec, not both.")

    if start_sec < 0:
        raise ValueError(f"start_sec must be non-negative, got {start_sec}")

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    midi_end_time = float(pm.get_end_time())

    if clip_seconds is not None:
        if clip_seconds <= 0:
            raise ValueError(f"clip_seconds must be positive, got {clip_seconds}")
        end_sec = start_sec + clip_seconds
    elif end_sec is None:
        end_sec = midi_end_time

    assert end_sec is not None

    if end_sec <= start_sec:
        raise ValueError(
            f"end_sec must be greater than start_sec, got {start_sec=} {end_sec=}"
        )

    # The requested window can extend past the MIDI end. This is okay:
    # the tail will simply remain all-zero.
    window_duration = float(end_sec - start_sec)
    num_frames = int(np.ceil(window_duration * frame_rate))
    num_pitches = int(midi_high - midi_low + 1)

    active = np.zeros((num_frames, num_pitches), dtype=np.float32)
    onset = np.zeros((num_frames, num_pitches), dtype=np.float32)
    velocity_onset = np.zeros((num_frames, num_pitches), dtype=np.float32)

    note_count_total = 0
    note_count_used = 0

    for instrument in pm.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            note_count_total += 1

            pitch = int(note.pitch)
            if pitch < midi_low or pitch > midi_high:
                continue
            if note.end <= start_sec or note.start >= end_sec:
                continue
            if note.end <= note.start:
                continue

            pitch_idx = _pitch_to_index(pitch, midi_low)

            # Active note frames, clipped to the requested window.
            rel_start = max(float(note.start) - start_sec, 0.0)
            rel_end = min(float(note.end) - start_sec, window_duration)

            start_frame = int(np.floor(rel_start * frame_rate))
            end_frame = int(np.ceil(rel_end * frame_rate))

            start_frame = max(0, min(start_frame, num_frames))
            end_frame = max(0, min(end_frame, num_frames))

            if end_frame > start_frame:
                active[start_frame:end_frame, pitch_idx] = 1.0
                note_count_used += 1

            # Onset is only marked if the note starts inside the window.
            if start_sec <= note.start < end_sec:
                rel_onset = float(note.start) - start_sec
                onset_frame = int(np.floor(rel_onset * frame_rate))
                onset_frame = max(0, min(onset_frame, num_frames - 1))

                onset_end = min(num_frames, onset_frame + max(1, onset_width_frames))
                onset[onset_frame:onset_end, pitch_idx] = 1.0

                vel = float(note.velocity) / 127.0
                # Use max in case overlapping notes of same pitch map to same frame.
                velocity_onset[onset_frame:onset_end, pitch_idx] = np.maximum(
                    velocity_onset[onset_frame:onset_end, pitch_idx],
                    vel,
                )

    features = np.stack([active, onset, velocity_onset], axis=0).astype(np.float32)

    metadata: Dict[str, float | int | str] = {
        "midi_path": str(midi_path),
        "midi_end_time": midi_end_time,
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
        "window_duration": float(window_duration),
        "frame_rate": float(frame_rate),
        "num_frames": int(num_frames),
        "midi_low": int(midi_low),
        "midi_high": int(midi_high),
        "num_pitches": int(num_pitches),
        "onset_width_frames": int(onset_width_frames),
        "note_count_total": int(note_count_total),
        "note_count_used_in_window": int(note_count_used),
    }

    return features, metadata


def summarize_pianoroll_features(features: np.ndarray) -> Dict[str, float | int]:
    """
    Compute simple sanity-check statistics for a 3-channel piano-roll tensor.

    Expected shape: (3, T, 88)
    """
    if features.ndim != 3:
        raise ValueError(f"Expected features with 3 dims (C, T, P), got {features.shape}")
    if features.shape[0] != 3:
        raise ValueError(f"Expected 3 channels, got {features.shape[0]}")

    active = features[0]
    onset = features[1]
    velocity_onset = features[2]

    active_notes_per_frame = active.sum(axis=1)
    onsets_per_frame = onset.sum(axis=1)

    nonzero_velocity = velocity_onset[velocity_onset > 0]

    stats: Dict[str, float | int] = {
        "channels": int(features.shape[0]),
        "num_frames": int(features.shape[1]),
        "num_pitches": int(features.shape[2]),
        "active_density": float(active.mean()),
        "onset_density": float(onset.mean()),
        "velocity_onset_density": float((velocity_onset > 0).mean()),
        "avg_active_notes_per_frame": float(active_notes_per_frame.mean()),
        "max_active_notes_per_frame": int(active_notes_per_frame.max()),
        "avg_onsets_per_frame": float(onsets_per_frame.mean()),
        "max_onsets_per_frame": int(onsets_per_frame.max()),
        "num_active_cells": int(active.sum()),
        "num_onset_cells": int(onset.sum()),
    }

    if nonzero_velocity.size > 0:
        stats.update(
            {
                "velocity_min_nonzero": float(nonzero_velocity.min()),
                "velocity_mean_nonzero": float(nonzero_velocity.mean()),
                "velocity_max_nonzero": float(nonzero_velocity.max()),
            }
        )
    else:
        stats.update(
            {
                "velocity_min_nonzero": 0.0,
                "velocity_mean_nonzero": 0.0,
                "velocity_max_nonzero": 0.0,
            }
        )

    return stats


def validate_pianoroll_features(features: np.ndarray) -> None:
    """
    Raise an error if the feature tensor has an invalid shape or invalid values.
    """
    if features.ndim != 3:
        raise ValueError(f"Expected shape (3, T, 88), got {features.shape}")

    channels, num_frames, num_pitches = features.shape

    if channels != 3:
        raise ValueError(f"Expected 3 channels, got {channels}")

    if num_frames <= 0:
        raise ValueError(f"Expected positive num_frames, got {num_frames}")

    if num_pitches != 88:
        raise ValueError(f"Expected 88 pitches, got {num_pitches}")

    if not np.isfinite(features).all():
        raise ValueError("Feature tensor contains NaN or Inf.")

    if features.min() < 0.0 or features.max() > 1.0:
        raise ValueError(
            f"Expected feature values in [0, 1], got min={features.min()}, max={features.max()}"
        )


def save_pianoroll_npz(
    output_path: str | Path,
    features: np.ndarray,
    metadata: Dict[str, float | int | str],
    channel_names: Optional[List[str]] = None,
) -> None:
    """
    Save a piano-roll feature tensor and metadata to a compressed .npz file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if channel_names is None:
        channel_names = CHANNEL_NAMES

    np.savez_compressed(
        output_path,
        features=features.astype(np.float32),
        channel_names=np.array(channel_names),
        metadata_keys=np.array(list(metadata.keys())),
        metadata_values=np.array([str(v) for v in metadata.values()]),
    )
