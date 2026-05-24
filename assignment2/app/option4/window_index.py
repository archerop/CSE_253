from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


REQUIRED_METADATA_COLUMNS = [
    "piece_id",
    "split",
    "composer",
    "title",
    "duration",
    "midi_path",
    "audio_path",
    "midi_exists",
    "audio_exists",
]


def _validate_metadata_columns(metadata: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_METADATA_COLUMNS if col not in metadata.columns]
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")


def build_window_index_for_split(
    metadata: pd.DataFrame,
    split: str,
    clip_seconds: float,
    stride_seconds: float,
    max_windows: Optional[int] = None,
    seed: int = 42,
    drop_incomplete_last_window: bool = True,
) -> pd.DataFrame:
    """
    Build an aligned MIDI/audio window index for one MAESTRO split.

    Each row corresponds to one training/evaluation example:
        MIDI window [start_sec, end_sec]
        audio window [start_sec, end_sec]

    This function only builds a CSV-friendly index. It does not compute or save
    piano-roll/log-mel tensors.
    """
    _validate_metadata_columns(metadata)

    if clip_seconds <= 0:
        raise ValueError(f"clip_seconds must be positive, got {clip_seconds}")

    if stride_seconds <= 0:
        raise ValueError(f"stride_seconds must be positive, got {stride_seconds}")

    split_df = metadata[
        (metadata["split"] == split)
        & (metadata["midi_exists"] == True)
        & (metadata["audio_exists"] == True)
        & (metadata["duration"] >= clip_seconds)
    ].copy()

    rows = []

    for _, row in split_df.iterrows():
        duration = float(row["duration"])

        if drop_incomplete_last_window:
            # Last valid start must allow a complete clip.
            last_start = duration - clip_seconds
        else:
            # Allows windows that may need padding at the end.
            last_start = duration

        if last_start < 0:
            continue

        starts = np.arange(0.0, last_start + 1e-6, stride_seconds)

        for start_sec in starts:
            end_sec = float(start_sec + clip_seconds)

            window_id = (
                f"option4_{split}_"
                f"{row['piece_id']}_"
                f"{int(round(start_sec * 1000)):010d}ms"
            )

            rows.append(
                {
                    "window_id": window_id,
                    "piece_id": row["piece_id"],
                    "split": split,
                    "composer": row["composer"],
                    "title": row["title"],
                    "duration": duration,
                    "midi_path": row["midi_path"],
                    "audio_path": row["audio_path"],
                    "start_sec": float(start_sec),
                    "end_sec": end_sec,
                    "clip_seconds": float(clip_seconds),
                    "stride_seconds": float(stride_seconds),
                }
            )

    index = pd.DataFrame(rows)

    if len(index) == 0:
        raise ValueError(
            f"No windows generated for split={split!r}. "
            f"Check clip_seconds={clip_seconds}, stride_seconds={stride_seconds}."
        )

    # Deterministic subset sampling.
    if max_windows is not None:
        if max_windows <= 0:
            raise ValueError(f"max_windows must be positive or None, got {max_windows}")

        if len(index) > max_windows:
            index = index.sample(n=max_windows, random_state=seed).copy()

            # Sort after sampling for stable reading/debugging.
            index = index.sort_values(
                by=["piece_id", "start_sec", "window_id"]
            ).reset_index(drop=True)
        else:
            index = index.reset_index(drop=True)
    else:
        index = index.reset_index(drop=True)

    return index


def summarize_window_index(index: pd.DataFrame) -> Dict[str, float | int]:
    """
    Return compact summary statistics for a window index.
    """
    if len(index) == 0:
        return {
            "num_windows": 0,
            "num_pieces": 0,
            "total_window_hours": 0.0,
            "clip_seconds": 0.0,
            "stride_seconds": 0.0,
        }

    total_seconds = float(index["clip_seconds"].sum())

    return {
        "num_windows": int(len(index)),
        "num_pieces": int(index["piece_id"].nunique()),
        "total_window_hours": total_seconds / 3600.0,
        "clip_seconds": float(index["clip_seconds"].iloc[0]),
        "stride_seconds": float(index["stride_seconds"].iloc[0]),
        "min_start_sec": float(index["start_sec"].min()),
        "max_start_sec": float(index["start_sec"].max()),
    }


def save_window_index(index: pd.DataFrame, output_path: str | Path) -> None:
    """
    Save window index as CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(output_path, index=False)


def load_window_index(index_csv: str | Path) -> pd.DataFrame:
    """
    Load a previously generated window index CSV.
    """
    index_csv = Path(index_csv)

    if not index_csv.exists():
        raise FileNotFoundError(f"Window index CSV not found: {index_csv}")

    index = pd.read_csv(index_csv)

    required = [
        "window_id",
        "piece_id",
        "split",
        "midi_path",
        "audio_path",
        "start_sec",
        "end_sec",
        "clip_seconds",
    ]

    missing = [col for col in required if col not in index.columns]
    if missing:
        raise ValueError(f"Window index is missing required columns: {missing}")

    return index
