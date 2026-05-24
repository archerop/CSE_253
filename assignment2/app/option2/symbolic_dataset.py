"""MAESTRO MIDI → prefix/continuation piano-roll windows for Option 2."""

import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pretty_midi
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from app.shared.config import (
    MAESTRO_ROOT,
    MIDI_LOW,
    N_PITCHES,
    OPTION2_CACHE_DIR,
    OPTION2_CONTINUATION_SECONDS,
    OPTION2_FRAME_RATE,
    OPTION2_PREFIX_SECONDS,
    OPTION2_STRIDE_SECONDS,
)
from app.shared.metadata import load_maestro_metadata

# Type alias: (midi_path, start_frame, prefix_end_frame, cont_end_frame)
WindowSpec = Tuple[str, int, int, int]

# Directory where pre-computed piano-rolls are stored as .npy files
PIANOROLL_CACHE_DIR = OPTION2_CACHE_DIR / "pianorolls"


def _midi_to_pianoroll(midi_path: str, frame_rate: float) -> np.ndarray:
    """Return binary piano-roll of shape (T, 88), dtype float32."""
    pm = pretty_midi.PrettyMIDI(midi_path)
    total_seconds = pm.get_end_time()
    n_frames = int(np.ceil(total_seconds * frame_rate)) + 2
    roll = np.zeros((n_frames, N_PITCHES), dtype=np.float32)

    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            pitch_idx = note.pitch - MIDI_LOW
            if not (0 <= pitch_idx < N_PITCHES):
                continue
            start_f = int(note.start * frame_rate)
            end_f = min(int(note.end * frame_rate), n_frames)
            roll[start_f:end_f, pitch_idx] = 1.0

    return roll


def precache_pianorolls(
    frame_rate: float = OPTION2_FRAME_RATE,
    cache_dir: Path = PIANOROLL_CACHE_DIR,
) -> None:
    """
    Pre-compute and save all MAESTRO piano-rolls as .npy files.
    Skips files that are already cached. Safe to re-run.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    df = load_maestro_metadata(MAESTRO_ROOT)

    to_process = []
    for _, row in df.iterrows():
        midi_path = row["midi_path"]
        npy_path = cache_dir / (Path(midi_path).stem + ".npy")
        if not npy_path.exists():
            to_process.append((midi_path, npy_path))

    if not to_process:
        print(f"All {len(df)} piano-rolls already cached in {cache_dir}")
        return

    print(f"Caching {len(to_process)} piano-rolls to {cache_dir} ...")
    for midi_path, npy_path in tqdm(to_process, unit="file"):
        roll = _midi_to_pianoroll(midi_path, frame_rate)
        np.save(npy_path, roll)

    print("Pre-caching complete.")


def _load_roll(midi_path: str, frame_rate: float) -> np.ndarray:
    """Load piano-roll from .npy cache if available, else parse MIDI."""
    npy_path = PIANOROLL_CACHE_DIR / (Path(midi_path).stem + ".npy")
    if npy_path.exists():
        return np.load(npy_path)
    return _midi_to_pianoroll(midi_path, frame_rate)


def build_window_index(
    split: str = "train",
    prefix_seconds: float = OPTION2_PREFIX_SECONDS,
    continuation_seconds: float = OPTION2_CONTINUATION_SECONDS,
    stride_seconds: float = OPTION2_STRIDE_SECONDS,
    frame_rate: float = OPTION2_FRAME_RATE,
    max_windows: Optional[int] = None,
    cache_path: Optional[Path] = None,
) -> List[WindowSpec]:
    """
    Scan MAESTRO MIDIs for the given split and build a list of window specs.
    Results are cached to cache_path if provided.
    """
    if cache_path is not None and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    df = load_maestro_metadata(MAESTRO_ROOT)
    df = df[df["split"] == split].reset_index(drop=True)

    prefix_len = int(prefix_seconds * frame_rate)
    cont_len = int(continuation_seconds * frame_rate)
    window_len = prefix_len + cont_len
    stride_len = max(1, int(stride_seconds * frame_rate))

    windows: List[WindowSpec] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Indexing [{split}]"):
        midi_path = row["midi_path"]
        total_frames = int(row["duration"] * frame_rate)

        start = 0
        while start + window_len <= total_frames:
            windows.append((midi_path, start, start + prefix_len, start + window_len))
            start += stride_len
            if max_windows is not None and len(windows) >= max_windows:
                break
        if max_windows is not None and len(windows) >= max_windows:
            break

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(windows, f)

    return windows


class SymbolicDataset(Dataset):
    """
    Each item: (prefix, continuation) float32 tensors of shape (T, 88).
    Loads piano-rolls from .npy cache (fast) or parses MIDI as fallback.
    """

    def __init__(
        self,
        windows: List[WindowSpec],
        frame_rate: float = OPTION2_FRAME_RATE,
    ) -> None:
        self.windows = windows
        self.frame_rate = frame_rate

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        midi_path, start, prefix_end, cont_end = self.windows[idx]
        roll = _load_roll(midi_path, self.frame_rate)

        prefix_len = prefix_end - start
        cont_len = cont_end - prefix_end

        # Extract with zero-padding guard for edge cases
        prefix = np.zeros((prefix_len, N_PITCHES), dtype=np.float32)
        p_end = min(prefix_end, len(roll))
        prefix[: p_end - start] = roll[start:p_end]

        cont = np.zeros((cont_len, N_PITCHES), dtype=np.float32)
        c_start = min(prefix_end, len(roll))
        c_end = min(cont_end, len(roll))
        if c_end > c_start:
            cont[: c_end - c_start] = roll[c_start:c_end]

        return torch.from_numpy(prefix), torch.from_numpy(cont)


def get_datasets(
    train_max: Optional[int] = None,
    val_max: Optional[int] = None,
) -> Tuple[SymbolicDataset, SymbolicDataset]:
    """Return (train_dataset, val_dataset), building window indices with caching."""
    OPTION2_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    train_windows = build_window_index(
        split="train",
        max_windows=train_max,
        cache_path=OPTION2_CACHE_DIR / "train_windows.pkl",
    )
    val_windows = build_window_index(
        split="validation",
        max_windows=val_max,
        cache_path=OPTION2_CACHE_DIR / "val_windows.pkl",
    )

    return SymbolicDataset(train_windows), SymbolicDataset(val_windows)
