"""MAESTRO MIDI → REMI token prefix/continuation windows for Option 2."""

import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import symusic
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from miditok import REMI, TokenizerConfig

from app.shared.config import (
    MAESTRO_ROOT,
    OPTION2_CACHE_DIR,
    OPTION2_CONTINUATION_SECONDS,
    OPTION2_CONT_MAX_LEN,
    OPTION2_PREFIX_MAX_LEN,
    OPTION2_PREFIX_SECONDS,
    OPTION2_STRIDE_SECONDS,
)
from app.shared.metadata import load_maestro_metadata

WindowSpec = Tuple[str, int, int, int]  # (midi_path, start_tok, prefix_end_tok, cont_end_tok)

TOKEN_CACHE_DIR = OPTION2_CACHE_DIR / "tokens"
TOKENIZER_DIR   = OPTION2_CACHE_DIR / "tokenizer"

_TOKENIZER_CONFIG = TokenizerConfig(
    pitch_range=(21, 109),
    beat_res={(0, 4): 8, (4, 12): 4},
    num_velocities=32,
    special_tokens=["PAD", "BOS", "EOS"],
    use_chords=False,
    use_rests=True,
    use_tempos=True,
    use_time_signatures=False,
    use_programs=False,
)


def build_tokenizer() -> REMI:
    """Build (or load saved) REMI tokenizer."""
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tok_file = TOKENIZER_DIR / "tokenizer.json"
    if tok_file.exists():
        return REMI(params=tok_file)
    tokenizer = REMI(_TOKENIZER_CONFIG)
    tokenizer.save_pretrained(str(TOKENIZER_DIR))
    return tokenizer


def _trim_score(score: symusic.Score, start_s: float, end_s: float) -> symusic.Score:
    """Return a symusic.Score clipped to [start_s, end_s) with time shifted to 0."""
    tpq = score.ticks_per_quarter
    qpm = score.tempos[0].qpm if score.tempos else 120.0
    ticks_per_sec = qpm * tpq / 60.0
    start_tick = int(start_s * ticks_per_sec)
    end_tick   = int(end_s   * ticks_per_sec)
    clipped = score.clip(start_tick, end_tick)
    return clipped.shift_time(-start_tick)


def _tokenize_midi(midi_path: str, tokenizer: REMI) -> List[int]:
    """Tokenize the entire MIDI file and return flat token id list."""
    score = symusic.Score(midi_path)
    seqs = tokenizer.encode(score)
    return seqs[0].ids if seqs else []


def precache_tokens(tokenizer: REMI, cache_dir: Path = TOKEN_CACHE_DIR) -> None:
    """Pre-tokenize all MAESTRO MIDIs and save per-file .pkl caches."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    df = load_maestro_metadata(MAESTRO_ROOT)

    to_process = []
    for _, row in df.iterrows():
        midi_path = row["midi_path"]
        pkl_path = cache_dir / (Path(midi_path).stem + ".pkl")
        if not pkl_path.exists():
            to_process.append((midi_path, pkl_path))

    if not to_process:
        print(f"All {len(df)} token caches already exist in {cache_dir}")
        return

    print(f"Tokenizing {len(to_process)} MIDI files → {cache_dir} ...")
    for midi_path, pkl_path in tqdm(to_process, unit="file"):
        tokens = _tokenize_midi(midi_path, tokenizer)
        with open(pkl_path, "wb") as f:
            pickle.dump(tokens, f)
    print("Token pre-caching complete.")


def _load_tokens(midi_path: str, tokenizer: REMI) -> List[int]:
    """Load token ids from .pkl cache, or tokenize on the fly."""
    pkl_path = TOKEN_CACHE_DIR / (Path(midi_path).stem + ".pkl")
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    return _tokenize_midi(midi_path, tokenizer)


def _pad_or_truncate(seq: List[int], max_len: int, pad_id: int = 0) -> List[int]:
    seq = seq[:max_len]
    return seq + [pad_id] * (max_len - len(seq))


def build_token_window_index(
    split: str,
    tokenizer: REMI,
    prefix_seconds: float = OPTION2_PREFIX_SECONDS,
    continuation_seconds: float = OPTION2_CONTINUATION_SECONDS,
    stride_seconds: float = OPTION2_STRIDE_SECONDS,
    max_windows: Optional[int] = None,
    cache_path: Optional[Path] = None,
) -> List[WindowSpec]:
    """
    Slide windows over each MIDI file's token sequence.
    Each window: (midi_path, start_tok, prefix_end_tok, cont_end_tok).
    Window size = PREFIX_MAX_LEN + CONT_MAX_LEN; stride = CONT_MAX_LEN // 2.
    """
    if cache_path is not None and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    window_len = OPTION2_PREFIX_MAX_LEN + OPTION2_CONT_MAX_LEN
    stride_len = max(1, OPTION2_CONT_MAX_LEN // 2)

    df = load_maestro_metadata(MAESTRO_ROOT)
    df = df[df["split"] == split].reset_index(drop=True)

    windows: List[WindowSpec] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Indexing [{split}]"):
        midi_path = row["midi_path"]
        tokens = _load_tokens(midi_path, tokenizer)
        total = len(tokens)

        start = 0
        while start + window_len <= total:
            windows.append((
                midi_path,
                start,
                start + OPTION2_PREFIX_MAX_LEN,
                start + window_len,
            ))
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
    Each item: (prefix_ids, cont_ids) — LongTensors of shape (PREFIX_MAX_LEN,) and (CONT_MAX_LEN,).
    """

    def __init__(self, windows: List[WindowSpec], tokenizer: REMI) -> None:
        self.windows   = windows
        self.tokenizer = tokenizer
        self.pad_id    = tokenizer["PAD_None"]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        midi_path, start, prefix_end, cont_end = self.windows[idx]
        tokens = _load_tokens(midi_path, self.tokenizer)

        prefix_ids = _pad_or_truncate(tokens[start:prefix_end], OPTION2_PREFIX_MAX_LEN, self.pad_id)
        cont_ids   = _pad_or_truncate(tokens[prefix_end:cont_end], OPTION2_CONT_MAX_LEN, self.pad_id)

        return torch.tensor(prefix_ids, dtype=torch.long), torch.tensor(cont_ids, dtype=torch.long)


def get_datasets(
    tokenizer: REMI,
    train_max: Optional[int] = None,
    val_max: Optional[int] = None,
) -> Tuple[SymbolicDataset, SymbolicDataset]:
    """Return (train_dataset, val_dataset) with cached token window indices."""
    OPTION2_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    train_windows = build_token_window_index(
        split="train",
        tokenizer=tokenizer,
        max_windows=train_max,
        cache_path=OPTION2_CACHE_DIR / "train_token_windows.pkl",
    )
    val_windows = build_token_window_index(
        split="validation",
        tokenizer=tokenizer,
        max_windows=val_max,
        cache_path=OPTION2_CACHE_DIR / "val_token_windows.pkl",
    )

    return SymbolicDataset(train_windows, tokenizer), SymbolicDataset(val_windows, tokenizer)
