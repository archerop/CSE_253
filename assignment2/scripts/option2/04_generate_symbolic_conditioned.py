"""
Generate symbolic_conditioned.mid using the trained Option 2 model.

Run from assignment2/:
    python scripts/option2/04_generate_symbolic_conditioned.py

Requires:
  - Trained checkpoint at outputs/checkpoints/option2_best.pt
  - MAESTRO dataset at data/maestro-v3.0.0/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from app.shared.config import (
    CHECKPOINT_DIR,
    MAESTRO_ROOT,
    OPTION2_CONTINUATION_SECONDS,
    OPTION2_FRAME_RATE,
    OPTION2_OUTPUT_DIR,
    OPTION2_PREFIX_SECONDS,
)
from app.shared.metadata import load_maestro_metadata, validate_maestro_paths
from app.option2.symbolic_dataset import _midi_to_pianoroll
from app.option2.symbolic_generate import (
    extract_prefix,
    generate_conditioned,
    save_symbolic_conditioned,
)
from app.option2.symbolic_models import CopyLastFrameBaseline, SymbolicTransformer
from app.option2.symbolic_train import load_best_checkpoint
from app.option2.symbolic_eval import evaluate_generation, print_metrics


def pick_test_midi() -> str:
    """Pick a test-split MIDI that exists on disk."""
    df = load_maestro_metadata(MAESTRO_ROOT)
    df = validate_maestro_paths(df)
    test_df = df[(df["split"] == "test") & df["midi_exists"]].reset_index(drop=True)
    if len(test_df) == 0:
        raise FileNotFoundError("No test-split MIDI files found. Is the dataset downloaded?")
    # Pick a mid-length piece for a representative example
    test_df = test_df.sort_values("duration")
    idx = len(test_df) // 2
    row = test_df.iloc[idx]
    print(f"Selected: {row['composer']} — {row['title']}  ({row['duration']:.1f}s)")
    return row["midi_path"]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    midi_path = pick_test_midi()

    # --- Baseline generation ---
    print("\n[Baseline] Copy-last-frame...")
    prefix_tensor = extract_prefix(midi_path)
    cont_len = int(OPTION2_CONTINUATION_SECONDS * OPTION2_FRAME_RATE)
    baseline = CopyLastFrameBaseline()
    baseline_roll = baseline(prefix_tensor.unsqueeze(0), cont_len).squeeze(0).numpy()

    baseline_out = OPTION2_OUTPUT_DIR / "symbolic_conditioned_baseline.mid"
    from app.option2.symbolic_generate import pianoroll_to_midi, save_symbolic_conditioned
    import pretty_midi
    bl_pm = pianoroll_to_midi(baseline_roll)
    OPTION2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bl_pm.write(str(baseline_out))
    print(f"Baseline MIDI saved: {baseline_out}")

    # --- Model generation ---
    ckpt_path = CHECKPOINT_DIR / "option2_best.pt"
    if not ckpt_path.exists():
        print(f"\nNo checkpoint found at {ckpt_path}.")
        print("Run  python scripts/option2/03_train_option2_model.py  first.")
        return

    print(f"\n[Model] Loading checkpoint from {ckpt_path}...")
    model = SymbolicTransformer()
    model = load_best_checkpoint(model, ckpt_path, device)

    output_path = save_symbolic_conditioned(
        prefix_midi_path=midi_path,
        model=model,
        device=device,
    )

    # --- Evaluation vs ground truth ---
    print("\n[Eval] Comparing model and baseline vs ground truth continuation...")
    roll = _midi_to_pianoroll(midi_path, OPTION2_FRAME_RATE)
    prefix_end = int(OPTION2_PREFIX_SECONDS * OPTION2_FRAME_RATE)
    gt_roll = roll[prefix_end : prefix_end + cont_len]

    model_roll = generate_conditioned(model, prefix_tensor, cont_len, device)

    print("\n-- Model metrics --")
    print_metrics(evaluate_generation(model_roll, gt_roll))

    print("-- Baseline metrics --")
    print_metrics(evaluate_generation(baseline_roll, gt_roll))

    print(f"\nFinal output: {output_path}")


if __name__ == "__main__":
    main()
