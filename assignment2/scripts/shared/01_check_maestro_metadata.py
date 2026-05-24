from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.shared.config import MAESTRO_ROOT, METADATA_CACHE_DIR
from app.shared.metadata import (
    load_maestro_metadata,
    validate_maestro_paths,
    summarize_metadata,
)


def main() -> None:
    print("=" * 80)
    print("Step 1: Check MAESTRO metadata")
    print("=" * 80)
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"MAESTRO_ROOT = {MAESTRO_ROOT}")
    print()

    df = load_maestro_metadata(MAESTRO_ROOT)
    summarize_metadata(df)

    checked = validate_maestro_paths(df)

    missing_midi = int((~checked["midi_exists"]).sum())
    missing_audio = int((~checked["audio_exists"]).sum())

    print()
    print("Path validation:")
    print(f"Missing MIDI files:  {missing_midi}")
    print(f"Missing audio files: {missing_audio}")

    if missing_midi > 0:
        print()
        print("Examples of missing MIDI files:")
        print(
            checked.loc[~checked["midi_exists"], ["piece_id", "midi_path"]]
            .head(10)
            .to_string(index=False)
        )

    if missing_audio > 0:
        print()
        print("Examples of missing audio files:")
        print(
            checked.loc[~checked["audio_exists"], ["piece_id", "audio_path"]]
            .head(10)
            .to_string(index=False)
        )

    if missing_midi == 0 and missing_audio == 0:
        print()
        print("All MIDI and audio paths exist.")

    METADATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = METADATA_CACHE_DIR / "maestro_resolved_metadata.csv"
    checked.to_csv(output_path, index=False)

    print()
    print(f"Saved resolved metadata to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
