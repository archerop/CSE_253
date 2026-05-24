from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "canonical_composer",
    "canonical_title",
    "split",
    "year",
    "midi_filename",
    "audio_filename",
    "duration",
]


def load_maestro_metadata(maestro_root: Path) -> pd.DataFrame:
    """
    Load MAESTRO v3.0.0 metadata and resolve absolute MIDI/audio paths.

    Returns a dataframe with:
    - piece_id
    - split
    - composer
    - title
    - year
    - duration
    - midi_path
    - audio_path
    """
    metadata_path = maestro_root / "maestro-v3.0.0.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found: {metadata_path}\n"
            f"Expected MAESTRO root to be: {maestro_root}"
        )

    df = pd.read_csv(metadata_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")

    out = pd.DataFrame()
    out["piece_id"] = [f"maestro_{i:04d}" for i in range(len(df))]
    out["split"] = df["split"].astype(str)
    out["composer"] = df["canonical_composer"].astype(str)
    out["title"] = df["canonical_title"].astype(str)
    out["year"] = df["year"]
    out["duration"] = df["duration"].astype(float)

    out["midi_path"] = df["midi_filename"].apply(
        lambda p: str((maestro_root / p).resolve())
    )
    out["audio_path"] = df["audio_filename"].apply(
        lambda p: str((maestro_root / p).resolve())
    )

    return out


def validate_maestro_paths(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean columns indicating whether MIDI/audio paths exist.
    """
    checked = df.copy()
    checked["midi_exists"] = checked["midi_path"].apply(lambda p: Path(p).exists())
    checked["audio_exists"] = checked["audio_path"].apply(lambda p: Path(p).exists())
    return checked


def summarize_metadata(df: pd.DataFrame) -> None:
    """
    Print a compact metadata summary.
    """
    print("Number of rows:", len(df))
    print()

    print("Split counts:")
    print(df["split"].value_counts())
    print()

    print("Duration by split, in hours:")
    print((df.groupby("split")["duration"].sum() / 3600).round(2))
    print()

    print("Duration statistics, in seconds:")
    print(df["duration"].describe().round(2))
    print()

    print("Example rows:")
    cols = ["piece_id", "split", "composer", "title", "year", "duration"]
    print(df[cols].head(5).to_string(index=False))
