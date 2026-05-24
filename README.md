# CSE 253 Assignment 2 Project Repo

This repo is for CSE 253 Assignment 2. Our project uses two task options:

- Option 2: symbolic conditioned generation
- Option 4: continuous conditioned generation

The assignment requires two generated music files, a clean exported notebook, and a presentation video. The final task files should match the selected options. :contentReference[oaicite:0]{index=0}

## Current repo structure

The repo is organized so that shared dataset/config code is separated from option-specific code.

```text
assignment2/
  app/
    shared/
      config.py
      metadata.py

    option2/
      __init__.py

    option4/
      midi_features.py
      audio_preprocessing.py
      window_index.py
      option4_dataset.py

  scripts/
    shared/
      01_check_maestro_metadata.py

    option2/

    option4/
      02_check_midi_features.py
      03_check_audio_preprocessing.py
      04_build_option4_window_index.py
      05_check_option4_dataset.py

  data/
  cache/
  outputs/
````

## Shared code

Only the truly shared code should go under:

```text
app/shared/
scripts/shared/
```

Currently shared means:

```text
app/shared/config.py
app/shared/metadata.py
```

These files contain:

* project paths
* MAESTRO dataset path
* cache/output directory paths
* metadata loading
* metadata validation
* official train / validation / test split handling

Please avoid putting option-specific preprocessing, models, training code, or evaluation code inside `app/shared/`.

## Dataset setup

We use MAESTRO v3.0.0.

Each teammate should download the dataset locally. The expected location is:

```text
~/CSE_253/assignment2/data/maestro-v3.0.0/
```

After setup, the folder should look like:

```text
data/
  maestro-v3.0.0/
    maestro-v3.0.0.csv
    maestro-v3.0.0.json
    2004/
    2006/
    2008/
    2011/
    2013/
    2014/
    2015/
    2017/
    2018/
```

Download commands:

```bash
cd ~/CSE_253/assignment2
mkdir -p data/downloads
cd data/downloads

wget -c https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip

cd ~/CSE_253/assignment2
unzip data/downloads/maestro-v3.0.0.zip -d data
```

After unzipping, the full dataset zip can be removed to save disk space:

```bash
rm data/downloads/maestro-v3.0.0.zip
```

Then run the metadata check:

```bash
cd ~/CSE_253/assignment2
source .venv/bin/activate

python scripts/shared/01_check_maestro_metadata.py
```

Expected result:

```text
Missing MIDI files:  0
Missing audio files: 0
```

This will create:

```text
cache/metadata/maestro_resolved_metadata.csv
```

Both Option 2 and Option 4 should use this metadata file rather than writing separate metadata loaders.

## Option 2 code location

Option 2 code should go under:

```text
app/option2/
scripts/option2/
outputs/option2/
cache/option2/
```

Suggested files for Option 2:

```text
app/option2/
  symbolic_dataset.py
  symbolic_models.py
  symbolic_train.py
  symbolic_generate.py
  symbolic_eval.py

scripts/option2/
  build_option2_windows.py
  check_option2_dataset.py
  train_option2_model.py
  generate_symbolic_conditioned.py
```

Option 2 should not modify Option 4 files.

## Option 4 code location

Option 4 code is under:

```text
app/option4/
scripts/option4/
outputs/option4/
cache/option4/
```

Current Option 4 pipeline includes:

```text
MIDI → piano-roll features
audio → log-mel spectrogram
aligned MIDI/audio window index
Option4Dataset / DataLoader
```

## Development rule

Shared files should stay minimal.

Use shared only for:

```text
dataset path
metadata loading
global project/cache/output paths
```

Do not put the following in shared:

```text
Option 2 model code
Option 4 model code
MIDI/audio preprocessing specific to one option
training scripts
evaluation scripts
generation scripts
```

This keeps Option 2 and Option 4 development independent.

