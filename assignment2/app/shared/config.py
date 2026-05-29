from pathlib import Path

# Project root: ~/CSE_253/assignment2
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MAESTRO_ROOT = DATA_DIR / "maestro-v3.0.0"

METADATA_CSV = MAESTRO_ROOT / "maestro-v3.0.0.csv"
METADATA_JSON = MAESTRO_ROOT / "maestro-v3.0.0.json"

CACHE_DIR = PROJECT_ROOT / "cache"
METADATA_CACHE_DIR = CACHE_DIR / "metadata"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
AUDIO_OUTPUT_DIR = OUTPUT_DIR / "audio"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# ---------------------------------------------------------------------
# MIDI/audio preprocessing parameters for Option 4
# ---------------------------------------------------------------------

SAMPLE_RATE = 22050
HOP_LENGTH = 256
FRAME_RATE = SAMPLE_RATE / HOP_LENGTH

# Full piano range: A0 to C8
MIDI_LOW = 21
MIDI_HIGH = 108
N_PITCHES = MIDI_HIGH - MIDI_LOW + 1

# Default local window length for MIDI/audio aligned examples
CLIP_SECONDS = 4.0

# We mark onset over a small number of frames to make the feature robust
# to frame quantization.
ONSET_WIDTH_FRAMES = 2

# ---------------------------------------------------------------------
# Audio preprocessing parameters for Option 4
# ---------------------------------------------------------------------

N_FFT = 1024
WIN_LENGTH = 1024
N_MELS = 80

# Piano fundamental range is roughly 27.5 Hz to 4186 Hz, but piano
# harmonics extend higher. We keep fmax at Nyquist for a compact but
# still broad log-mel target.
FMIN = 30.0
FMAX = SAMPLE_RATE / 2

# Use centered STFT frames so each spectrogram frame is centered around
# the corresponding time step. This makes the frame count naturally
# match the MIDI piano-roll grid when using the same sample_rate/hop_length.
CENTER = True

# log(1 + mel) scaling for stable regression target.
LOG_EPS = 1e-6

# ---------------------------------------------------------------------
# Option 4 window-index / dataset defaults
# ---------------------------------------------------------------------

DEFAULT_SUBSET_NAME = "debug"

DEFAULT_CLIP_SECONDS = 4.0
DEFAULT_STRIDE_SECONDS = 4.0

DEFAULT_TRAIN_MAX_WINDOWS = 2000
DEFAULT_VAL_MAX_WINDOWS = 400
DEFAULT_TEST_MAX_WINDOWS = 400

DEFAULT_RANDOM_SEED = 42

# ---------------------------------------------------------------------
# Shared project layout constants
# ---------------------------------------------------------------------

SHARED_OUTPUT_DIR = OUTPUT_DIR / "shared"
OPTION2_OUTPUT_DIR = OUTPUT_DIR / "option2"
OPTION4_OUTPUT_DIR = OUTPUT_DIR / "option4"
FINAL_OUTPUT_DIR = OUTPUT_DIR / "final"

WINDOW_INDEX_CACHE_DIR = CACHE_DIR / "window_index"
OPTION2_CACHE_DIR = CACHE_DIR / "option2"
OPTION4_CACHE_DIR = CACHE_DIR / "option4"

# ---------------------------------------------------------------------
# Option 2: Symbolic conditioned generation constants
# ---------------------------------------------------------------------

OPTION2_FRAME_RATE = 40.0        # fps for symbolic piano-roll (~25 ms per frame)
OPTION2_PREFIX_SECONDS = 4.0     # prefix duration fed as conditioning
OPTION2_CONTINUATION_SECONDS = 4.0  # continuation duration to generate
OPTION2_STRIDE_SECONDS = 2.0     # stride between consecutive windows

OPTION2_D_MODEL = 256            # was 128 — width bump (Option A), ~3.2M params
OPTION2_NHEAD = 8                # was 4  — keep d_model/nhead = 32
OPTION2_NUM_LAYERS = 4           # unchanged — depth preserved, scale width first
OPTION2_DIM_FEEDFORWARD = 1024   # was 512 — keep 4× d_model ratio
OPTION2_DROPOUT = 0.1

OPTION2_BATCH_SIZE = 32
OPTION2_LEARNING_RATE = 3e-4     # was 1e-3 — lower LR for stabler training
OPTION2_WEIGHT_DECAY = 1e-4
OPTION2_MAX_EPOCHS = 60       # bumped from 50 to give cosine LR schedule room to anneal
OPTION2_WARMUP_EPOCHS = 3     # ~5% of max_epochs — linear warmup before cosine decay
OPTION2_PATIENCE = 12         # was 8 — bumped so cosine schedule can finish without early-stopping

OPTION2_PREFIX_MAX_LEN = 256     # max tokens for 4s prefix
OPTION2_CONT_MAX_LEN   = 256     # max tokens for 4s continuation
OPTION2_MAX_SEQ_LEN    = 512     # PREFIX + CONT fed to model
OPTION2_GPT2_N_LAYER   = 6
OPTION2_GPT2_N_HEAD    = 8
OPTION2_GPT2_N_EMBD    = 384

OPTION2_MODEL_TYPE     = 'transformer'  # 'lstm' | 'gru' | 'transformer' | 'gpt2'
OPTION2_HIDDEN_SIZE    = 256            # hidden dim for LSTM / GRU
OPTION2_RNN_LAYERS     = 2             # depth for LSTM / GRU

