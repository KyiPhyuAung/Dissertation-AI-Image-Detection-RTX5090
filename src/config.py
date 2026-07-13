from pathlib import Path
import torch

# =============================================================================
# Project Directories
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "datasets"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# =============================================================================
# Dataset Locations
# =============================================================================

# Keep both datasets available. Update only these paths if folders are moved.
TINY_GENIMAGE_DIR = DATASET_DIR / "tiny-genimage"
TIGAS_DIR = DATASET_DIR / "TIGAS"

# =============================================================================
# Hardware
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 8
PIN_MEMORY = DEVICE.type == "cuda"
PREFETCH_FACTOR = 4
PERSISTENT_WORKERS = NUM_WORKERS > 0

# RTX/CUDA acceleration. These do not change the model architecture.
USE_AMP = DEVICE.type == "cuda"
USE_TF32 = DEVICE.type == "cuda"
CUDNN_BENCHMARK = DEVICE.type == "cuda"
CHANNELS_LAST = DEVICE.type == "cuda"

# =============================================================================
# Dataset / Training
# =============================================================================

IMAGE_SIZE = 384
NUM_CLASSES = 2
VALIDATION_FRACTION = 0.10  # Tiny GenImage only; TIGAS uses its official val split.

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
RANDOM_SEED = 42
CHECKPOINT_METRIC = "val_loss"
