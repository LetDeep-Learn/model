
import os

# ----------------------------
# Dataset paths
# ----------------------------
# DATA_ROOT = "dataset"  
DATA_ROOT = "dataset"            # original paired dataset (photos + sketches)
          # original paired dataset (photos + sketches)
SKETCH_ROOT = "dataset_unpaired/sketches" # sketches-only dataset for phase 2

# ----------------------------
# Checkpoints
# ----------------------------
DRIVE_PATH = "/content/drive/MyDrive/sketch_project/checkpoints"           # Phase 1 checkpoints
PHASE2_CHECKPOINT_DIR = "/content/drive/MyDrive/sketch_project/checkpoints_phase2"  # Phase 2
os.makedirs(DRIVE_PATH, exist_ok=True)
os.makedirs(PHASE2_CHECKPOINT_DIR, exist_ok=True)

# Resume Phase 2 from Phase 1 generator-only checkpoint
PHASE2_RESUME_PATH = os.path.join(DRIVE_PATH, "generator_stage1_epoch30.pth")

# ----------------------------
# Training hyperparameters
# ----------------------------
IMAGE_SIZE = 1024
BATCH_SIZE = 16
VAL_SPLIT = 0.05

# Phase 1 & Phase 2 can have different epoch counts
# EPOCHS_PHASE1 = 100
EPOCHS = 60
EPOCHS_PHASE2 = 50  # integer! number of epochs

LR = 2e-4
BETAS = (0.5, 0.999)
LAMBDA_L1 = 100.0
LAMBDA_PERC = 1.0
SAVE_EVERY = 5
SAVE_LATEST = True

USE_AMP = True
SEED = 42
RESUME_PATH = None   # Phase 1 resume path if needed

