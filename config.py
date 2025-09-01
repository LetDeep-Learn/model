
# import os

# # ----------------------------
# # Paths
# # ----------------------------
# SKETCH_ROOT="dataset_unpaired/sketches"
# DATA_ROOT = "dataset"  # must contain subfolders: photos/ and sketches/

# DRIVE_PATH = "/content/drive/MyDrive/sketch_project/checkpoints"
# os.makedirs(DRIVE_PATH, exist_ok=True)

# EPOCHS_PHASE2="/content/drive/MyDrive/sketch_project/checkpoints_phase2"
# os.makedirs(EPOCHS_PHASE2, exist_ok=True)


# # PHASE2_RESUME_PATH = "/content/drive/MyDrive/sketch_project/checkpoints/generator_stage1_epoch30.pth"
# RESUME_PHASE2 = os.path.join(
#     DRIVE_PATH, "generator_stage1_epoch30.pth"
# )
# # ----------------------------
# # Training hyperparameters
# # ----------------------------
# IMAGE_SIZE = 512             # drop to 256 on small GPUs
# BATCH_SIZE = 4
# VAL_SPLIT = 0.2              # random split from single folder
# EPOCHS = 100
# LR = 1e-4
# BETAS = (0.5, 0.999)

# # Loss weights
# LAMBDA_L1 = 100.0
# LAMBDA_PERC = 1.0

# # Checkpointing
# SAVE_EVERY = 5              # save every N epochs
# RESUME_PATH = None           # set to a .pth file to resume, or keep None
# SAVE_LATEST = True           # also maintain DRIVE_PATH/"latest.pth"

# # Mixed precision (for Colab T4/V100 etc.)
# USE_AMP = True

# # Reproducibility
# SEED = 1337

import os

# ----------------------------
# Dataset paths
# ----------------------------
DATA_ROOT = "dataset"            # original paired dataset (photos + sketches)
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
IMAGE_SIZE = 512
BATCH_SIZE = 4
VAL_SPLIT = 0.05

# Phase 1 & Phase 2 can have different epoch counts
# EPOCHS_PHASE1 = 100
EPOCHS = 100
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

