
import os

# ----------------------------
# Paths
# ----------------------------
DATA_ROOT = "dataset"  # must contain subfolders: photos/ and sketches/
DRIVE_PATH = "/content/drive/MyDrive/sketch_project/checkpoints"
os.makedirs(DRIVE_PATH, exist_ok=True)

# ----------------------------
# Training hyperparameters
# ----------------------------
IMAGE_SIZE = 512             # drop to 256 on small GPUs
BATCH_SIZE = 4
VAL_SPLIT = 0.2              # random split from single folder
EPOCHS = 100
LR = 1e-4
BETAS = (0.5, 0.999)

# Loss weights
LAMBDA_L1 = 100.0
LAMBDA_PERC = 1.0

# Checkpointing
SAVE_EVERY = 5              # save every N epochs
RESUME_PATH = None           # set to a .pth file to resume, or keep None
SAVE_LATEST = True           # also maintain DRIVE_PATH/"latest.pth"

# Mixed precision (for Colab T4/V100 etc.)
USE_AMP = True

# Reproducibility
SEED = 1337

