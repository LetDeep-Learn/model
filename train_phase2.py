import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast

from config import (
    SKETCH_ROOT, PHASE2_CHECKPOINT_DIR, PHASE2_RESUME_PATH,
    IMAGE_SIZE, BATCH_SIZE, VAL_SPLIT, EPOCHS_PHASE2,
    LR, BETAS, LAMBDA_L1, LAMBDA_PERC, SAVE_EVERY, USE_AMP, SEED
)
from working_scripts.dataset import SketchDataset, resize_with_padding
from helpful_python_scripts.model import UNetGenerator, PerceptualLoss

torch.autograd.set_detect_anomaly(True)

# Reproducibility
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def resize_with_padding(img, target_size=512, pad_color=(0, 0, 0)):
    """
    Resize image while keeping aspect ratio and pad to target_size x target_size.
    Args:
        img (PIL.Image): Input image
        target_size (int): Final desired size (e.g. 512)
        pad_color (tuple or int): padding color (tuple for RGB, int for L)
    Returns:
        PIL.Image: Resized and padded image (target_size x target_size)
    """
    # Scale based on longest side
    ratio = float(target_size) / max(img.size)
    new_size = tuple([int(x * ratio) for x in img.size])
    img = img.resize(new_size, Image.BICUBIC)

    # Pad to target_size x target_size
    delta_w = target_size - new_size[0]
    delta_h = target_size - new_size[1]
    padding = (delta_w // 2, delta_h // 2,
               delta_w - (delta_w // 2), delta_h - (delta_h // 2))

    # Adjust pad_color for image mode
    if img.mode == 'L' and isinstance(pad_color, tuple):
        pad_color = pad_color[0]  # single-channel
    img = ImageOps.expand(img, padding, fill=pad_color)

    return img
# ----------------------------
# Dataset loaders
# ----------------------------
def make_loaders():
    ds = SketchDataset(SKETCH_ROOT, image_size=IMAGE_SIZE)
    val_len = max(1, int(len(ds) * VAL_SPLIT))
    train_len = len(ds) - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader

# ----------------------------
# Checkpoint helpers
# ----------------------------
def save_ckpt(epoch, G, optG, scaler, path):
    payload = {
        "epoch": epoch,
        "G": G.state_dict(),
        "optG": optG.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "image_size": IMAGE_SIZE,
    }
    torch.save(payload, path)

def maybe_resume(path, G, optG, scaler):
    if path is None or not os.path.exists(path):
        print("No checkpoint found, training from scratch.")
        return 0
    ckpt = torch.load(path, map_location=device)
    G.load_state_dict(ckpt) if "G" not in ckpt else G.load_state_dict(ckpt["G"])
    if optG is not None and "optG" in ckpt:
        optG.load_state_dict(ckpt["optG"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = int(ckpt.get("epoch", 0))
    print(f"Resumed generator-only checkpoint from {path} at epoch {start_epoch}")
    return start_epoch

# ----------------------------
# Build model
# ----------------------------
train_loader, val_loader = make_loaders()
G = UNetGenerator().to(device)

criterion_l1 = nn.L1Loss()
criterion_perc = PerceptualLoss(
    layer_ids=(3, 8, 15, 22),
    layer_weights={3:1.0, 8:0.75, 15:0.5, 22:0.25},
    perceptual_weight=0.2,
    pixel_weight=1.0
).to(device)

optG = optim.Adam(G.parameters(), lr=LR, betas=BETAS)
scaler = GradScaler(enabled=USE_AMP)

start_epoch = maybe_resume(PHASE2_RESUME_PATH, G, optG, scaler)

# ----------------------------
# Training loop
# ----------------------------
for epoch in range(start_epoch, EPOCHS_PHASE2):
    G.train()
    running_g = 0.0

    for batch in train_loader:
        sketch = batch["sketch"].to(device, non_blocking=True)

        optG.zero_grad(set_to_none=True)
        with autocast(enabled=USE_AMP):
            fake = G(sketch)  # input = sketch
            l1 = criterion_l1(fake, sketch) * LAMBDA_L1
            perc = criterion_perc(fake, sketch)
            loss_g = l1 + perc
        scaler.scale(loss_g).backward()
        scaler.step(optG)
        scaler.update()

        running_g += loss_g.item()

    avg_g = running_g / max(1, len(train_loader))
    print(f"Epoch {epoch+1}/{EPOCHS_PHASE2} | G: {avg_g:.4f}")

    # Validation
    G.eval()
    with torch.no_grad():
        val_l1 = 0.0
        for batch in val_loader:
            sketch = batch["sketch"].to(device)
            fake = G(sketch)
            val_l1 += criterion_l1(fake, sketch).item()
        val_l1 /= max(1, len(val_loader))
    print(f"  Validation L1: {val_l1:.4f}")

    # Save checkpoints
    # if ((epoch + 1) % SAVE_EVERY) == 0 or (epoch + 1) == EPOCHS_PHASE2:
    #     ckpt_path = os.path.join(PHASE2_CHECKPOINT_DIR, f"epoch{epoch+1}.pth")
    #     save_ckpt(epoch+1, G, optG, scaler, ckpt_path)
    #     print(f"Saved Phase 2 checkpoint at epoch {epoch+1}")
# Save checkpoints
# Save checkpoints
    if ((epoch + 1) % SAVE_EVERY) == 0 or (epoch + 1) == EPOCHS_PHASE2:
        # Full checkpoint for resuming Phase 2 (generator + optimizer + scaler)
        full_ckpt_path = os.path.join(PHASE2_CHECKPOINT_DIR, f"full_epoch{epoch+1}.pth")
        save_ckpt(epoch+1, G, optG, scaler, full_ckpt_path)  # only pass G, optG, scaler

        # Generator-only checkpoint for inference
        g_only_path = os.path.join(PHASE2_CHECKPOINT_DIR, f"generator_epoch{epoch+1}.pth")
        torch.save(G.state_dict(), g_only_path)

        print(f"Saved Phase 2 full and generator-only checkpoints for epoch {epoch+1}")

print("Phase 2 Training Complete.")
