## train_phase2.py

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast

from config import (
    SKETCH_ROOT, DRIVE_PATH, IMAGE_SIZE, BATCH_SIZE, VAL_SPLIT, EPOCHS_PHASE2,
    LR, BETAS, LAMBDA_L1, LAMBDA_PERC, SAVE_EVERY, RESUME_PHASE2, SAVE_LATEST,
    USE_AMP, SEED
)
from dataset import SketchDataset, resize_with_padding  # Phase 2 dataset class
from model import UNetGenerator, PatchDiscriminator, PerceptualLoss

# ----------------------------
# Safety & Repro
# ----------------------------
torch.autograd.set_detect_anomaly(True)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Helpers
# ----------------------------
def make_loaders():
    ds = SketchDataset(SKETCH_ROOT, image_size=IMAGE_SIZE)
    val_len = max(1, int(len(ds) * VAL_SPLIT))
    train_len = len(ds) - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


def save_ckpt(epoch, G, D, optG, optD, scaler, path):
    payload = {
        "epoch": epoch,
        "G": G.state_dict(),
        "D": D.state_dict(),
        "optG": optG.state_dict(),
        "optD": optD.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "image_size": IMAGE_SIZE,
    }
    torch.save(payload, path)


def maybe_resume(path, G, D, optG, optD, scaler):
    if path is None or not os.path.exists(path):
        return 0
    ckpt = torch.load(path, map_location=device)
    G.load_state_dict(ckpt["G"])
    D.load_state_dict(ckpt["D"])
    if "optG" in ckpt: optG.load_state_dict(ckpt["optG"])
    if "optD" in ckpt: optD.load_state_dict(ckpt["optD"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = int(ckpt.get("epoch", 0))
    print(f"Resumed Phase 2 from {path} at epoch {start_epoch}")
    return start_epoch


# ----------------------------
# Build
# ----------------------------
train_loader, val_loader = make_loaders()

G = UNetGenerator().to(device)  # Load stage-1 weights before Phase 2
if RESUME_PHASE2 and os.path.exists(RESUME_PHASE2):
    G.load_state_dict(torch.load(RESUME_PHASE2, map_location=device))
D = PatchDiscriminator().to(device)

criterion_gan = nn.BCEWithLogitsLoss()
criterion_l1 = nn.L1Loss()
criterion_perc = PerceptualLoss(
    layer_ids=(3, 8, 15, 22),
    layer_weights={3: 1.0, 8: 0.75, 15: 0.5, 22: 0.25},
    perceptual_weight=0.2,
    pixel_weight=1.0
).to(device)

optG = optim.Adam(G.parameters(), lr=LR, betas=BETAS)
optD = optim.Adam(D.parameters(), lr=LR, betas=BETAS)
scaler = GradScaler(enabled=USE_AMP)

start_epoch = maybe_resume(RESUME_PHASE2, G, D, optG, optD, scaler)


# ----------------------------
# Training Loop
# ----------------------------
for epoch in range(start_epoch, EPOCHS_PHASE2):
    G.train(); D.train()
    running_g, running_d = 0.0, 0.0

    for batch in train_loader:
        sketch = batch["sketch"].to(device, non_blocking=True)

        # ------------------
        # Train D
        # ------------------
        optD.zero_grad(set_to_none=True)
        with autocast(enabled=USE_AMP):
            fake = G(sketch)  # In Phase 2, sketch -> refined sketch
            logits_real = D(sketch, sketch)
            logits_fake = D(sketch, fake.detach())
            loss_d_real = criterion_gan(logits_real, torch.ones_like(logits_real))
            loss_d_fake = criterion_gan(logits_fake, torch.zeros_like(logits_fake))
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
        scaler.scale(loss_d).backward()
        scaler.step(optD)

        # ------------------
        # Train G
        # ------------------
        optG.zero_grad(set_to_none=True)
        with autocast(enabled=USE_AMP):
            fake = G(sketch)
            logits_fake_for_g = D(sketch, fake)
            adv_g = criterion_gan(logits_fake_for_g, torch.ones_like(logits_fake_for_g))
            l1 = criterion_l1(fake, sketch) * LAMBDA_L1
            perc = criterion_perc(fake, sketch)
            loss_g = adv_g + l1 + perc
        scaler.scale(loss_g).backward()
        scaler.step(optG)
        scaler.update()

        running_d += loss_d.item()
        running_g += loss_g.item()

    avg_d = running_d / max(1, len(train_loader))
    avg_g = running_g / max(1, len(train_loader))
    print(f"Phase2 Epoch {epoch+1}/{EPOCHS_PHASE2} | D: {avg_d:.4f} | G: {avg_g:.4f}")

    # ------------------
    # Validation (L1)
    # ------------------
    G.eval()
    with torch.no_grad():
        val_l1 = 0.0
        for batch in val_loader:
            sketch = batch["sketch"].to(device)
            fake = G(sketch)
            val_l1 += criterion_l1(fake, sketch).item()
        val_l1 /= max(1, len(val_loader))
    print(f"  Validation L1: {val_l1:.4f}")

    # ------------------
    # Checkpoint
    # ------------------
    if ((epoch + 1) % SAVE_EVERY) == 0 or (epoch + 1) == EPOCHS_PHASE2:
        ep_path = os.path.join(DRIVE_PATH, f"phase2_epoch{epoch+1}.pth")
        save_ckpt(epoch + 1, G, D, optG, optD, scaler, ep_path)
        if SAVE_LATEST:
            latest = os.path.join(DRIVE_PATH, "phase2_latest.pth")
            save_ckpt(epoch + 1, G, D, optG, optD, scaler, latest)
        g_only = os.path.join(DRIVE_PATH, f"generator_phase2_epoch{epoch+1}.pth")
        torch.save(G.state_dict(), g_only)
        print(f"Saved Phase 2 checkpoints to {DRIVE_PATH}")

print("Phase 2 Fine-tuning Complete.")
