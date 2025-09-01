## `train.py`

import os
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast

from config import (
    DATA_ROOT, DRIVE_PATH, IMAGE_SIZE, BATCH_SIZE, VAL_SPLIT, EPOCHS,
    LR, BETAS, LAMBDA_L1, LAMBDA_PERC, SAVE_EVERY, RESUME_PATH, SAVE_LATEST,
    USE_AMP, SEED
)
from dataset import PairedDataset
from model import UNetGenerator, PatchDiscriminator, PerceptualLoss

# Extra safety to locate autograd issues
torch.autograd.set_detect_anomaly(True)

# Reproducibility
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Helpers
# ----------------------------

def make_loaders():
    ds = PairedDataset(DATA_ROOT, image_size=IMAGE_SIZE)
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
    G.load_state_dict(ckpt["G"])  # strict=True on purpose
    D.load_state_dict(ckpt["D"])  # strict=True to catch arch drift
    optG.load_state_dict(ckpt["optG"]) if "optG" in ckpt else None
    optD.load_state_dict(ckpt["optD"]) if "optD" in ckpt else None
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = int(ckpt.get("epoch", 0))
    print(f"Resumed from {path} at epoch {start_epoch}")
    return start_epoch


# ----------------------------
# Build
# ----------------------------
train_loader, val_loader = make_loaders()
G = UNetGenerator().to(device)
D = PatchDiscriminator().to(device)

criterion_gan = nn.BCEWithLogitsLoss()
criterion_l1 = nn.L1Loss()
criterion_perc = PerceptualLoss(weight=LAMBDA_PERC).to(device)

optG = optim.Adam(G.parameters(), lr=LR, betas=BETAS)
optD = optim.Adam(D.parameters(), lr=LR, betas=BETAS)

scaler = GradScaler(enabled=USE_AMP)

# Resume
start_epoch = 0
resume_source = RESUME_PATH if RESUME_PATH else (os.path.join(DRIVE_PATH, "latest.pth") if SAVE_LATEST else None)
start_epoch = maybe_resume(resume_source, G, D, optG, optD, scaler)

# ----------------------------
# Training
# ----------------------------
for epoch in range(start_epoch, EPOCHS):
    G.train(); D.train()
    running_g, running_d = 0.0, 0.0

    for batch in train_loader:
        photo = batch["photo"].to(device, non_blocking=True)
        sketch = batch["sketch"].to(device, non_blocking=True)

        # ------------------
        # Train D
        # ------------------
        optD.zero_grad(set_to_none=True)
        with autocast(enabled=USE_AMP):
            fake = G(photo)
            logits_real = D(photo, sketch)
            logits_fake = D(photo, fake.detach())
            loss_d_real = criterion_gan(logits_real, torch.ones_like(logits_real))
            loss_d_fake = criterion_gan(logits_fake, torch.zeros_like(logits_fake))
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
        scaler.scale(loss_d).backward()
        scaler.step(optD)
        # don't update scaler between optimizers; scale carries across safely

        # ------------------
        # Train G
        # ------------------
        optG.zero_grad(set_to_none=True)
        with autocast(enabled=USE_AMP):
            fake = G(photo)
            logits_fake_for_g = D(photo, fake)
            adv_g = criterion_gan(logits_fake_for_g, torch.ones_like(logits_fake_for_g))
            l1 = criterion_l1(fake, sketch) * LAMBDA_L1
            perc = criterion_perc(fake, sketch)  # already weighted
            loss_g = adv_g + l1 + perc
        scaler.scale(loss_g).backward()
        scaler.step(optG)
        scaler.update()

        running_d += loss_d.item()
        running_g += loss_g.item()

    avg_d = running_d / max(1, len(train_loader))
    avg_g = running_g / max(1, len(train_loader))
    print(f"Epoch {epoch+1}/{EPOCHS} | D: {avg_d:.4f} | G: {avg_g:.4f}")

    # ------------------
    # Validation (L1 only for fidelity)
    # ------------------
    G.eval()
    with torch.no_grad():
        val_l1 = 0.0
        for batch in val_loader:
            photo = batch["photo"].to(device)
            sketch = batch["sketch"].to(device)
            fake = G(photo)
            val_l1 += criterion_l1(fake, sketch).item()
        val_l1 /= max(1, len(val_loader))
    print(f"  Validation L1: {val_l1:.4f}")

    # ------------------
    # Checkpointing
    # ------------------
    if ((epoch + 1) % SAVE_EVERY) == 0 or (epoch + 1) == EPOCHS:
        ep_path = os.path.join(DRIVE_PATH, f"epoch{epoch+1}.pth")
        save_ckpt(epoch + 1, G, D, optG, optD, scaler, ep_path)
        if SAVE_LATEST:
            latest = os.path.join(DRIVE_PATH, "latest.pth")
            save_ckpt(epoch + 1, G, D, optG, optD, scaler, latest)
        # Also save a Stage-1 generator-only snapshot for Phase 2 fine-tune
        g_only = os.path.join(DRIVE_PATH, f"generator_stage1_epoch{epoch+1}.pth")
        torch.save(G.state_dict(), g_only)
        print(f"Saved checkpoints to {DRIVE_PATH}")

print("Training complete.")
