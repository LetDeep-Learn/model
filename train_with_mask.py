# train.py (your original file, with mask support added)
# - Adds EdgeLoss (Sobel) to sharpen contours
# - Keeps PerceptualLoss you already use
# - Adds TV loss, label smoothing, instance noise, LR scheduler, AMP, grad clip
# Minimal changes: integrate mask from dataset and apply to L1 / perceptual / edge losses.

import os
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from config import (
    DATA_ROOT, DRIVE_PATH, IMAGE_SIZE, BATCH_SIZE, VAL_SPLIT, EPOCHS,
    LR, BETAS, LAMBDA_L1, LAMBDA_PERC, SAVE_EVERY, RESUME_PATH, SAVE_LATEST,
    USE_AMP, SEED
)
from dataset import PairedDataset
from model import UNetGenerator, PatchDiscriminator, PerceptualLoss

# ----------------------------
# Safety / reproducibility
# ----------------------------
torch.autograd.set_detect_anomaly(False)
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
    G.load_state_dict(ckpt["G"])
    D.load_state_dict(ckpt["D"])
    if "optG" in ckpt and optG is not None:
        optG.load_state_dict(ckpt["optG"])
    if "optD" in ckpt and optD is not None:
        optD.load_state_dict(ckpt["optD"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = int(ckpt.get("epoch", 0))
    print(f"Resumed from {path} at epoch {start_epoch}")
    return start_epoch


# ----------------------------
# Regularizers / custom losses
# ----------------------------
class EdgeLoss(nn.Module):
    """Sobel-based edge magnitude L1 loss. Converts RGB->gray if needed."""
    def __init__(self, eps=1e-6):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]]).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.],
                           [ 0.,  0.,  0.],
                           [ 1.,  2.,  1.]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", kx)
        self.register_buffer("sobel_y", ky)
        self.eps = eps

    def forward(self, pred, target):
        # pred, target: [B, C, H, W] -- convert to grayscale internally
        if target.size(1) == 3:
            target = target.mean(dim=1, keepdim=True)
        if pred.size(1) == 3:
            pred = pred.mean(dim=1, keepdim=True)

        Gx_t = nn.functional.conv2d(target, self.sobel_x, padding=1)
        Gy_t = nn.functional.conv2d(target, self.sobel_y, padding=1)
        Gx_p = nn.functional.conv2d(pred,   self.sobel_x, padding=1)
        Gy_p = nn.functional.conv2d(pred,   self.sobel_y, padding=1)

        mag_t = torch.sqrt(Gx_t**2 + Gy_t**2 + self.eps)
        mag_p = torch.sqrt(Gx_p**2 + Gy_p**2 + self.eps)
        return nn.functional.l1_loss(mag_p, mag_t)


class TVLoss(nn.Module):
    """Total variation loss to discourage blotchy artifacts."""
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    def forward(self, x):
        dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        return self.weight * (dh + dw)


# ----------------------------
# Build
# ----------------------------
train_loader, val_loader = make_loaders()
G = UNetGenerator().to(device)
D = PatchDiscriminator().to(device)

criterion_gan = nn.BCEWithLogitsLoss()
# CHANGE: make L1 reduction='none' so we can mask it
criterion_l1  = nn.L1Loss(reduction="none")
criterion_perc = PerceptualLoss(
    layer_ids=(3, 8, 15, 22),
    layer_weights={3: 1.0, 8: 0.75, 15: 0.5, 22: 0.25},
    perceptual_weight=0.2,
    pixel_weight=1.0
).to(device)
criterion_edge = EdgeLoss().to(device)
criterion_tv   = TVLoss(weight=1.0).to(device)

optG = optim.Adam(G.parameters(), lr=LR, betas=BETAS)
optD = optim.Adam(D.parameters(), lr=LR, betas=BETAS)

# MultiStepLR: 2e-4 -> 1e-4 at epoch 20 -> 5e-5 at epoch 30
G_scheduler = optim.lr_scheduler.MultiStepLR(optG, milestones=[20, 30], gamma=0.5)
D_scheduler = optim.lr_scheduler.MultiStepLR(optD, milestones=[20, 30], gamma=0.5)

scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

# Resume if requested
resume_source = RESUME_PATH if RESUME_PATH else (os.path.join(DRIVE_PATH, "latest.pth") if SAVE_LATEST else None)
start_epoch = maybe_resume(resume_source, G, D, optG, optD, scaler)

# ----------------------------
# Training config
# ----------------------------
EDGE_WEIGHT = 8.0        # strong push on edges
TV_WEIGHT   = 0.1         # light smoothing
REAL_LABEL  = 0.85         # label smoothing for real
FAKE_LABEL  = 0.04

def add_instance_noise(tensor, epoch, total_epochs, base_sigma=0.05):
    """Gaussian noise to D inputs, decays to 0 by end of training."""
    if base_sigma <= 0:
        return tensor
    # cosine decay from base_sigma -> 0
    decay = 0.5 * (1 + math.cos(math.pi * min(epoch, total_epochs) / max(1, total_epochs)))
    sigma = base_sigma * decay
    if sigma <= 0:
        return tensor
    noise = torch.randn_like(tensor) * sigma
    return tensor + noise

# ----------------------------
# Mask helpers (NEW)
# ----------------------------
_eps = 1e-6
def _ensure_same_channels(a, b):
    """Return (a_adj, b_adj) so they have same channel count by repeating single channel if needed."""
    if a.size(1) == b.size(1):
        return a, b
    if a.size(1) == 1 and b.size(1) == 3:
        return a.repeat(1,3,1,1), b
    if a.size(1) == 3 and b.size(1) == 1:
        return a, b.repeat(1,3,1,1)
    return a, b

def masked_l1(pred, target, mask):
    """
    pred, target: tensors. We'll compute L1 in grayscale domain for stability.
    mask: [B,1,H,W] with 1 for real pixels, 0 for padding.
    """
    # compute grayscale representations (1-channel)
    if pred.size(1) == 3:
        pred_gray = pred.mean(dim=1, keepdim=True)
    else:
        pred_gray = pred
    if target.size(1) == 3:
        target_gray = target.mean(dim=1, keepdim=True)
    else:
        target_gray = target

    diff = torch.abs(pred_gray - target_gray) * mask  # [B,1,H,W]
    denom = mask.sum() + _eps
    return diff.sum() / denom

def masked_perceptual(pred, target, mask):
    """
    PerceptualLoss usually expects 3-channel inputs.
    We'll expand grayscale to 3 channels if necessary, and multiply by mask (per-channel mask).
    """
    # make 3-channel tensors
    if pred.size(1) == 1:
        pred_3 = pred.repeat(1,3,1,1)
    else:
        pred_3 = pred
    if target.size(1) == 1:
        target_3 = target.repeat(1,3,1,1)
    else:
        target_3 = target

    mask3 = mask.repeat(1,3,1,1)  # [B,3,H,W]
    # apply mask to images so VGG features don't get polluted by padded areas
    pred_masked = pred_3 * mask3
    target_masked = target_3 * mask3
    # note: criterion_perc should operate on image tensors; it's pre-existing
    return criterion_perc(pred_masked, target_masked)

def masked_edge(pred, target, mask):
    """
    EdgeLoss converts to grayscale internally; apply mask first.
    """
    # apply mask to all channels (if pred 3-ch, mask will broadcast)
    pred_masked = pred * mask
    target_masked = target * mask
    return criterion_edge(pred_masked, target_masked)


# # ----------------------------
# # Training loop
# # ----------------------------
# for epoch in range(start_epoch, EPOCHS):
#     G.train(); D.train()
#     running_g, running_d = 0.0, 0.0

#     for batch in train_loader:
#         photo  = batch["photo"].to(device, non_blocking=True)   # [B,3,H,W]
#         sketch = batch["sketch"].to(device, non_blocking=True)  # [B,1,H,W] or [B,3,H,W] depending on dataset
#         mask   = batch.get("mask", None)
#         if mask is None:
#             # fallback: if dataset didn't provide mask, assume all valid
#             mask = torch.ones((photo.size(0), 1, IMAGE_SIZE, IMAGE_SIZE), device=device)
#         else:
#             mask = mask.to(device, non_blocking=True)           # [B,1,H,W]

#         # ------------------
#         # Train D
#         # ------------------
#         optD.zero_grad(set_to_none=True)
#         with torch.amp.autocast("cuda", enabled=USE_AMP):
#             fake = G(photo).detach()  # [B, Cf, H, W]

#             # prepare fake/sketch for Discriminator: D expects same channel shape for 'sketch' input as during training.
#             # If sketch is 1-channel and fake is 3-ch, convert fake to 1-ch for D; if sketch is 3-ch, ensure fake is 3-ch.
#             if sketch.size(1) == 1 and fake.size(1) == 3:
#                 fake_for_d = fake.mean(dim=1, keepdim=True)
#             elif sketch.size(1) == 3 and fake.size(1) == 1:
#                 fake_for_d = fake.repeat(1,3,1,1)
#             else:
#                 fake_for_d = fake

#             # Instance noise to stabilize D (applied on the sketch/fake branch only as before)
#             sketch_noisy = add_instance_noise(sketch, epoch, EPOCHS)
#             fake_noisy   = add_instance_noise(fake_for_d, epoch, EPOCHS)

#             logits_real = D(photo, sketch_noisy)
#             logits_fake = D(photo, fake_noisy)

#             labels_real = torch.full_like(logits_real, REAL_LABEL)
#             labels_fake = torch.full_like(logits_fake, FAKE_LABEL)

#             loss_d_real = criterion_gan(logits_real, labels_real)
#             loss_d_fake = criterion_gan(logits_fake, labels_fake)
#             loss_d = 0.5 * (loss_d_real + loss_d_fake)

#         scaler.scale(loss_d).backward()
#         scaler.step(optD)

#         # ------------------
#         # Train G
#         # ------------------
#         optG.zero_grad(set_to_none=True)
#         with torch.amp.autocast("cuda", enabled=USE_AMP):
#             fake = G(photo)  # fresh fake for generator update

#             # prepare fake_for_discriminator similarly
#             if sketch.size(1) == 1 and fake.size(1) == 3:
#                 fake_for_d = fake.mean(dim=1, keepdim=True)
#             elif sketch.size(1) == 3 and fake.size(1) == 1:
#                 fake_for_d = fake.repeat(1,3,1,1)
#             else:
#                 fake_for_d = fake

#             logits_fake_for_g = D(photo, fake_for_d)
#             adv_g = criterion_gan(logits_fake_for_g, torch.ones_like(logits_fake_for_g) * REAL_LABEL)

#             # --- Masked losses ---
#             # L1: compute in grayscale domain and mask padding
#             l1 = masked_l1(fake, sketch, mask) * LAMBDA_L1

#             # Perceptual: operate on 3-channel masked inputs
#             perc = masked_perceptual(fake, sketch, mask) * LAMBDA_PERC

#             # Edge: mask applied inside helper
#             edge = masked_edge(fake, sketch, mask) * EDGE_WEIGHT

#             # TV on raw fake (no mask) — small smoothing
#             tv   = criterion_tv(fake) * TV_WEIGHT

#             loss_g = adv_g + l1 + perc + edge + tv

#         scaler.scale(loss_g).backward()
#         # gradient clipping to keep G sane
#         scaler.unscale_(optG)
#         nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
#         scaler.step(optG)
#         scaler.update()

#         running_d += float(loss_d.detach().item())
#         running_g += float(loss_g.detach().item())

#     avg_d = running_d / max(1, len(train_loader))
#     avg_g = running_g / max(1, len(train_loader))
#     print(f"Epoch {epoch+1}/{EPOCHS} | D: {avg_d:.4f} | G: {avg_g:.4f} | LR(G): {optG.param_groups[0]['lr']:.6f}")

#     # ------------------
#     # Validation (masked L1 only)
#     # ------------------
#     G.eval()
#     with torch.no_grad():
#         val_l1 = 0.0
#         for batch in val_loader:
#             photo  = batch["photo"].to(device, non_blocking=True)
#             sketch = batch["sketch"].to(device, non_blocking=True)
#             mask   = batch.get("mask", None)
#             if mask is None:
#                 mask = torch.ones((photo.size(0), 1, IMAGE_SIZE, IMAGE_SIZE), device=device)
#             else:
#                 mask = mask.to(device, non_blocking=True)

#             fake = G(photo)
#             val_l1 += masked_l1(fake, sketch, mask).item()
#         val_l1 /= max(1, len(val_loader))
#     print(f"  Validation L1: {val_l1:.4f}")

#     # Step schedulers AFTER validation per epoch
#     G_scheduler.step()
#     D_scheduler.step()

#     # ------------------
#     # Checkpointing
#     # ------------------
#     if ((epoch + 1) % SAVE_EVERY) == 0 or (epoch + 1) == EPOCHS:
#         ep_path = os.path.join(DRIVE_PATH, f"epoch{epoch+1}.pth")
#         save_ckpt(epoch + 1, G, D, optG, optD, scaler, ep_path)
#         if SAVE_LATEST:
#             latest = os.path.join(DRIVE_PATH, "latest.pth")
#             save_ckpt(epoch + 1, G, D, optG, optD, scaler, latest)
#         # generator-only snapshot for inference
#         g_only = os.path.join(DRIVE_PATH, f"mask_epoch{epoch+1}.pth")
#         torch.save(G.state_dict(), g_only)
#         print(f"Saved checkpoints to {DRIVE_PATH}")

# print("Training complete.")

# ----------------------------
# Training loop (only prints changed)
# ----------------------------
for epoch in range(start_epoch, EPOCHS):
    G.train(); D.train()
    running_g, running_d = 0.0, 0.0

    for batch in train_loader:
        photo  = batch["photo"].to(device, non_blocking=True)   # [B,3,H,W]
        sketch = batch["sketch"].to(device, non_blocking=True)  # [B,1,H,W] or [B,3,H,W] depending on dataset
        mask   = batch.get("mask", None)
        if mask is None:
            # fallback: if dataset didn't provide mask, assume all valid
            mask = torch.ones((photo.size(0), 1, IMAGE_SIZE, IMAGE_SIZE), device=device)
        else:
            mask = mask.to(device, non_blocking=True)           # [B,1,H,W]

        # ------------------
        # Train D
        # ------------------
        optD.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            fake = G(photo).detach()  # [B, Cf, H, W]

            # prepare fake/sketch for Discriminator: D expects same channel shape for 'sketch' input as during training.
            # If sketch is 1-channel and fake is 3-ch, convert fake to 1-ch for D; if sketch is 3-ch, ensure fake is 3-ch.
            if sketch.size(1) == 1 and fake.size(1) == 3:
                fake_for_d = fake.mean(dim=1, keepdim=True)
            elif sketch.size(1) == 3 and fake.size(1) == 1:
                fake_for_d = fake.repeat(1,3,1,1)
            else:
                fake_for_d = fake

            # Instance noise to stabilize D (applied on the sketch/fake branch only as before)
            sketch_noisy = add_instance_noise(sketch, epoch, EPOCHS)
            fake_noisy   = add_instance_noise(fake_for_d, epoch, EPOCHS)

            logits_real = D(photo, sketch_noisy)
            logits_fake = D(photo, fake_noisy)

            labels_real = torch.full_like(logits_real, REAL_LABEL)
            labels_fake = torch.full_like(logits_fake, FAKE_LABEL)

            loss_d_real = criterion_gan(logits_real, labels_real)
            loss_d_fake = criterion_gan(logits_fake, labels_fake)
            loss_d = 0.5 * (loss_d_real + loss_d_fake)

        scaler.scale(loss_d).backward()
        scaler.step(optD)

        # ------------------
        # Train G
        # ------------------
        optG.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            fake = G(photo)  # fresh fake for generator update

            # prepare fake_for_discriminator similarly
            if sketch.size(1) == 1 and fake.size(1) == 3:
                fake_for_d = fake.mean(dim=1, keepdim=True)
            elif sketch.size(1) == 3 and fake.size(1) == 1:
                fake_for_d = fake.repeat(1,3,1,1)
            else:
                fake_for_d = fake

            logits_fake_for_g = D(photo, fake_for_d)
            adv_g = criterion_gan(logits_fake_for_g, torch.ones_like(logits_fake_for_g) * REAL_LABEL)

            # --- Masked losses ---
            # L1: compute in grayscale domain and mask padding
            l1 = masked_l1(fake, sketch, mask) * LAMBDA_L1

            # Perceptual: operate on 3-channel masked inputs
            perc = masked_perceptual(fake, sketch, mask) * LAMBDA_PERC

            # Edge: mask applied inside helper
            edge = masked_edge(fake, sketch, mask) * EDGE_WEIGHT

            # TV on raw fake (no mask) — small smoothing
            tv   = criterion_tv(fake) * TV_WEIGHT

            loss_g = adv_g + l1 + perc + edge + tv

        scaler.scale(loss_g).backward()
        # gradient clipping to keep G sane
        scaler.unscale_(optG)
        nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
        scaler.step(optG)
        scaler.update()

        running_d += float(loss_d.detach().item())
        running_g += float(loss_g.detach().item())

    avg_d = running_d / max(1, len(train_loader))
    avg_g = running_g / max(1, len(train_loader))

    # ------------------ pretty epoch summary print ------------------
    # small dataset info (best-effort) and decorative box
    dataset_count = len(train_loader.dataset) if hasattr(train_loader, "dataset") else "N/A"
    print("\n╔════════════════════════════════════════════════════════════════════════════╗")
    print(f"║  Epoch {epoch+1:3d}/{EPOCHS:3d}  ·  Samples: {dataset_count}  ·  Batch: {BATCH_SIZE:2d} ", end="")
    print(f"· 🔁 LR(G): {optG.param_groups[0]['lr']:.6f} ║")
    print("╟────────────────────────────────────────────────────────────────────────────╢")
    print(f"║ 🛡️ Discriminator loss (D) : {avg_d:7.4f}     |   ⚙️  Generator loss (G) : {avg_g:7.4f}   ║")
    # tiny health-check nudges
    d_health = "OK" if 0.35 <= avg_d <= 0.75 else ("LOW" if avg_d < 0.35 else "HIGH")
    g_note = "improving" if avg_g < 5.0 else "training"
    print(f"║ ⚑ Status: D={d_health}  ·  G={g_note}  ·  (lower L1/val is better)              ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")

    # ------------------
    # Validation (masked L1 only)
    # ------------------
    G.eval()
    with torch.no_grad():
        val_l1 = 0.0
        for batch in val_loader:
            photo  = batch["photo"].to(device, non_blocking=True)
            sketch = batch["sketch"].to(device, non_blocking=True)
            mask   = batch.get("mask", None)
            if mask is None:
                mask = torch.ones((photo.size(0), 1, IMAGE_SIZE, IMAGE_SIZE), device=device)
            else:
                mask = mask.to(device, non_blocking=True)

            fake = G(photo)
            val_l1 += masked_l1(fake, sketch, mask).item()
        val_l1 /= max(1, len(val_loader))

    # decorated validation line
    val_flag = "✅" if val_l1 < 0.15 else ("⚠️" if val_l1 < 0.30 else "❗")
    print(f"  {val_flag}  Validation L1 (masked): {val_l1:.4f}   ({'good' if val_l1 < 0.15 else 'needs work'})")

    # Step schedulers AFTER validation per epoch
    G_scheduler.step()
    D_scheduler.step()

    # ------------------
    # Checkpointing
    # ------------------
    if ((epoch + 1) % SAVE_EVERY) == 0 or (epoch + 1) == EPOCHS:
        ep_path = os.path.join(DRIVE_PATH, f"epoch{epoch+1}.pth")
        save_ckpt(epoch + 1, G, D, optG, optD, scaler, ep_path)
        if SAVE_LATEST:
            latest = os.path.join(DRIVE_PATH, "latest.pth")
            save_ckpt(epoch + 1, G, D, optG, optD, scaler, latest)
        # generator-only snapshot for inference
        g_only = os.path.join(DRIVE_PATH, f"mask_epoch{epoch+1}.pth")
        torch.save(G.state_dict(), g_only)
        print(f"💾  Saved checkpoints to {DRIVE_PATH}  (epoch {epoch+1})")

print("🏁  Training complete. Your model survived another run.")
