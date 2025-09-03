

# - Adds **EdgeLoss (Sobel)** to sharpen contours
# - Keeps **PerceptualLoss** you already use
# - Adds light **TV loss** to reduce blotchy noise
# - Uses **label smoothing** so D doesn’t become a tyrant
# - Adds **instance noise** to stabilize D (decays over epochs)
# - Adds **MultiStepLR** to drop LR after epochs 20 and 30
# - Uses new **`torch.amp.autocast("cuda")`** API
# - **Gradient clipping** on G to prevent exploding updates
# - Robust handling of sketches with 3 channels (for losses only)

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
from working_scripts.dataset import PairedDataset
from helpful_python_scripts.model import UNetGenerator, PatchDiscriminator, PerceptualLoss

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
criterion_l1  = nn.L1Loss()
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
EDGE_WEIGHT = 5.0        # strong push on edges
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
# Training loop
# ----------------------------
for epoch in range(start_epoch, EPOCHS):
    G.train(); D.train()
    running_g, running_d = 0.0, 0.0

    for batch in train_loader:
        photo  = batch["photo"].to(device, non_blocking=True)
        sketch = batch["sketch"].to(device, non_blocking=True)

        # ------------------
        # Train D
        # ------------------
        optD.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            fake = G(photo).detach()
            # Instance noise to stabilize D
            sketch_noisy = add_instance_noise(sketch, epoch, EPOCHS)
            fake_noisy   = add_instance_noise(fake,   epoch, EPOCHS)

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
            fake = G(photo)
            logits_fake_for_g = D(photo, fake)
            adv_g = criterion_gan(logits_fake_for_g, torch.ones_like(logits_fake_for_g) * REAL_LABEL)

            # L1 + perceptual + edge + light TV
            l1   = criterion_l1(fake, sketch) * LAMBDA_L1
            perc = criterion_perc(fake, sketch)
            edge = criterion_edge(fake, sketch) * EDGE_WEIGHT
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
    print(f"Epoch {epoch+1}/{EPOCHS} | D: {avg_d:.4f} | G: {avg_g:.4f} | LR(G): {optG.param_groups[0]['lr']:.6f}")

    # ------------------
    # Validation (L1 only)
    # ------------------
    G.eval()
    with torch.no_grad():
        val_l1 = 0.0
        for batch in val_loader:
            photo  = batch["photo"].to(device, non_blocking=True)
            sketch = batch["sketch"].to(device, non_blocking=True)
            fake = G(photo)
            val_l1 += criterion_l1(fake, sketch).item()
        val_l1 /= max(1, len(val_loader))
    print(f"  Validation L1: {val_l1:.4f}")

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
        g_only = os.path.join(DRIVE_PATH, f"final_epoch{epoch+1}.pth")
        torch.save(G.state_dict(), g_only)
        print(f"Saved checkpoints to {DRIVE_PATH}")

print("Training complete.")
# ```

# ---

# # 🎛 Optional post-processing: `postprocess.py`

# This polishes any generated sketch so it looks like graphite on paper:
# - Unsharp mask for crisp edges
# - CLAHE for punchy contrast
# - Optional OpenCV pencil-sketch filter
# - Optional paper texture overlay

# > Save as `postprocess.py`. Requires `opencv-python` and a paper texture image if you want the overlay.

# ```python
# import cv2 as cv
# import numpy as np

# def unsharp_mask(gray, ksize=0, sigma=1.5, amount=1.0, thresh=0):
#     if ksize <= 0:
#         ksize = 0  # auto
#     blur = cv.GaussianBlur(gray, (0,0), sigma)
#     sharp = cv.addWeighted(gray, 1 + amount, blur, -amount, 0)
#     if thresh > 0:
#         low_contrast_mask = np.absolute(gray - blur) < thresh
#         np.copyto(sharp, gray, where=low_contrast_mask)
#     return np.clip(sharp, 0, 255).astype(np.uint8)

# def apply_clahe(gray, clip=2.0, tile=(8,8)):
#     clahe = cv.createCLAHE(clipLimit=clip, tileGridSize=tile)
#     return clahe.apply(gray)

# def pencilize(gray, sigma_s=60, sigma_r=0.07, shade_factor=0.05):
#     # OpenCV pencil sketch returns (dst_gray, dst_color)
#     dst_gray, _ = cv.pencilSketch(gray, sigma_s=sigma_s, sigma_r=sigma_r, shade_factor=shade_factor)
#     return dst_gray

# def overlay_paper(gray, paper_path, alpha=0.15):
#     paper = cv.imread(paper_path, cv.IMREAD_GRAYSCALE)
#     if paper is None:
#         return gray
#     paper = cv.resize(paper, (gray.shape[1], gray.shape[0]), interpolation=cv.INTER_AREA)
#     # Multiply-like blend: darker paper texture, preserve highlights
#     blended = cv.addWeighted(gray, 1.0, paper, alpha, 0)
#     return blended

# def process(in_path, out_path, paper_texture=None, use_pencil=False):
#     img = cv.imread(in_path, cv.IMREAD_GRAYSCALE)
#     if img is None:
#         raise FileNotFoundError(in_path)

#     # Step 1: contrast boost
#     img = apply_clahe(img, clip=2.0, tile=(8,8))

#     # Step 2: optional pencil shading
#     if use_pencil:
#         img = pencilize(img, sigma_s=60, sigma_r=0.07, shade_factor=0.03)

#     # Step 3: edge crisping
#     img = unsharp_mask(img, sigma=1.2, amount=1.2)

#     # Step 4: subtle paper texture
#     if paper_texture:
#         img = overlay_paper(img, paper_texture, alpha=0.12)

#     cv.imwrite(out_path, img)

# if __name__ == "__main__":
#     # example
#     import argparse
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--input", required=True)
#     ap.add_argument("--output", required=True)
#     ap.add_argument("--paper", default=None, help="optional path to paper texture")
#     ap.add_argument("--use_pencil", action="store_true")
#     args = ap.parse_args()
#     process(args.input, args.output, paper_texture=args.paper, use_pencil=args.use_pencil)
# ```

# ### How to run post-processing
# ```bash
# python postprocess.py --input generated.png --output sketch_final.png --paper paper_texture.jpg --use_pencil
# ```

# ---

# ## Where this gets you
# - **Sharper edges**: EdgeLoss + unsharp mask
# - **Realistic strokes**: discriminator pressure + optional pencil shading
# - **Higher contrast**: CLAHE + label smoothing preventing washout
# - **Fewer random black blobs**: LR decay, instance noise, TV loss, grad clipping

# If this still produces artifacts, we can ratchet the knobs: bump `EDGE_WEIGHT` a bit, lower `TV_WEIGHT`, or decouple G/D LRs. But this setup should already look way closer to an actual pencil sketch and way less like “sad gray towel.”
