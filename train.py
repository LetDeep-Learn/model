# train.py (updated)
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
from working_scripts.dataset import PairedDataset ,resize_with_padding
from helpful_python_scripts.model import UNetGenerator, PatchDiscriminator, PerceptualLoss

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
criterion_perc = PerceptualLoss(
    layer_ids=(3, 8, 15, 22),
    layer_weights={3: 1.0, 8: 0.75, 15: 0.5, 22: 0.25},
    perceptual_weight=LAMBDA_PERC,
    pixel_weight=1.0
).to(device)

optG = optim.Adam(G.parameters(), lr=LR, betas=BETAS)
optD = optim.Adam(D.parameters(), lr=LR, betas=BETAS)

# LR schedulers to calm things down mid-training
g_scheduler = torch.optim.lr_scheduler.StepLR(optG, step_size=30, gamma=0.5)
d_scheduler = torch.optim.lr_scheduler.StepLR(optD, step_size=30, gamma=0.5)

scaler = GradScaler(enabled=USE_AMP)

# Resume
start_epoch = 0
resume_source = RESUME_PATH if RESUME_PATH else (os.path.join(DRIVE_PATH, "latest.pth") if SAVE_LATEST else None)
start_epoch = maybe_resume(resume_source, G, D, optG, optD, scaler)

# ----------------------------
# Training utilities / hyperparams
# ----------------------------
label_smooth_real_min = 0.8
label_smooth_real_max = 1.0
label_smooth_fake_min = 0.0
label_smooth_fake_max = 0.2
label_flip_prob = 0.03  # occasionally flip labels for robustness
d_input_noise_std = 0.03  # small noise added to D inputs
grad_clip_norm = 5.0  # optional gradient clipping

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
        # Forward / Range alignment safety
        # ------------------
        # We try to automatically align target sketch tensor to fake's output range to avoid
        # the classic mismatch: network outputs in [-1,1] but targets are [0,1] (or vice versa).
        # We'll compute fake first (no grads) and adjust sketch for loss computations accordingly.
        with torch.no_grad():
            provisional_fake = G(photo)

        # Detect ranges
        fake_min, fake_max = float(provisional_fake.min()), float(provisional_fake.max())
        sketch_min, sketch_max = float(sketch.min()), float(sketch.max())

        # If fake is in [-1,1] and sketch in [0,1] -> map sketch to [-1,1]
        if fake_min >= -1.01 and fake_max <= 1.01 and sketch_min >= -0.01 and sketch_max <= 1.01:
            sketch_for_loss = sketch * 2.0 - 1.0
        # If fake is in [0,1] and sketch in [-1,1] -> map sketch to [0,1]
        elif fake_min >= -0.01 and fake_max <= 1.01 and sketch_min >= -1.01 and sketch_max <= 1.01:
            sketch_for_loss = (sketch + 1.0) / 2.0
        else:
            # Unknown ranges (either both match or both weird). Use sketch as-is.
            sketch_for_loss = sketch

        # ------------------
        # Train D
        # ------------------
        optD.zero_grad(set_to_none=True)
        with autocast(enabled=USE_AMP):
            fake = G(photo)
            # Add tiny gaussian noise to real and fake inputs before D (stabilizes)
            real_input = sketch_for_loss + d_input_noise_std * torch.randn_like(sketch_for_loss)
            fake_input = fake.detach() + d_input_noise_std * torch.randn_like(fake.detach())

            logits_real = D(photo, real_input)
            logits_fake = D(photo, fake_input)

            # Label smoothing
            real_label = torch.empty_like(logits_real).uniform_(label_smooth_real_min, label_smooth_real_max).to(device)
            fake_label = torch.empty_like(logits_fake).uniform_(label_smooth_fake_min, label_smooth_fake_max).to(device)

            # occasional label flip to improve robustness
            if random.random() < label_flip_prob:
                real_label, fake_label = fake_label, real_label

            loss_d_real = criterion_gan(logits_real, real_label)
            loss_d_fake = criterion_gan(logits_fake, fake_label)
            loss_d = 0.5 * (loss_d_real + loss_d_fake)

        scaler.scale(loss_d).backward()
        # optional gradient clipping
        if grad_clip_norm is not None:
            scaler.unscale_(optD)
            torch.nn.utils.clip_grad_norm_(D.parameters(), grad_clip_norm)
        scaler.step(optD)
        # don't call scaler.update() yet; it will be called after G step

        # ------------------
        # Train G
        # ------------------
        optG.zero_grad(set_to_none=True)
        with autocast(enabled=USE_AMP):
            fake = G(photo)
            # For adversarial objective, we want D(fake) -> real (smoothed 1s)
            logits_fake_for_g = D(photo, fake)
            # Use smoothed "real" targets for generator loss as well
            gen_real_targets = torch.empty_like(logits_fake_for_g).uniform_(label_smooth_real_min, label_smooth_real_max).to(device)

            adv_g = criterion_gan(logits_fake_for_g, gen_real_targets)
            # Use the sketch_for_loss computed earlier so the pixel/perceptual losses compare correctly
            l1 = criterion_l1(fake, sketch_for_loss) * LAMBDA_L1
            perc = criterion_perc(fake, sketch_for_loss)  # PerceptualLoss handles internal weighting
            loss_g = adv_g + l1 + perc

        scaler.scale(loss_g).backward()
        # optional gradient clipping
        if grad_clip_norm is not None:
            scaler.unscale_(optG)
            torch.nn.utils.clip_grad_norm_(G.parameters(), grad_clip_norm)
        scaler.step(optG)
        scaler.update()

        running_d += loss_d.item()
        running_g += loss_g.item()

    # Step schedulers each epoch to reduce LR slowly
    g_scheduler.step()
    d_scheduler.step()

    avg_d = running_d / max(1, len(train_loader))
    avg_g = running_g / max(1, len(train_loader))
    print(f"Epoch {epoch+1}/{EPOCHS} | D: {avg_d:.4f} | G: {avg_g:.4f} | LR_G: {g_scheduler.get_last_lr()[0]:.2e}")

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

            # Align validation sketch to fake's range just like in training
            fake_min, fake_max = float(fake.min()), float(fake.max())
            sketch_min, sketch_max = float(sketch.min()), float(sketch.max())
            if fake_min >= -1.01 and fake_max <= 1.01 and sketch_min >= -0.01 and sketch_max <= 1.01:
                sketch_for_val = sketch * 2.0 - 1.0
            elif fake_min >= -0.01 and fake_max <= 1.01 and sketch_min >= -1.01 and sketch_max <= 1.01:
                sketch_for_val = (sketch + 1.0) / 2.0
            else:
                sketch_for_val = sketch

            val_l1 += criterion_l1(fake, sketch_for_val).item()
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
        g_only = os.path.join(DRIVE_PATH, f"phase_one_epoch{epoch+1}.pth")
        torch.save(G.state_dict(), g_only)
        print(f"Saved checkpoints to {DRIVE_PATH}")

print("Training complete.")




# ## `train.py` OLD

# import os
# import math
# import random
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
# from torch.cuda.amp import GradScaler, autocast

# from config import (
#     DATA_ROOT, DRIVE_PATH, IMAGE_SIZE, BATCH_SIZE, VAL_SPLIT, EPOCHS,
#     LR, BETAS, LAMBDA_L1, LAMBDA_PERC, SAVE_EVERY, RESUME_PATH, SAVE_LATEST,
#     USE_AMP, SEED
# )
# from dataset import PairedDataset ,resize_with_padding
# from model import UNetGenerator, PatchDiscriminator, PerceptualLoss

# # Extra safety to locate autograd issues
# torch.autograd.set_detect_anomaly(True)

# # Reproducibility
# random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ----------------------------
# # Helpers
# # ----------------------------

# def make_loaders():
#     ds = PairedDataset(DATA_ROOT, image_size=IMAGE_SIZE)
#     val_len = max(1, int(len(ds) * VAL_SPLIT))
#     train_len = len(ds) - val_len
#     train_ds, val_ds = random_split(ds, [train_len, val_len])
#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
#     val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
#     return train_loader, val_loader


# def save_ckpt(epoch, G, D, optG, optD, scaler, path):
#     payload = {
#         "epoch": epoch,
#         "G": G.state_dict(),
#         "D": D.state_dict(),
#         "optG": optG.state_dict(),
#         "optD": optD.state_dict(),
#         "scaler": scaler.state_dict() if scaler is not None else None,
#         "image_size": IMAGE_SIZE,
#     }
#     torch.save(payload, path)


# def maybe_resume(path, G, D, optG, optD, scaler):
#     if path is None or not os.path.exists(path):
#         return 0
#     ckpt = torch.load(path, map_location=device)
#     G.load_state_dict(ckpt["G"])  # strict=True on purpose
#     D.load_state_dict(ckpt["D"])  # strict=True to catch arch drift
#     optG.load_state_dict(ckpt["optG"]) if "optG" in ckpt else None
#     optD.load_state_dict(ckpt["optD"]) if "optD" in ckpt else None
#     if scaler is not None and ckpt.get("scaler") is not None:
#         scaler.load_state_dict(ckpt["scaler"])
#     start_epoch = int(ckpt.get("epoch", 0))
#     print(f"Resumed from {path} at epoch {start_epoch}")
#     return start_epoch


# # ----------------------------
# # Build
# # ----------------------------
# train_loader, val_loader = make_loaders()
# G = UNetGenerator().to(device)
# D = PatchDiscriminator().to(device)

# criterion_gan = nn.BCEWithLogitsLoss()
# criterion_l1 = nn.L1Loss()
# # old     criterion_perc = PerceptualLoss(weight=LAMBDA_PERC).to(device)
# criterion_perc = PerceptualLoss(
#     layer_ids=(3, 8, 15, 22),
#     layer_weights={3: 1.0, 8: 0.75, 15: 0.5, 22: 0.25},
#     perceptual_weight=0.2,
#     pixel_weight=1.0
# ).to(device)
# optG = optim.Adam(G.parameters(), lr=LR, betas=BETAS)
# optD = optim.Adam(D.parameters(), lr=LR, betas=BETAS)

# scaler = GradScaler(enabled=USE_AMP)

# # Resume
# start_epoch = 0
# resume_source = RESUME_PATH if RESUME_PATH else (os.path.join(DRIVE_PATH, "latest.pth") if SAVE_LATEST else None)
# start_epoch = maybe_resume(resume_source, G, D, optG, optD, scaler)

# # ----------------------------
# # Training
# # ----------------------------
# for epoch in range(start_epoch, EPOCHS):
#     G.train(); D.train()
#     running_g, running_d = 0.0, 0.0

#     for batch in train_loader:
#         photo = batch["photo"].to(device, non_blocking=True)
#         sketch = batch["sketch"].to(device, non_blocking=True)

#         # ------------------
#         # Train D
#         # ------------------
#         optD.zero_grad(set_to_none=True)
#         with autocast(enabled=USE_AMP):
#             fake = G(photo)
#             logits_real = D(photo, sketch)
#             logits_fake = D(photo, fake.detach())
#             loss_d_real = criterion_gan(logits_real, torch.ones_like(logits_real))
#             loss_d_fake = criterion_gan(logits_fake, torch.zeros_like(logits_fake))
#             loss_d = 0.5 * (loss_d_real + loss_d_fake)
#         scaler.scale(loss_d).backward()
#         scaler.step(optD)
#         # don't update scaler between optimizers; scale carries across safely

#         # ------------------
#         # Train G
#         # ------------------
#         optG.zero_grad(set_to_none=True)
#         with autocast(enabled=USE_AMP):
#             fake = G(photo)
#             logits_fake_for_g = D(photo, fake)
#             adv_g = criterion_gan(logits_fake_for_g, torch.ones_like(logits_fake_for_g))
#             l1 = criterion_l1(fake, sketch) * LAMBDA_L1
#             perc = criterion_perc(fake, sketch)  # already weighted
#             loss_g = adv_g + l1 + perc
#         scaler.scale(loss_g).backward()
#         scaler.step(optG)
#         scaler.update()

#         running_d += loss_d.item()
#         running_g += loss_g.item()

#     avg_d = running_d / max(1, len(train_loader))
#     avg_g = running_g / max(1, len(train_loader))
#     print(f"Epoch {epoch+1}/{EPOCHS} | D: {avg_d:.4f} | G: {avg_g:.4f}")

#     # ------------------
#     # Validation (L1 only for fidelity)
#     # ------------------
#     G.eval()
#     with torch.no_grad():
#         val_l1 = 0.0
#         for batch in val_loader:
#             photo = batch["photo"].to(device)
#             sketch = batch["sketch"].to(device)
#             fake = G(photo)
#             val_l1 += criterion_l1(fake, sketch).item()
#         val_l1 /= max(1, len(val_loader))
#     print(f"  Validation L1: {val_l1:.4f}")

#     # ------------------
#     # Checkpointing
#     # ------------------
#     if ((epoch + 1) % SAVE_EVERY) == 0 or (epoch + 1) == EPOCHS:
#         ep_path = os.path.join(DRIVE_PATH, f"epoch{epoch+1}.pth")
#         save_ckpt(epoch + 1, G, D, optG, optD, scaler, ep_path)
#         if SAVE_LATEST:
#             latest = os.path.join(DRIVE_PATH, "latest.pth")
#             save_ckpt(epoch + 1, G, D, optG, optD, scaler, latest)
#         # Also save a Stage-1 generator-only snapshot for Phase 2 fine-tune
#         g_only = os.path.join(DRIVE_PATH, f"phase_one_epoch{epoch+1}.pth")
#         torch.save(G.state_dict(), g_only)
#         print(f"Saved checkpoints to {DRIVE_PATH}")

# print("Training complete.")
