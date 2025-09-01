import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg16
from PIL import Image

# ----------------------------
# Global Drive Path
# ----------------------------
DRIVE_PATH = "/content/drive/MyDrive/sketch_project/phase2_checkpoints"
os.makedirs(DRIVE_PATH, exist_ok=True)

# ----------------------------
# Dataset classes
# ----------------------------
class PairedDataset(Dataset):
    """(photo, sketch) pairs"""
    def __init__(self, root_dir, split="train", image_size=512):
        self.photo_dir = os.path.join(root_dir, split, "photos")
        self.sketch_dir = os.path.join(root_dir, split, "sketches")
        self.files = os.listdir(self.photo_dir)

        self.transform_photo = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])
        self.transform_sketch = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        fname = self.files[idx]
        photo = Image.open(os.path.join(self.photo_dir, fname)).convert("RGB")
        sketch = Image.open(os.path.join(self.sketch_dir, fname)).convert("L")
        return {"photo": self.transform_photo(photo),
                "sketch": self.transform_sketch(sketch)}

class SketchDataset(Dataset):
    """unpaired sketches only"""
    def __init__(self, root_dir, split="train", image_size=512):
        self.sketch_dir = os.path.join(root_dir, split, "sketches")
        self.files = os.listdir(self.sketch_dir)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        sketch = Image.open(os.path.join(self.sketch_dir, self.files[idx])).convert("L")
        return self.transform(sketch)

# ----------------------------
# Generator (reuse UNet from Stage-1)
# ----------------------------
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, act="relu", dropout=False):
        super().__init__()
        if down:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        self.dropout = nn.Dropout(0.5) if dropout else None
    def forward(self, x):
        x = self.conv(x)
        if self.dropout: x = self.dropout(x)
        return x

class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.d1 = UNetBlock(in_ch, 64)
        self.d2 = UNetBlock(64, 128)
        self.d3 = UNetBlock(128, 256)
        self.d4 = UNetBlock(256, 512)
        self.d5 = UNetBlock(512, 512)
        self.d6 = UNetBlock(512, 512)
        self.d7 = UNetBlock(512, 512)
        self.d8 = UNetBlock(512, 512)

        self.u1 = UNetBlock(512, 512, down=False, dropout=True)
        self.u2 = UNetBlock(1024, 512, down=False, dropout=True)
        self.u3 = UNetBlock(1024, 512, down=False, dropout=True)
        self.u4 = UNetBlock(1024, 512, down=False)
        self.u5 = UNetBlock(1024, 256, down=False)
        self.u6 = UNetBlock(512, 128, down=False)
        self.u7 = UNetBlock(256, 64, down=False)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_ch, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(d1); d3 = self.d3(d2); d4 = self.d4(d3)
        d5 = self.d5(d4); d6 = self.d6(d5); d7 = self.d7(d6); d8 = self.d8(d7)

        u1 = self.u1(d8)
        u2 = self.u2(torch.cat([u1, d7], 1))
        u3 = self.u3(torch.cat([u2, d6], 1))
        u4 = self.u4(torch.cat([u3, d5], 1))
        u5 = self.u5(torch.cat([u4, d4], 1))
        u6 = self.u6(torch.cat([u5, d3], 1))
        u7 = self.u7(torch.cat([u6, d2], 1))
        return self.final(torch.cat([u7, d1], 1))

# ----------------------------
# PatchGAN Discriminator
# ----------------------------
class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 4, 1, 1)  # Patch output
        )
    def forward(self, x): return self.model(x)

# ----------------------------
# Perceptual Loss
# ----------------------------
class PerceptualLoss(nn.Module):
    def __init__(self, layer_ids=[3, 8, 15], weight=1.0):
        super().__init__()
        vgg = vgg16(weights="IMAGENET1K_V1").features
        self.layers = nn.ModuleList([vgg[i] for i in layer_ids])
        for p in self.layers: 
            for param in p.parameters():
                param.requires_grad = False
        self.weight = weight
    def forward(self, x, y):
        loss = 0
        for layer in self.layers:
            x = layer(x); y = layer(y)
            loss += nn.functional.l1_loss(x, y)
        return loss * self.weight

# ----------------------------
# Feature Matching Loss
# ----------------------------
def feature_matching_loss(fake_feats, real_feats):
    loss = 0
    for f_fake, f_real in zip(fake_feats, real_feats):
        loss += nn.functional.l1_loss(f_fake, f_real)
    return loss

# ----------------------------
# Training Loop
# ----------------------------
def train(resume_checkpoint=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    paired_ds = PairedDataset("dataset", "train")
    paired_loader = DataLoader(paired_ds, batch_size=4, shuffle=True)
    sketch_ds = SketchDataset("dataset_unpaired", "train")
    sketch_loader = DataLoader(sketch_ds, batch_size=4, shuffle=True)
    sketch_iter = iter(sketch_loader)

    # Models
    G = UNetGenerator().to(device)
    G.load_state_dict(torch.load("generator_stage1.pth", map_location=device))  # start from Stage-1
    D_c = PatchDiscriminator(in_ch=3+1).to(device)
    D_u = PatchDiscriminator(in_ch=1).to(device)

    # Losses
    l1_loss = nn.L1Loss()
    perc_loss = PerceptualLoss().to(device)
    bce_loss = nn.BCEWithLogitsLoss()

    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D_c = torch.optim.Adam(D_c.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D_u = torch.optim.Adam(D_u.parameters(), lr=2e-4, betas=(0.5, 0.999))

    start_epoch = 0
    # Resume
    if resume_checkpoint:
        ckpt = torch.load(resume_checkpoint, map_location=device)
        G.load_state_dict(ckpt["G"])
        D_c.load_state_dict(ckpt["D_c"])
        D_u.load_state_dict(ckpt["D_u"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D_c.load_state_dict(ckpt["opt_D_c"])
        opt_D_u.load_state_dict(ckpt["opt_D_u"])
        start_epoch = ckpt["epoch"]
        print(f"Resumed from {resume_checkpoint}, starting at epoch {start_epoch}")

    # Training
    epochs = 50
    for epoch in range(start_epoch, epochs):
        for batch in paired_loader:
            x = batch["photo"].to(device)
            y = batch["sketch"].to(device)
            fake = G(x)

            # --- D_c ---
            opt_D_c.zero_grad()
            real_in = torch.cat([x, y], 1)
            fake_in = torch.cat([x, fake.detach()], 1)
            d_real = D_c(real_in)
            d_fake = D_c(fake_in)
            loss_Dc = (bce_loss(d_real, torch.ones_like(d_real)) +
                       bce_loss(d_fake, torch.zeros_like(d_fake))) * 0.5
            loss_Dc.backward(); opt_D_c.step()

            # --- D_u ---
            try:
                real_sketch = next(sketch_iter)
            except StopIteration:
                sketch_iter = iter(sketch_loader)
                real_sketch = next(sketch_iter)
            real_sketch = real_sketch.to(device)

            opt_D_u.zero_grad()
            d_real = D_u(real_sketch)
            d_fake = D_u(fake.detach())
            loss_Du = (bce_loss(d_real, torch.ones_like(d_real)) +
                       bce_loss(d_fake, torch.zeros_like(d_fake))) * 0.5
            loss_Du.backward(); opt_D_u.step()

            # --- Generator ---
            opt_G.zero_grad()
            loss_l1 = l1_loss(fake, y) * 100
            loss_perc = perc_loss(fake.repeat(1,3,1,1), y.repeat(1,3,1,1))
            adv_c = bce_loss(D_c(torch.cat([x, fake],1)), torch.ones_like(d_fake))
            adv_u = bce_loss(D_u(fake), torch.ones_like(d_fake))

            # Feature matching
            feats_fake, feats_real = [], []
            f = fake; r = real_sketch
            for layer in D_u.model[:-1]:
                f = layer(f); r = layer(r)
                feats_fake.append(f); feats_real.append(r)
            fm_loss = feature_matching_loss(feats_fake, feats_real) * 10

            loss_G = loss_l1 + loss_perc + adv_c + adv_u + fm_loss
            loss_G.backward(); opt_G.step()

        print(f"Epoch {epoch+1}/{epochs} | G {loss_G.item():.3f} | Dc {loss_Dc.item():.3f} | Du {loss_Du.item():.3f}")

        # Save checkpoint
        if (epoch+1) % 10 == 0:
            ckpt_path = os.path.join(DRIVE_PATH, f"stage2_epoch{epoch+1}.pth")
            torch.save({
                "epoch": epoch+1,
                "G": G.state_dict(),
                "D_c": D_c.state_dict(),
                "D_u": D_u.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D_c": opt_D_c.state_dict(),
                "opt_D_u": opt_D_u.state_dict()
            }, ckpt_path)
            print(f"Checkpoint saved at {ckpt_path}")

if __name__ == "__main__":
    # Example: resume training
    # train(resume_checkpoint=os.path.join(DRIVE_PATH, "stage2_epoch20.pth"))
    train()
