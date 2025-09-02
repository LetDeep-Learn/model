import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg16
from PIL import Image
torch.autograd.set_detect_anomaly(True)
# ----------------------------
# Global Drive Path
# ----------------------------
DRIVE_PATH = "/content/drive/MyDrive/sketch_project/checkpoints"
os.makedirs(DRIVE_PATH, exist_ok=True)

# ----------------------------
# Dataset
# ----------------------------
class PairedDataset(Dataset):
    def __init__(self, root_dir,image_size=512):
        self.photo_dir = os.path.join(root_dir,"photos")
        self.sketch_dir = os.path.join(root_dir,"sketches")
        self.files = os.listdir(self.photo_dir)

        self.transform_photo = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])  # RGB → [-1,1]
        ])

        self.transform_sketch = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # grayscale → [-1,1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        photo = Image.open(os.path.join(self.photo_dir, fname)).convert("RGB")
        sketch = Image.open(os.path.join(self.sketch_dir, fname)).convert("L")

        return {
            "photo": self.transform_photo(photo),
            "sketch": self.transform_sketch(sketch)
        }

# ----------------------------
# UNet Generator
# ----------------------------
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, dropout=False):
        super().__init__()
        if down:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=False)
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=False)
            )
        self.dropout = nn.Dropout(0.5) if dropout else None

    def forward(self, x):
        x = self.conv(x)
        if self.dropout:
            x = self.dropout(x)
        return x

class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        # encoder
        self.d1 = UNetBlock(in_ch, 64)       # 256 -> 128
        self.d2 = UNetBlock(64, 128)         # 128 -> 64
        self.d3 = UNetBlock(128, 256)        # 64 -> 32
        self.d4 = UNetBlock(256, 512)        # 32 -> 16
        self.d5 = UNetBlock(512, 512)        # 16 -> 8
        self.d6 = UNetBlock(512, 512)        # 8 -> 4
        self.d7 = UNetBlock(512, 512)        # 4 -> 2
        self.d8 = UNetBlock(512, 512)        # 2 -> 1

        # decoder
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
        out = self.final(torch.cat([u7, d1], 1))
        return out

# ----------------------------
# Perceptual Loss (VGG16 features)
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
# Training Loop
# ----------------------------
def train(resume_checkpoint=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    train_ds = PairedDataset("dataset")
    val_ds = PairedDataset("dataset")
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)

    # Model
    G = UNetGenerator().to(device)
    l1_loss = nn.L1Loss()
    perc_loss = PerceptualLoss().to(device)

    opt = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))

    start_epoch = 0
    # Resume if checkpoint provided
    if resume_checkpoint:
        ckpt = torch.load(resume_checkpoint, map_location=device)
        G.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"]
        print(f"Resumed from {resume_checkpoint}, starting at epoch {start_epoch}")

    # Train
    epochs = 100
    for epoch in range(start_epoch, epochs):
        G.train()
        total_loss = 0
        for batch in train_loader:
            x = batch["photo"].to(device)
            y = batch["sketch"].to(device)

            y_pred = G(x)

            loss_l1 = l1_loss(y_pred, y) * 100
            loss_perc = perc_loss(y_pred.repeat(1,3,1,1), y.repeat(1,3,1,1))
            loss = loss_l1 + loss_perc

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_loss/len(train_loader):.4f}")

        # Save checkpoint every 10 epochs
        if (epoch+1) % 10 == 0:
            ckpt_path = os.path.join(DRIVE_PATH, f"generator_epoch{epoch+1}.pth")
            torch.save({
                "epoch": epoch+1,
                "model_state": G.state_dict(),
                "optimizer_state": opt.state_dict()
            }, ckpt_path)
            print(f"Checkpoint saved at {ckpt_path}")

if __name__ == "__main__":
    # Example usage: resume from epoch 20
    # train(resume_checkpoint=os.path.join(DRIVE_PATH, "generator_epoch20.pth"))
    train()
