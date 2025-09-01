# train_im2pencil.py
"""
Minimal image->pencil training script (Pix2Pix-style) with procedural pair synthesis.
One-file quickstart: set paths in CONFIG and run.

Requirements:
    pip install torch torchvision opencv-python Pillow tqdm

Usage:
    python train_im2pencil.py
"""

import os
import math
import random
from glob import glob
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, utils

# ---------------------------
# CONFIG (edit these)
# ---------------------------
CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'epochs': 40,
    'batch_size': 8,
    'lr': 2e-4,
    'img_size': 256,
    'data_photos_dir': 'dataset/photos',        # put training photos here (optional)
    'data_drawings_dir': 'dataset/sketches',               # optional: pencil drawings (if you have them)
    'output_dir': '/content/drive/MyDrive/model/checkpoints',
    'save_every': 1000,                      # steps
    'val_every': 1000,
    'seed': 42,
    'use_procedural_pairs': True,            # True -> make pairs procedurally from photos
    'perceptual_weight': 1.0,
    'l1_weight': 100.0,
    'adv_weight': 1.0,
}
os.makedirs(CONFIG['output_dir'], exist_ok=True)
random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])

# ---------------------------
# Utilities: XDoG + guided filter
# ---------------------------
def to_gray_float(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
    return (g.astype(np.float32) / 255.0)

def xdog(img_gray, sigma=1.8, k=1.6, tau=0.99, eps=0.08, phi=200.0):
    I = img_gray.astype(np.float32)
    s1 = max(3, int(2*(np.ceil(3*sigma))+1))
    g1 = cv2.GaussianBlur(I, (s1,s1), sigma)
    sigma2 = sigma * k
    s2 = max(3, int(2*(np.ceil(3*sigma2))+1))
    g2 = cv2.GaussianBlur(I, (s2,s2), sigma2)
    D = g1 - tau * g2
    X = np.ones_like(D, dtype=np.float32)
    mask = D < eps
    X[mask] = 1.0 + np.tanh(phi * (D[mask] - eps))
    out = 1.0 - X
    out = np.clip(out, 0, 1)
    return out

def box_filter(img, r):
    return cv2.boxFilter(img, -1, (r*2+1, r*2+1), normalize=True)

def guided_filter(I, p, r=12, eps=1e-6):
    I = I.astype(np.float32); p = p.astype(np.float32)
    mean_I = box_filter(I,r)
    mean_p = box_filter(p,r)
    mean_Ip = box_filter(I*p,r)
    cov_Ip = mean_Ip - mean_I*mean_p
    mean_II = box_filter(I*I, r)
    var_I = mean_II - mean_I*mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = box_filter(a,r)
    mean_b = box_filter(b,r)
    q = mean_a * I + mean_b
    return q

def extract_tone(img_gray, radius=12):
    return guided_filter(img_gray, img_gray, r=radius)

# procedural hatching (simple)
def generate_hatch(shape, spacing=8, angle=0, thickness=1):
    h,w = shape
    canvas = np.zeros((h,w), dtype=np.uint8)
    for y in range(0, h+spacing, spacing):
        cv2.line(canvas, (0,y), (w,y), 255, thickness=thickness)
    if angle != 0:
        M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0)
        canvas = cv2.warpAffine(canvas, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return canvas.astype(np.float32)/255.0

# combine into procedural "target" (simple)
def procedural_target_from_photo(img_bgr, img_size=256):
    gray = to_gray_float(cv2.resize(img_bgr, (img_size, img_size)))
    tone = extract_tone(gray, radius=8)
    outline = xdog(gray, sigma=1.4, eps=0.06, phi=120)
    hatch = generate_hatch((img_size,img_size), spacing=8, angle=0)
    # soft compose: inverted hatch ink + outline
    hatch_ink = 1.0 - hatch
    hatch_contrib = hatch_ink * tone * 0.6
    final = 1.0 - hatch_contrib
    final = final - (1.0 - outline) * 0.9
    final = np.clip(final, 0.0, 1.0)
    return final  # float 0..1 white->black

# ---------------------------
# Dataset
# ---------------------------
class PhotoToSketchDataset(Dataset):
    def __init__(self, photos_dir, drawings_dir=None, img_size=256, transform=None, use_procedural=True):
        self.photos = sorted(glob(os.path.join(photos_dir, '*.*')))
        if len(self.photos)==0:
            raise RuntimeError("No photos found in " + photos_dir)
        self.drawings = sorted(glob(os.path.join(drawings_dir, '*.*'))) if drawings_dir else None
        self.img_size = img_size
        self.transform = transform
        self.use_procedural = use_procedural

    def __len__(self):
        return len(self.photos)

    def _load_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        return img

    def __getitem__(self, idx):
        img = self._load_img(self.photos[idx])
        # input: normalized RGB image
        input_img = img.astype(np.float32)/255.0
        # target: if user provided drawings, try to pair (index mod) else procedural
        if self.use_procedural or self.drawings is None:
            target = procedural_target_from_photo(cv2.cvtColor((input_img*255).astype(np.uint8), cv2.COLOR_RGB2BGR), img_size=self.img_size)
        else:
            dpath = self.drawings[idx % len(self.drawings)]
            dimg = cv2.imread(dpath, cv2.IMREAD_GRAYSCALE)
            dimg = cv2.resize(dimg, (self.img_size, self.img_size)).astype(np.float32)/255.0
            target = dimg
        # convert to tensors: input 3xHxW, target 1xHxW (white->black)
        input_t = torch.from_numpy(input_img.transpose(2,0,1)).float()
        target_t = torch.from_numpy(target[np.newaxis,:,:]).float()
        return input_t, target_t

# ---------------------------
# Models (UNet generator + PatchGAN discriminator)
# ---------------------------
def weight_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class UNetDown(nn.Module):
    def __init__(self, in_ch, out_ch, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=64):
        super().__init__()
        # encoder
        self.down1 = UNetDown(in_channels, features, normalize=False)
        self.down2 = UNetDown(features, features*2)
        self.down3 = UNetDown(features*2, features*4)
        self.down4 = UNetDown(features*4, features*8)
        self.down5 = UNetDown(features*8, features*8)
        self.down6 = UNetDown(features*8, features*8)
        # decoder
        self.up1 = UNetUp(features*8, features*8, dropout=0.5)
        self.up2 = UNetUp(features*16, features*8, dropout=0.5)
        self.up3 = UNetUp(features*16, features*8, dropout=0.5)
        self.up4 = UNetUp(features*16, features*4)
        self.up5 = UNetUp(features*8, features*2)
        self.up6 = UNetUp(features*4, features)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features*2, out_channels, 4, 2, 1),
            nn.Tanh()  # output in [-1,1] -> we will map to [0,1]
        )
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        u1 = self.up1(d6)
        u1 = torch.cat([u1, d5], dim=1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d4], dim=1)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d3], dim=1)
        u4 = self.up4(u3)
        u4 = torch.cat([u4, d2], dim=1)
        u5 = self.up5(u4)
        u5 = torch.cat([u5, d1], dim=1)
        u6 = self.up6(u5)
        out = self.final(torch.cat([u6, x], dim=1))
        # map tanh [-1,1] to [0,1]
        out = (out + 1.0) / 2.0
        return out

# PatchGAN discriminator
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=4, features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features, features*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features*2, features*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features*4, features*8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(features*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features*8, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------
# Perceptual loss (VGG)
# ---------------------------
class PerceptualVGG(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.selected = nn.Sequential(*[vgg[i] for i in range(0, 21)])  # relu4_1
        for p in self.selected.parameters(): p.requires_grad = False
    def forward(self, x, y):
        # expects x,y in [0,1], 1-channel or 3-channel.
        # expand to 3 channels for vgg if needed
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
            y = y.repeat(1,3,1,1)
        fx = self.selected(x)
        fy = self.selected(y)
        return F.l1_loss(fx, fy)

# ---------------------------
# Training loop
# ---------------------------
def train():
    cfg = CONFIG
    device = torch.device(cfg['device'])
    ds = PhotoToSketchDataset(cfg['data_photos_dir'], drawings_dir=cfg['data_drawings_dir'],
                              img_size=cfg['img_size'], use_procedural=cfg['use_procedural_pairs'])
    dl = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, drop_last=True)

    G = UNetGenerator(in_channels=3, out_channels=1).to(device)
    D = PatchDiscriminator(in_channels=4).to(device)
    G.apply(weight_init); D.apply(weight_init)

    optimizer_G = torch.optim.Adam(G.parameters(), lr=cfg['lr'], betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=cfg['lr'], betas=(0.5, 0.999))

    adversarial_loss = nn.MSELoss().to(device)
    l1_loss = nn.L1Loss().to(device)
    perceptual = PerceptualVGG().to(device)

    step = 0
    for epoch in range(cfg['epochs']):
        for inputs, targets in tqdm(dl, desc=f"Epoch {epoch+1}/{cfg['epochs']}"):
            inputs = inputs.to(device)                  # [B,3,H,W]
            targets = targets.to(device)                # [B,1,H,W]

            # ---------------------
            # Train Discriminator
            # ---------------------
            G.eval()
            with torch.no_grad():
                fake = G(inputs)                        # [B,1,H,W] in [0,1]
            real_pair = torch.cat([inputs, targets], dim=1)  # [B,4,H,W]
            fake_pair = torch.cat([inputs, fake], dim=1)
            pred_real = D(real_pair)
            pred_fake = D(fake_pair)
            valid = torch.ones_like(pred_real, device=device)
            fake_label = torch.zeros_like(pred_fake, device=device)
            loss_D = 0.5 * (adversarial_loss(pred_real, valid) + adversarial_loss(pred_fake, fake_label))

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # ---------------------
            # Train Generator
            # ---------------------
            G.train()
            fake = G(inputs)
            fake_pair = torch.cat([inputs, fake], dim=1)
            pred_fake = D(fake_pair)
            loss_G_adv = adversarial_loss(pred_fake, valid) * cfg['adv_weight']
            loss_G_l1 = l1_loss(fake, targets) * cfg['l1_weight']
            loss_G_perc = perceptual(fake, targets) * cfg['perceptual_weight']

            loss_G = loss_G_adv + loss_G_l1 + loss_G_perc

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            # Logging / save
            # ---------------------
            if step % 50 == 0:
                print(f"Step {step} D_loss {loss_D.item():.4f} G_loss {loss_G.item():.4f} (adv {loss_G_adv.item():.4f} l1 {loss_G_l1.item():.4f} perc {loss_G_perc.item():.4f})")
            if step % cfg['save_every'] == 0:
                # save model
                torch.save({'G':G.state_dict(),'D':D.state_dict(),'step':step}, os.path.join(cfg['output_dir'], f"ckpt_{step}.pth"))
            if step % cfg['val_every'] == 0:
                # save sample images
                G.eval()
                with torch.no_grad():
                    out = G(inputs[:4]).cpu()
                # make grid
                inp_cpu = inputs[:4].cpu()
                tar_cpu = targets[:4].cpu()
                grid_in = utils.make_grid(inp_cpu, nrow=4, normalize=True, scale_each=True)
                grid_out = utils.make_grid(out.repeat(1,3,1,1), nrow=4, normalize=True, scale_each=True)
                grid_tar = utils.make_grid(tar_cpu.repeat(1,3,1,1), nrow=4, normalize=True, scale_each=True)
                utils.save_image(grid_in, os.path.join(cfg['output_dir'], f"inp_{step}.png"))
                utils.save_image(grid_out, os.path.join(cfg['output_dir'], f"out_{step}.png"))
                utils.save_image(grid_tar, os.path.join(cfg['output_dir'], f"tar_{step}.png"))
            step += 1
        # end epoch
    # end training
    torch.save({'G':G.state_dict(),'D':D.state_dict(),'step':step}, os.path.join(cfg['output_dir'], f"final.pth"))
    print("Training finished. Models saved to", cfg['output_dir'])

# ---------------------------
# Inference helper
# ---------------------------
def infer_single(model_path, input_path, out_path, img_size=256, device=None):
    if device is None:
        device = torch.device(CONFIG['device'])
    G = UNetGenerator(in_channels=3, out_channels=1).to(device)
    ckpt = torch.load(model_path, map_location=device)
    G.load_state_dict(ckpt['G'])
    G.eval()
    img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size)).astype(np.float32)/255.0
    inp_t = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).to(device).float()
    with torch.no_grad():
        out = G(inp_t).cpu().squeeze(0).clamp(0,1).numpy()
    out_img = (out[0]*255).astype(np.uint8)
    cv2.imwrite(out_path, out_img)
    print("Saved sketch to", out_path)

# ---------------------------
# Main entry
# ---------------------------
if __name__ == "__main__":
    train()
    # Example inference after training:
    # infer_single('out_im2pencil/final.pth', 'test_photo.jpg', 'test_sketch.png')
