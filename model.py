import torch
import torch.nn as nn
from torchvision.models import vgg16

# # ----------------------------
# # UNet Block
# # ----------------------------
# class UNetBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, down=True, dropout=False):
#         super().__init__()
#         if down:
#             self.seq = nn.Sequential(
#                 nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
#                 nn.BatchNorm2d(out_ch),
#                 nn.LeakyReLU(0.2, inplace=False)
#             )
#         else:
#             self.seq = nn.Sequential(
#                 nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
#                 nn.BatchNorm2d(out_ch),
#                 nn.ReLU(inplace=False)
#             )
#         self.do = nn.Dropout(0.5) if dropout else None

#     def forward(self, x):
#         x = self.seq(x)
#         if self.do is not None:
#             x = self.do(x)
#         return x

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, dropout=False):
        super().__init__()
        if down:
            # Downsampling block
            self.seq = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            # Upsampling block (replace ConvTranspose2d with Upsample + Conv)
            self.seq = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.do = nn.Dropout(0.5) if dropout else None

    def forward(self, x):
        x = self.seq(x)
        if self.do is not None:
            x = self.do(x)
        return x

# ----------------------------
# UNet Generator (pix2pix-style)
# ----------------------------
class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        # Downsample
        self.d1 = UNetBlock(in_ch, 64)
        self.d2 = UNetBlock(64, 128)
        self.d3 = UNetBlock(128, 256)
        self.d4 = UNetBlock(256, 512)
        self.d5 = UNetBlock(512, 512)
        self.d6 = UNetBlock(512, 512)
        self.d7 = UNetBlock(512, 512)
        self.d8 = UNetBlock(512, 512)
        # Upsample
        self.u1 = UNetBlock(512, 512, down=False, dropout=True)
        self.u2 = UNetBlock(1024, 512, down=False, dropout=True)
        self.u3 = UNetBlock(1024, 512, down=False, dropout=True)
        self.u4 = UNetBlock(1024, 512, down=False)
        self.u5 = UNetBlock(1024, 256, down=False)
        self.u6 = UNetBlock(512, 128, down=False)
        self.u7 = UNetBlock(256, 64, down=False)
        # self.final = nn.Sequential(
        #     nn.ConvTranspose2d(128, out_ch, 4, 2, 1),
        #     nn.Tanh()
        #     # nn.Sigmoid()
        # )
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(128, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        u1 = self.u1(d8)
        u2 = self.u2(torch.cat([u1, d7], dim=1))
        u3 = self.u3(torch.cat([u2, d6], dim=1))
        u4 = self.u4(torch.cat([u3, d5], dim=1))
        u5 = self.u5(torch.cat([u4, d4], dim=1))
        u6 = self.u6(torch.cat([u5, d3], dim=1))
        u7 = self.u7(torch.cat([u6, d2], dim=1))
        out = self.final(torch.cat([u7, d1], dim=1))
        return out


# ----------------------------
# PatchGAN Discriminator (conditional)
# ----------------------------
class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3 + 1):
        super().__init__()
        ch = 64
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(ch, ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 4),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(ch * 4, ch * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ch * 8),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(ch * 8, 1, 4, 1, 1)
        )

    def forward(self, photo, sketch):
        x = torch.cat([photo, sketch], dim=1)
        return self.model(x)


# ----------------------------
# Perceptual + Pixel Loss
# ----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

class PerceptualLoss(nn.Module):
    def __init__(
        self,
        layer_ids=(3, 8, 15),  # Early layers for edge and texture
        layer_weights=None,   # Dict: {layer_id: weight}
        perceptual_weight=0.3,
        pixel_weight=0.8
    ):
        super().__init__()
        vgg = vgg16(weights="IMAGENET1K_V1").features
        self.slices = nn.ModuleList([vgg[i] for i in layer_ids])
        for m in self.slices:
            for p in m.parameters():
                p.requires_grad = False

        if layer_weights is None:
            layer_weights = {i: 1.0 for i in layer_ids}
        self.layer_weights = layer_weights

        self.perceptual_weight = perceptual_weight
        self.pixel_weight = pixel_weight

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def _to_vgg_space(self, x):
        x = (x + 1) / 2  # [-1,1] to [0,1]
        return (x - self.mean) / self.std

    def forward(self, x, y):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.size(1) == 1:
            y = y.repeat(1, 3, 1, 1)

        x = self._to_vgg_space(x)
        y = self._to_vgg_space(y)

        perceptual_loss = 0.0
        for i, layer in zip(self.layer_weights.keys(), self.slices):
            x = layer(x)
            y = layer(y)
            perceptual_loss += self.layer_weights[i] * F.l1_loss(x, y)

        pixel_loss = F.l1_loss(x, y)

        return (self.perceptual_weight * perceptual_loss) + (self.pixel_weight * pixel_loss)

# class PerceptualLoss(nn.Module):
#     def __init__(self, layer_ids=(3, 8, 15, 22, 29), layer_weights=None, perceptual_weight=0.2, pixel_weight=1.0):
#         super().__init__()
#         vgg = vgg16(weights="IMAGENET1K_V1").features
#         self.slices = nn.ModuleList([vgg[i] for i in layer_ids])
#         for m in self.slices:
#             for p in m.parameters():
#                 p.requires_grad = False

#         if layer_weights is None:
#             layer_weights = [1.0] * len(layer_ids)
#         self.layer_weights = layer_weights

#         self.perceptual_weight = perceptual_weight
#         self.pixel_weight = pixel_weight

#         self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
#         self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

#     @torch.no_grad()
#     def _to_vgg_space(self, x):
#         x = (x + 1) / 2  # [-1,1] to [0,1]
#         return (x - self.mean) / self.std

#     def forward(self, x, y):
#         if x.size(1) == 1:
#             x = x.repeat(1, 3, 1, 1)
#         if y.size(1) == 1:
#             y = y.repeat(1, 3, 1, 1)

#         x = self._to_vgg_space(x)
#         y = self._to_vgg_space(y)

#         perceptual_loss = 0.0
#         for w, layer in zip(self.layer_weights, self.slices):
#             x = layer(x)
#             y = layer(y)
#             perceptual_loss += w * nn.functional.l1_loss(x, y)

#         pixel_loss = nn.functional.l1_loss(x, y)

#         return (self.perceptual_weight * perceptual_loss) + (self.pixel_weight * pixel_loss)
