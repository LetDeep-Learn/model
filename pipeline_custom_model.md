# **Pipeline Summary — Photo → Pencil Sketch**

## 1. **Data Setup**

* **Paired set**: \~900 (photo, sketch) pairs — cleaned & aligned.
* **Unpaired set**: \~15k sketches (sketch-only). [ ] Optional
* **Split**:

  * Train: \~800 pairs
  * Val/Test: \~100 pairs [ ] Optional
* **Preprocessing**:

  * Resize to 1024x1024 [Model Input Size].
  * Normalize: `[0,1]` → `[-1,1]`.
  * Augmentation (on paired photos): flips, crops, color jitter, mild rotation.

---

## 2. **Model Architectures**

### Generator (**G**)

* **Type**: UNet (pix2pix-style) or ResNet encoder-decoder.
* **Input**: RGB photo (3×1024x1024) With Padding if needed.
* **Output**: grayscale sketch (1×1024×1024) Removes Padding if added.
* **Why UNet**: It Preserves spatial details (edges) via skip connections.

### Discriminators (**optional, for Stage 2**)

1. **Conditional Discriminator (D\_c)**

   * Input: concat(photo, sketch).
   * Checks if sketch matches the photo.
   * Based on PatchGAN (70×70).

2. **Unconditional Discriminator (D\_u)** [] Optional

   * Input: sketch only.
   * Real sketches (from 15k) vs generated ones.
   * Improves realism of stroke textures.

### (Optional) **Sketch Autoencoder**

* Learns “sketch manifold” from 15k sketches.
* Provides feature-matching loss for extra realism.
* Not strictly required at first.

---

## 3. **Loss Functions**

* **Stage 1 (paired only):**

  * **L1 Loss** between generated sketch & ground truth. (weight \~100).
  * **Perceptual Loss** (VGG features). (weight \~1–5).
* **Stage 2 (paired + unpaired):**

  * **+ Conditional Adversarial Loss** from D\_c.
  * **+ Unconditional Adversarial Loss** from D\_u.
  * **+ Feature Matching Loss** (from D features, stabilizes GAN).
* Balance: Keep **L1/Perceptual dominant** to preserve fidelity; adversarial only for style realism.

---

## 4. **Training Steps**

1. **Stage 1 — Supervised Pretrain**

   * Train G only with paired data.
   * Loss = `L1 + perceptual`.
   * Goal: faithful photo→sketch reproduction.
   * Save checkpoint (baseline).

2. **Stage 2 — Adversarial Fine-Tune (optional)**

   * Add D\_c (paired) + D\_u (unpaired sketches).
   * Loss = `L1 + perceptual + λ_adv (GAN losses) + feature matching`.
   * Slowly ramp up adversarial weight.
   * Monitor validation (don’t let fidelity drop).

---

## 5. **Data Flow Diagram**

```
           (Photo)
              │
              ▼
        ┌──────────┐
        │ Generator│───► Generated Sketch
        └──────────┘
              │
     ┌────────┴──────────┐
     │                   │
     ▼                   ▼
   Compare             Discriminators
 with paired           (optional):
   GT sketch           - D_c: checks photo+sketch
 (L1, perceptual)      - D_u: checks sketch realism
```

---
## 6. **Output Weigths**

1. Output Weights are Saved on Drive after every Five Epoches. (We can change to any number)
2. Three Checkpoints :- Epoch, Generator and latest.pth are saved after each specified Number.
3. latest.pth overrided after each new checkpoint .
4. Generator.pth is Ready to test the model .
5. Download the latest generator.pth and load it in inference script and its ready for testing.

---

## 7. **Inference Flow**

1. Load trained **G** weights.
2. Preprocess input photo (resize, normalize).
3. Forward pass → predicted sketch.
4. Post-process (denormalize, save as PNG/JPEG).

---

## 8. **Minimal Inference Code**

```python
import torch
import torchvision.transforms as T
from PIL import Image
import cv2

# --- Load Generator (UNet) ---
class UNetGenerator(torch.nn.Module):
    # (define same as in training)
    def __init__(self, in_ch=3, out_ch=1, nf=64):
        super().__init__()
        # define encoder-decoder with skip connections...
        # omitted for brevity
    def forward(self, x):
        # forward pass
        return x

# Load checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"
G = UNetGenerator().to(device)
G.load_state_dict(torch.load("generator_stage1.pth", map_location=device))
G.eval()

# --- Preprocessing ---
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # to [-1,1]
])

# Load input photo
img = Image.open("input.jpg").convert("RGB")
x = transform(img).unsqueeze(0).to(device)

# --- Forward pass ---
with torch.no_grad():
    y_pred = G(x)

# --- Postprocess ---
y_pred = (y_pred.squeeze().cpu().numpy() + 1) / 2.0  # back to [0,1]
y_pred = (y_pred * 255).astype("uint8")

# Save sketch
cv2.imwrite("sketch.png", y_pred)
```

---

## Summery

* **Start simple**: train UNet with `L1 + perceptual` on 900 pairs.
* **Optional refinement**: add GAN (D\_c + D\_u) to leverage 15k unpaired sketches.
* **Inference**: only generator is needed → very lightweight, fast photo→sketch.

---
