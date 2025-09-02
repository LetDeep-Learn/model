# Image-to-Sketch Project

This project converts real photos into pencil-style sketches using a **pix2pix framework**.  
It is based on a **U-Net generator** and a **PatchGAN discriminator**.  

---

## 🚀 Overview

- **Input:** A real photo (RGB image).
- **Output:** A pencil-like sketch (grayscale or edge-style image).
- **Model:** Conditional GAN (pix2pix).
  - **Generator (U-Net):** Learns to translate photos into sketches.
  - **Discriminator (PatchGAN):** Judges small patches of the output to ensure realistic local details.

---

## 📂 Project Structure

```
project_root/
│── data/                # Training data (photos + sketches)
│── saved_models/        # Saved generator & discriminator weights
│── train.py             # Training script
│── inference.py         # For generating sketches from new photos
│── models.py            # U-Net (Generator) + PatchGAN (Discriminator)
│── utils.py             # Helper functions
│── README.md            # Documentation (this file)
```

---

## 🔧 How It Works

### 1. Generator (U-Net)
- Encoder: Compresses image into feature maps.
- Bottleneck: Learns abstract representation.
- Decoder: Reconstructs sketch, with skip connections to preserve details.

### 2. Discriminator (PatchGAN)
- Evaluates **70×70 patches** of the image instead of the whole image.
- Checks if local regions look real or fake.
- Helps generator produce sharp, detailed sketches.

---

## 🧠 Data Flow Diagram

### Training Flow
```mermaid
flowchart LR
    A[Photo + Sketch Pair] --> B[Generator (U-Net)]
    B --> C[Fake Sketch]
    C --> D[Discriminator (PatchGAN)]
    A --> D
    D -->|Real/Fake Loss| E[Backpropagation]
    E --> B
    E --> D
```

### Inference Flow
```mermaid
flowchart LR
    X[Photo] --> G[Trained Generator (U-Net)]
    G --> Y[Generated Sketch]
```

---

## 💾 What Gets Saved?

- `generator.pth` → Contains all learned weights of the U-Net (encoder + decoder).
- `discriminator.pth` → Contains PatchGAN weights (used only during training).
- For inference, **only the generator is needed**.

---

## ⚙️ Training Details

- **Optimizer:** Adam  
- **Learning Rate:** 2e-4 (can decay later if loss plateaus)  
- **Loss Function:**  
  - L1 Loss (for pixel similarity)  
  - GAN Loss (for realism)  

---

## ▶️ How to Run

### Train the Model
## Phase 2 has incomplete Work, but phase 1 is enough 
```bash
python train.py 
```
```bash
python train_phase1_clear.py 
```


### Generate Sketches

## We have option for Both Batch and Single Image Conversion 
<!-- ```bash
python inference.py --input path/to/photo.jpg --output result.jpg
```

--- -->

## 📌 Notes
- During training, both Generator & Discriminator are updated.  
- For inference, you only load the trained Generator weights.  
- Learning rate adjustments can help stabilize training if the model gets stuck.  

---


