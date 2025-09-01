import os
import torch
from torchvision import transforms
from PIL import Image
from dataset import resize_with_padding
from model import UNetGenerator

# ===== CONFIG =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "weights/generator_stage1_epoch30_clear.pth"
# CHECKPOINT_PATH = "weights/generator_epoch3.pth"

IMG_SIZE = 512
INPUT_DIR = "test_img"
OUTPUT_DIR = "output"

# ===== LOAD MODEL =====
generator = UNetGenerator(3, 1)
state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
generator.load_state_dict(state_dict)
generator.to(DEVICE)
generator.eval()

# ===== PREPROCESS SETUP =====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ===== PROCESS ALL IMAGES =====
os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
        img_path = os.path.join(INPUT_DIR, filename)
        save_path = os.path.join(OUTPUT_DIR, filename)

        # Load and preprocess image
        img = Image.open(img_path).convert("L")
        # img = Image.open(img_path).convert("RGB")

        img = resize_with_padding(img, IMG_SIZE)
        input_tensor = transform(img).unsqueeze(0)

        if input_tensor.shape[1] == 1:
            input_tensor = input_tensor.repeat(1, 3, 1, 1)

        input_tensor = input_tensor.to(DEVICE)

        # Inference
        with torch.no_grad():
            fake_img = generator(input_tensor)

        # Postprocess and save
        fake_img = fake_img.squeeze(0).cpu()
        fake_img = (fake_img * 0.5 + 0.5).clamp(0, 1)
        fake_pil = transforms.ToPILImage()(fake_img)
        fake_pil.save(save_path)

        print(f"✅ Saved: {save_path}")
