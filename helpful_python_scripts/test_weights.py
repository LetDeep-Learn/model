import os
import torch
from torchvision import transforms
from PIL import Image
from working_scripts.dataset import resize_with_padding
from model import UNetGenerator

# ===== CONFIG =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512
INPUT_DIR = "test_jpg"
# OUTPUT_DIR = "weights_output"
# WEIGHTS_DIR = "weights"
OUTPUT_DIR = "new_weights_output"
WEIGHTS_DIR = "new_weights"
# ===== PREPROCESS SETUP =====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ===== PROCESS EACH CHECKPOINT =====
for checkpoint_file in os.listdir(WEIGHTS_DIR):
    if checkpoint_file.endswith(".pth"):
        checkpoint_path = os.path.join(WEIGHTS_DIR, checkpoint_file)
        model_name = os.path.splitext(checkpoint_file)[0]
        model_output_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        # Load model
        generator = UNetGenerator(3, 1)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        generator.load_state_dict(state_dict)
        generator.to(DEVICE)
        generator.eval()

        print(f"🚀 Running inference with model: {model_name}")

        # Process all images
        for filename in os.listdir(INPUT_DIR):
            if filename.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                img_path = os.path.join(INPUT_DIR, filename)
                base_name = os.path.splitext(filename)[0]
                save_path = os.path.join(model_output_dir, base_name + ".jpg")

                # Load and preprocess image
                # img = Image.open(img_path).convert("L")
                img = Image.open(img_path)

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
