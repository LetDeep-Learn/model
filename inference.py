# import torch
# from torchvision import transforms
# from PIL import Image
# import matplotlib.pyplot as plt
# from dataset resize_with_padding
# # from model.generator import Generator  # if you had this
# from model import UNetGenerator

# # ===== CONFIG =====
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # CHECKPOINT_PATH = "/content/rive/MyDrive/sketch_project/checkpoints/generator_stage1_epoch25.pth"
# CHECKPOINT_PATH = "weights/generator_stage1_epoch5.pth"

# IMG_PATH = "dataset/photos/0.jpg"  # <-- replace with your test sketch path
# IMG_SIZE = 512  # match whatever size you trained with
# SAVE_PATH = "generated_output.png"  # where to save result

# # ===== LOAD MODEL =====
# generator = UNetGenerator(3, 1)  # adjust in/out channels if needed

# # Always load checkpoint to CPU first, then move model to DEVICE
# state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
# generator.load_state_dict(state_dict)
# generator.to(DEVICE)
# generator.eval()

# # ===== PREPROCESS INPUT =====
# transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])  # same normalization as training
# ])

# img = Image.open(IMG_PATH).convert("L")  # sketch is grayscale
# img=resize_with_padding(img,512)

# input_tensor = transform(img).unsqueeze(0)  # [1,1,H,W]

# # If model expects 3 channels, repeat grayscale 3 times
# if input_tensor.shape[1] == 1:
#     input_tensor = input_tensor.repeat(1, 3, 1, 1)

# input_tensor = input_tensor.to(DEVICE)

# # ===== RUN INFERENCE =====
# with torch.no_grad():
#     fake_img = generator(input_tensor)

# # ===== POSTPROCESS =====
# fake_img = fake_img.squeeze(0).cpu()
# fake_img = (fake_img * 0.5 + 0.5).clamp(0, 1)  # de-normalize


# # Convert tensor to PIL for saving and visualization
# fake_pil = transforms.ToPILImage()(fake_img)

# # final_img = restore_original_aspect(fake_pil, new_size, padding, orig_size)
# # final_img.save("restored_output.jpg")

# # ===== SAVE =====
# fake_pil.save(SAVE_PATH)
# print(f"Generated image saved at {SAVE_PATH}")

# # ===== VISUALIZE =====
# plt.figure(figsize=(8,4))
# plt.subplot(1,2,1)
# plt.title("Input Sketch")
# plt.imshow(img, cmap="gray")
# plt.axis("off")

# plt.subplot(1,2,2)
# plt.title("Generated Image")
# plt.imshow(fake_pil)
# plt.axis("off")
# plt.show()


import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from dataset import resize_with_padding
from model import UNetGenerator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CHECKPOINT_PATH = "weights/generator_stage1_epoch5.pth"
CHECKPOINT_PATH = "/content/rive/MyDrive/sketch_project/checkpoints/generator_stage1_epoch25.pth"

IMG_PATH = "dataset/photos/0.jpg"
IMG_SIZE = 512
SAVE_PATH = "generated_output.png"

# Load model
generator = UNetGenerator(3, 1)
state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
generator.load_state_dict(state_dict)
generator.to(DEVICE)
generator.eval()

# Preprocess
img = Image.open(IMG_PATH).convert("L")
img = resize_with_padding(img, IMG_SIZE)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

input_tensor = transform(img).unsqueeze(0)
if input_tensor.shape[1] == 1:
    input_tensor = input_tensor.repeat(1,3,1,1)
input_tensor = input_tensor.to(DEVICE)

# Inference
with torch.no_grad():
    fake_img = generator(input_tensor)

# Postprocess
fake_img = fake_img.squeeze(0).cpu()
fake_img = (fake_img * 0.5 + 0.5).clamp(0,1)
fake_pil = transforms.ToPILImage()(fake_img)

# Save & visualize
fake_pil.save(SAVE_PATH)
print(f"Generated image saved at {SAVE_PATH}")

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Input Sketch")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Generated Image")
plt.imshow(fake_pil)
plt.axis("off")
plt.show()
