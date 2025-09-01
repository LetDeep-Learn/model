import torch
from PIL import Image
import torchvision.transforms as T
import cv2

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
G = UNetGenerator().to(device)
G.load_state_dict(torch.load("generator_epoch100.pth", map_location=device))
G.eval()

# Preprocess input
transform = T.Compose([
    T.Resize((256,256)),
    T.ToTensor(),
    T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

img = Image.open("input.jpg").convert("RGB")
x = transform(img).unsqueeze(0).to(device)

# Forward pass
with torch.no_grad():
    y_pred = G(x)

# Postprocess
y_pred = (y_pred.squeeze().cpu().numpy() + 1) / 2.0  # to [0,1]
y_pred = (y_pred * 255).astype("uint8")

cv2.imwrite("sketch.png", y_pred)
