import os
from PIL import Image

# Directories
input_dir = "test"
output_dir = "test_jpg"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all PNG files
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".png"):
        input_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + ".jpg"
        output_path = os.path.join(output_dir, output_filename)

        with Image.open(input_path) as img:
            # Ensure image has an alpha channel
            if img.mode in ("RGBA", "LA"):
                # Create white background image
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])  # Paste using alpha channel as mask
                background.save(output_path, "JPEG")
            else:
                # No transparency, just convert
                rgb_img = img.convert("RGB")
                rgb_img.save(output_path, "JPEG")

        print(f"✅ Converted with white background: {filename} → {output_filename}")
