# import os
# import cv2
# import numpy as np

# def post_process_image(image):
#     """Apply post-processing pipeline to make image look more like a pencil sketch."""
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Gaussian blur
#     blurred = cv2.GaussianBlur(gray, (3, 3), 0)

#     # Edge detection
#     edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

#     # Invert edges for pencil sketch look
#     inverted = cv2.bitwise_not(edges)

#     # Sharpen the image (to bring out pencil strokes)
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5, -1],
#                        [0, -1, 0]])
#     sharpened = cv2.filter2D(inverted, -1, kernel)

#     # Normalize intensity (contrast enhancement)
#     final = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)

#     return final


# def process_directory(input_dir, output_dir):
#     """Recursively process all images in input_dir and save in output_dir"""
#     for root, _, files in os.walk(input_dir):
#         # Recreate folder structure in output
#         rel_path = os.path.relpath(root, input_dir)
#         save_dir = os.path.join(output_dir, rel_path)
#         os.makedirs(save_dir, exist_ok=True)

#         for file in files:
#             if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
#                 input_path = os.path.join(root, file)
#                 output_path = os.path.join(save_dir, file)

#                 # Read image
#                 img = cv2.imread(input_path)
#                 if img is None:
#                     print(f"❌ Could not read {input_path}")
#                     continue

#                 # Post-process
#                 sketch = post_process_image(img)

#                 # Save
#                 cv2.imwrite(output_path, sketch)
#                 print(f"✅ Saved: {output_path}")

# # def process_directory(input_dir, output_dir):
# #     """Recursively process all images in input_dir and save to output_dir."""
# #     if not os.path.exists(output_dir):
# #         os.makedirs(output_dir)

# #     for root, _, files in os.walk(input_dir):
# #         for file in files:
# #             if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
# #                 input_path = os.path.join(root, file)
                
# #                 # Mirror directory structure
# #                 relative_path = os.path.relpath(root, input_dir)
# #                 output_subdir = os.path.join(output_dir, relative_path)
# #                 os.makedirs(output_subdir, exist_ok=True)

# #                 output_path = os.path.join(output_subdir, file)

# #                 # Read image
# #                 image = cv2.imread(input_path)
# #                 if image is None:
# #                     print(f"Skipping {input_path}, not a valid image.")
# #                     continue

# #                 # Apply post processing
# #                 processed = post_process_image(image)

# #                 # Save result
# #                 cv2.imwrite(output_path, processed)
# #                 print(f"Processed: {input_path} → {output_path}")

# if __name__ == "__main__":
#     input_directory = "new_weights_output"        # Change this to your input folder
#     output_directory = "post_processed_images"  # Output folder
#     process_directory(input_directory, output_directory)




import os
import cv2
import numpy as np

# Input and output directories
input_root = "new_weights_output"
output_root = "post_processed_images"

# Ensure output root exists
os.makedirs(output_root, exist_ok=True)

def post_process_image(img):
    """Apply post-processing to make it look more like a pencil sketch"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert image
    inv = 255 - gray

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(inv, (21, 21), sigmaX=0, sigmaY=0)

    # Blend using dodge technique
    sketch = cv2.divide(gray, 255 - blur, scale=256)

    # Optional: Enhance edges
    edges = cv2.Canny(gray, 50, 150)
    sketch = cv2.addWeighted(sketch, 0.8, edges, 0.2, 0)

    return sketch

def process_directory(input_dir, output_dir):
    """Recursively process all images in input_dir and save in output_dir"""
    for root, _, files in os.walk(input_dir):
        # Recreate folder structure in output
        rel_path = os.path.relpath(root, input_dir)
        save_dir = os.path.join(output_dir, rel_path)
        os.makedirs(save_dir, exist_ok=True)

        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                input_path = os.path.join(root, file)
                output_path = os.path.join(save_dir, file)

                # Read image
                img = cv2.imread(input_path)
                if img is None:
                    print(f"❌ Could not read {input_path}")
                    continue

                # Post-process
                sketch = post_process_image(img)

                # Save
                cv2.imwrite(output_path, sketch)
                print(f"✅ Saved: {output_path}")

if __name__ == "__main__":
    process_directory(input_root, output_root)
    print("🎉 Batch post-processing complete!")
