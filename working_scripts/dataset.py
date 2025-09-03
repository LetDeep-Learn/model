
## `dataset.py`

import os
from PIL import Image, ImageOps
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms




# dataset_phase2.py

class SketchDataset(Dataset):
    """
    Dataset for sketch-only images for Phase 2 fine-tuning.
    Resizes images with padding to preserve aspect ratio.
    """

    def __init__(self, root_dir, image_size=512):
        """
        Args:
            root_dir (str): Path to directory containing sketch images.
            image_size (int): Target image size (e.g., 512x512)
        """
        super().__init__()
        self.root_dir = root_dir
        self.image_size = image_size

        # Collect all image paths
        self.files = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        # Transform: convert to tensor and normalize to [-1,1]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # sketch is grayscale
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("L")  # grayscale

        # Resize with padding to preserve aspect ratio
        img = resize_with_padding(img, target_size=self.image_size, pad_color=(0, 0, 0))

        # Apply transforms
        img_tensor = self.transform(img)
        if img_tensor.shape[0] == 1:  # [C,H,W]
          img_tensor = img_tensor.repeat(3, 1, 1)

        return {
            "sketch": img_tensor  # [1, H, W] in [-1,1]
        }








class PairedDataset(Dataset):
    """Paired (photo, sketch) dataset under one root: photos/ and sketches/.
    Filenames are expected to match across both subfolders.
    """
    def __init__(self, root_dir, image_size=512):
        super().__init__()
        self.photo_dir = os.path.join(root_dir, "photos")
        self.sketch_dir = os.path.join(root_dir, "sketches")
        if not os.path.isdir(self.photo_dir) or not os.path.isdir(self.sketch_dir):
            raise FileNotFoundError(
                f"Expecting '{root_dir}/photos' and '{root_dir}/sketches' folders.")
        # Only keep files that exist in both
        photo_files = set(os.listdir(self.photo_dir))
        sketch_files = set(os.listdir(self.sketch_dir))
        self.files = sorted(list(photo_files.intersection(sketch_files)))
        if len(self.files) == 0:
            raise RuntimeError("No matching filenames found in photos/ and sketches/.")

        self.tf_photo = transforms.Compose([
            transforms.Lambda(lambda img: resize_with_padding(img, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.tf_sketch = transforms.Compose([
            transforms.Lambda(lambda img: resize_with_padding(img, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # self.tf_photo = transforms.Compose([
        #     transforms.Resize((image_size, image_size)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # ])
        # self.tf_sketch = transforms.Compose([
        #     transforms.Resize((image_size, image_size)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5], std=[0.5])
        # ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        photo = Image.open(os.path.join(self.photo_dir, fname)).convert("RGB")
        sketch = Image.open(os.path.join(self.sketch_dir, fname)).convert("L")
        return {
            "photo": self.tf_photo(photo),        # [3, H, W] in [-1,1]
            "sketch": self.tf_sketch(sketch)      # [1, H, W] in [-1,1]
        }



def resize_with_padding(img, target_size=512, pad_color=None):
    if pad_color is None:
        if img.mode == "L":
            pad_color = 0           # grayscale black
        else:
            pad_color = (0, 0, 0) 
    # Scale based on longest side
    ratio = float(target_size) / max(img.size)
    new_size = tuple([int(x * ratio) for x in img.size])
    img = img.resize(new_size, Image.BICUBIC)

    # Pad to 512x512
    delta_w = target_size - new_size[0]
    delta_h = target_size - new_size[1]
    padding = (delta_w // 2, delta_h // 2,
               delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    img = ImageOps.expand(img, padding, fill=pad_color)

    return img


def restore_original_aspect(output_img, new_size, padding, orig_size):
    """
    Restore model output (square) back to original aspect ratio.
    """
    # Crop padding
    left, top, right, bottom = padding
    crop_box = (left, top, 512 - right, 512 - bottom)
    cropped = output_img.crop(crop_box)

    # Resize to original
    restored = cropped.resize(orig_size, Image.BICUBIC)
    return restored


def remove_padding_and_resize(img, original_size):
    """
    Crop out padding from a square image and resize back to original size.
    Args:
        img (PIL.Image): Model output (square, e.g. 512x512).
        original_size (tuple): (width, height) of the original input.
    Returns:
        PIL.Image: Restored image with original aspect ratio.
    """
    target_size = img.size[0]  # square side (e.g. 512)
    orig_w, orig_h = original_size

    # Find scaling ratio
    ratio = float(target_size) / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)

    # Compute padding (must match resize_with_padding)
    delta_w = target_size - new_w
    delta_h = target_size - new_h
    left, top = delta_w // 2, delta_h // 2
    right, bottom = left + new_w, top + new_h

    # Crop out the padded region
    img_cropped = img.crop((left, top, right, bottom))

    # Resize back to original
    img_restored = img_cropped.resize((orig_w, orig_h), Image.BICUBIC)
    return img_restored
