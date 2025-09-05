import os
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import numpy as np


class SketchDataset(Dataset):
    """
    Dataset for sketch-only images for Phase 2 fine-tuning.
    Resizes images with padding to preserve aspect ratio.
    Also returns a binary mask: 1 for real pixels, 0 for padding.
    """

    def __init__(self, root_dir, image_size=1024):
        super().__init__()
        self.root_dir = root_dir
        self.image_size = image_size

        self.files = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # grayscale normalization
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("L")  # grayscale

        img, mask = resize_with_padding(img, target_size=self.image_size, pad_color=255, return_mask=True)

        img_tensor = self.transform(img)
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)

        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()  # [1,H,W]

        return {"sketch": img_tensor, "mask": mask_tensor}


class PairedDataset(Dataset):
    """
    Paired (photo, sketch) dataset with masks for both.
    """

    def __init__(self, root_dir, image_size=1024):
        super().__init__()
        self.photo_dir = os.path.join(root_dir, "photos")
        self.sketch_dir = os.path.join(root_dir, "sketches")
        self.image_size = image_size

        if not os.path.isdir(self.photo_dir) or not os.path.isdir(self.sketch_dir):
            raise FileNotFoundError(
                f"Expecting '{root_dir}/photos' and '{root_dir}/sketches' folders."
            )

        photo_files = set(os.listdir(self.photo_dir))
        sketch_files = set(os.listdir(self.sketch_dir))
        self.files = sorted(list(photo_files.intersection(sketch_files)))
        if len(self.files) == 0:
            raise RuntimeError("No matching filenames found in photos/ and sketches/.")

        self.tf_photo = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.tf_sketch = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        photo = Image.open(os.path.join(self.photo_dir, fname)).convert("RGB")
        sketch = Image.open(os.path.join(self.sketch_dir, fname)).convert("L")

        photo, mask_p = resize_with_padding(photo, self.image_size, pad_color=(255, 255, 255), return_mask=True)
        sketch, mask_s = resize_with_padding(sketch, self.image_size, pad_color=255, return_mask=True)

        photo_tensor = self.tf_photo(photo)
        sketch_tensor = self.tf_sketch(sketch)
        mask_tensor = torch.from_numpy(mask_p).unsqueeze(0).float()  # [1,H,W]

        return {
            "photo": photo_tensor,
            "sketch": sketch_tensor,
            "mask": mask_tensor
        }

def resize_with_padding(img, target_size=1024, pad_color=255, return_mask=False, randomize_padding=False):
    """
    Resize image with aspect ratio preserved and pad to square target_size.
    Optionally return a binary mask (1=real pixels, 0=padding).
    If randomize_padding=True, distribute padding randomly instead of centering.
    """
    orig_w, orig_h = img.size
    ratio = float(target_size) / max(img.size)
    new_size = tuple([int(x * ratio) for x in img.size])
    img_resized = img.resize(new_size, Image.BICUBIC)

    delta_w = target_size - new_size[0]
    delta_h = target_size - new_size[1]

    if randomize_padding:
        pad_left  = random.randint(0, delta_w)
        pad_right = delta_w - pad_left
        pad_top   = random.randint(0, delta_h)
        pad_bottom= delta_h - pad_top
    else:
        pad_left  = delta_w // 2
        pad_right = delta_w - pad_left
        pad_top   = delta_h // 2
        pad_bottom= delta_h - pad_top

    padding = (pad_left, pad_top, pad_right, pad_bottom)

    # image with padding
    img_padded = ImageOps.expand(img_resized, padding, fill=pad_color)

    if not return_mask:
        return img_padded

    # mask (1=real, 0=padding)
    mask = np.ones((new_size[1], new_size[0]), dtype="uint8")
    mask = ImageOps.expand(Image.fromarray(mask), padding, fill=0)
    mask = np.array(mask, dtype="uint8")

    return img_padded, mask
# def resize_with_padding(img, target_size=1024, pad_color=255, return_mask=False):
#     """
#     Resize with padding to target_size, return optional binary mask.
#     Mask: 1 = real pixels, 0 = padding.
#     """
#     orig_w, orig_h = img.size
#     ratio = float(target_size) / max(img.size)
#     new_size = tuple([int(x * ratio) for x in img.size])
#     img_resized = img.resize(new_size, Image.BICUBIC)

#     delta_w = target_size - new_size[0]
#     delta_h = target_size - new_size[1]
#     padding = (
#         delta_w // 2,
#         delta_h // 2,
#         delta_w - (delta_w // 2),
#         delta_h - (delta_h // 2),
#     )

#     # image with padding
#     img_padded = ImageOps.expand(img_resized, padding, fill=pad_color)

#     if not return_mask:
#         return img_padded

#     # mask (before padding = 1, padding = 0)
#     import numpy as np
#     mask = np.ones((new_size[1], new_size[0]), dtype="uint8")
#     mask = ImageOps.expand(Image.fromarray(mask), padding, fill=0)
#     mask = np.array(mask)

#     return img_padded, mask


def remove_padding_and_resize(img, original_size):
    """
    Crop out padding from a square image and resize back to original size.
    Args:
        img (PIL.Image): Model output (square, e.g. 1024x1024).
        original_size (tuple): (width, height) of the original input.
    Returns:
        PIL.Image: Restored image with original aspect ratio.
    """
    target_size = img.size[0]  # square side
    orig_w, orig_h = original_size

    ratio = float(target_size) / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)

    delta_w = target_size - new_w
    delta_h = target_size - new_h
    left, top = delta_w // 2, delta_h // 2
    right, bottom = left + new_w, top + new_h

    img_cropped = img.crop((left, top, right, bottom))
    img_restored = img_cropped.resize((orig_w, orig_h), Image.BICUBIC)
    return img_restored
