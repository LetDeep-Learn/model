
## `dataset.py`

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

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
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.tf_sketch = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

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

