# load data and do augmentation

import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import json


class SphereDataset(Dataset):
    def __init__(self, split, config, transform=None):
        """
        Args:
            split: 'train', 'val', or 'test'
            config: Config object
            transform: torchvision transforms
        """
        self.config = config
        self.transform = transform

        # Load metadata (list of per-image dicts from data_gen.py)
        metadata_path = os.path.join(config.metadata_root, "metadata.json")
        with open(metadata_path, 'r') as f:
            all_entries = json.load(f)

        # Build parallel lists from the metadata entries
        self.image_paths = [
            os.path.join(config.images_root, entry["filename"])
            for entry in all_entries
        ]
        self.sh_coeffs = np.array(
            [entry["sh_coefficients"] for entry in all_entries], dtype=np.float32
        )
        self.material_labels = np.array(
            [entry["material_label"] for entry in all_entries], dtype=np.int64
        )

        # Train / val / test split (70 / 15 / 15)
        n = len(all_entries)
        rng = np.random.RandomState(config.seed)
        indices = rng.permutation(n)

        n_train = int(0.70 * n)
        n_val = int(0.15 * n)

        if split == "train":
            self.indices = indices[:n_train]
        elif split == "val":
            self.indices = indices[n_train:n_train + n_val]
        elif split == "test":
            self.indices = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split '{split}', expected train/val/test")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        # Load image
        image = Image.open(self.image_paths[real_idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'lighting': torch.FloatTensor(self.sh_coeffs[real_idx]),
            'material': torch.LongTensor([self.material_labels[real_idx]])[0],
            'index': real_idx,
        }
