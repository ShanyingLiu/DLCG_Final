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

        # Load and merge metadata from all data roots
        all_entries_with_paths = []
        for images_root, metadata_root in config.data_roots:
            metadata_path = os.path.join(metadata_root, "metadata.json")
            with open(metadata_path, 'r') as f:
                entries = json.load(f)
            for entry in entries:
                all_entries_with_paths.append((
                    os.path.join(images_root, entry["filename"]),
                    entry,
                ))

        # Build parallel lists
        self.image_paths = [p for p, _ in all_entries_with_paths]
        self.sh_coeffs = np.array(
            [e["sh_coefficients"] for _, e in all_entries_with_paths], dtype=np.float32
        )
        self.material_labels = np.array(
            [e["material_label"] for _, e in all_entries_with_paths], dtype=np.int64
        )

        # Train / val / test split (70 / 15 / 15)
        n = len(self.image_paths)
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
