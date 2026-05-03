# load data and do augmentation

import os
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SphereDataset(Dataset):
    def __init__(self, split, config, transform=None):
        self.config = config
        self.transform = transform

        metadata_path = os.path.join(config.metadata_root, "metadata.json")
        with open(metadata_path, 'r') as f:
            all_entries = json.load(f)

        self.image_paths = [
            os.path.join(config.images_root, entry["filename"])
            for entry in all_entries
        ]

        # Envmap path = envmaps_root / <stem>.npy where stem comes from hdri_file.
        self.envmap_paths = [
            os.path.join(config.envmaps_root,
                         os.path.splitext(entry["hdri_file"])[0] + ".npy")
            for entry in all_entries
        ]
        self.world_strengths = np.array(
            [entry["world_strength"] for entry in all_entries], dtype=np.float32
        )
        self.rotation_z_rads = np.array(
            [entry["rotation_z_rad"] for entry in all_entries], dtype=np.float32
        )

        # Material class label kept only for per-material grouping at eval time;
        # the model itself regresses continuous parameters below.
        self.material_labels = np.array(
            [entry["material_label"] for entry in all_entries], dtype=np.int64
        )

        ior_range = config.ior_max - config.ior_min
        self.material_params = np.array([
            [
                entry["metallic"],
                entry["roughness"],
                entry["specular"],
                entry["transmission"],
                (entry["ior"] - config.ior_min) / ior_range,
            ]
            for entry in all_entries
        ], dtype=np.float32)

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

    def _load_envmap(self, real_idx):
        env = np.load(self.envmap_paths[real_idx])  # (H, W, 3) float32
        env = env * float(self.world_strengths[real_idx])

        # Z-rotation: shift columns along width. Equirect azimuth maps to a
        # horizontal pixel roll. Sign chosen to match Blender's mapping-node
        # Z rotation; verify empirically when first integrating.
        H, W, _ = env.shape
        rot = float(self.rotation_z_rads[real_idx])
        shift = int(round(rot / (2.0 * np.pi) * W))
        if shift != 0:
            env = np.roll(env, shift, axis=1)
        return env  # (H, W, 3)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])

        image = Image.open(self.image_paths[real_idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)

        env = self._load_envmap(real_idx)                            # (H, W, 3)
        env_t = torch.from_numpy(env).permute(2, 0, 1).contiguous()  # (3, H, W)

        return {
            'image': image,
            'lighting': env_t,
            'material_params': torch.FloatTensor(self.material_params[real_idx]),
            'material_label': torch.LongTensor([self.material_labels[real_idx]])[0],
            'index': real_idx,
        }
