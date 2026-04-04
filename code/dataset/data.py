# load data and do augmentation


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
        
        # Load metadata
        metadata_path = f"{config.metadata_root}/metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.image_paths = self.metadata['image_paths']
        self.sh_coeffs = np.array(self.metadata['sh_coeffs'])
        self.material_labels = np.array(self.metadata['material_labels'])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'lighting': torch.FloatTensor(self.sh_coeffs[idx]),
            'material': torch.LongTensor([self.material_labels[idx]])[0],
            'index': idx  # For debugging
        }