import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as T

from config import config
from data.dataset import SphereDataset
from models.multi_task_model import MaterialAwareLightingNet
from training.losses import MultiTaskLoss
from training.train import Trainer

def main():
    # Set device
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Transforms