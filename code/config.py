# all config stuff with gpu usage

import torch


def _best_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# All hyperparameters in one place
class Config:
    # Data — list of (images_root, metadata_root, source_tag) tuples
    data_roots = [
        ("./dataset/renders/images",    "./dataset/renders/",    "black"),
        ("./dataset/renders_bg/images", "./dataset/renders_bg/", "hdri"),
    ]

    # Curriculum learning — list of (last_epoch, {source_tag: weight})
    # Weights are relative mix proportions (don't need to sum to 1)
    curriculum = [
        (15, {"black": 1.0, "hdri": 0.0}),   # Phase 1: black bg only
        (25, {"black": 0.7, "hdri": 0.3}),   # Phase 2: mostly black
        (35, {"black": 0.5, "hdri": 0.5}),   # Phase 3: equal mix
        (50, {"black": 0.0, "hdri": 1.0}),   # Phase 4: hdri only
    ]
    num_material_classes = 5
    sh_dim = 27
    image_size = 128
    
    # Training
    batch_size = 32
    learning_rate = 5e-4
    num_epochs = 50
    weight_decay = 1e-4
    
    # Loss weights
    lighting_loss_weight = 1.0
    material_loss_weight = 0.5
    
    # Model
    backbone = "resnet50" # "resnet18"/"efficientnet"
    pretrained = True
    
    # System
    device = _best_device()
    num_workers = 1
    seed = 42
    
    # Logging
    log_interval = 10  # print every N batches
    save_dir = "./models"
    experiment_name = "material_aware_lighting"

config = Config()