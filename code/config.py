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
    # Data
    images_root = "./dataset/renders/images"
    metadata_root = "./dataset/renders/" # metadata.json
    # Material parameter regression: predict the 5 Principled BSDF params
    # in the same order everywhere (dataset / model / metrics).
    material_param_names = ["metallic", "roughness", "specular",
                            "transmission", "ior"]
    num_material_params = 5
    # ior is normalized to [0,1] via (ior - ior_min) / (ior_max - ior_min);
    # the other four params are already in [0,1].
    ior_min = 1.3
    ior_max = 1.7
    sh_dim = 27
    image_size = 128
    
    # Training
    batch_size = 32
    learning_rate = 5e-4
    num_epochs = 50
    weight_decay = 1e-4
    
    # Loss weights
    lighting_loss_weight = 1.0
    material_loss_weight = 0.3
    
    # Model
    backbone = "resnet50" # "resnet18"/"efficientnet"
    pretrained = True
    
    # System
    device = _best_device()
    num_workers = 2
    seed = 42
    
    # Logging
    log_interval = 10  # print every N batches
    save_dir = "./models"
    experiment_name = "material_aware_lighting"

config = Config()