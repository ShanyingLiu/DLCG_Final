import torch
import torch.nn as nn

class MultiTaskLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lighting_weight = config.lighting_loss_weight
        self.material_weight = config.material_loss_weight
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, pred_lighting, pred_material, target_lighting, target_material):
        loss_lighting = self.mse_loss(pred_lighting, target_lighting)
        loss_material = self.ce_loss(pred_material, target_material)
        
        total = (self.lighting_weight * loss_lighting + 
                 self.material_weight * loss_material)
        
        return total, loss_lighting, loss_material