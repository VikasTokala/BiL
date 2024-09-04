import torch
import torch.nn as nn


class BinauralOneSourceLoss(nn.Module):
    def __init__(self, mode="cosine"):
        super().__init__()
        
        if mode not in ["cosine", "cosine_abs", "mse"]:
            raise ValueError("mode must be 'cosine', 'cosine_abs' or 'mse'")
    
        self._mode = mode
    
    def forward(self, model_output, target):
        out_doas = model_output["doa_cart"]
        target_doas = target["doa_cart"]

        if self._mode.startswith("cosine"):
            out_doas_norm = torch.linalg.norm(out_doas, dim=-1)
            target_doas_norm = torch.linalg.norm(target_doas, dim=-1)

            cos_angle = torch.sum(out_doas * target_doas, dim=-1)/(
                out_doas_norm * target_doas_norm + 1e-10)
            
            if self._mode == "cosine_abs":
                cos_angle = cos_angle.abs()

            loss = torch.mean(1 - cos_angle)
            
        elif self._mode == "mse":
            loss = torch.mean((out_doas - target_doas)**2)

        return {
            "loss": loss
        }
