from ViTIQA.model.vit import ViT 
from omegaconf import OmegaConf
import torch
import time


config = OmegaConf.load('ViTIQA/config/base.yaml')
model = ViT(config)
print(model)
print(sum(p.numel() for p in model.parameters()))

x = torch.rand(60, 3, 32, 1024)
t = time.time()
x = model(x)
print(time.time() - t)