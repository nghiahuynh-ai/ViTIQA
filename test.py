from ViTIQA.iqaregressor import IQARegressor 
from omegaconf import OmegaConf
import torch
import time


config = OmegaConf.load('ViTIQA/config/base.yaml')
model = IQARegressor(config).to('cuda')
print(model)
print(sum(p.numel() for p in model.parameters()))

# x = torch.rand(60, 3, 32, 1024).to('cuda')
# y = torch.rand(60)
# t = time.time()
# x = model.training_step((x, y), 0)
# print(time.time() - t)