import torch
import torch.nn as nn
from ViTIQA.model.pos_enc import PositionalEncoding2D
from einops.layers.torch import Rearrange
from ViTIQA.model.backbone import ResNetBackbone


class IQAEmbedding(nn.Module):
    
    def __init__(self, config):
        super(IQAEmbedding, self).__init__()

        self.feature_extractor = ResNetBackbone(config['backbone'])
        self.patch_emb = nn.Sequential(
            PositionalEncoding2D(config['d_model']),
            nn.Dropout(config['dropout']),
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(config['d_model']),
            nn.Linear(config['d_model'], config['d_model']),
            nn.LayerNorm(config['d_model']),
        )
        
    def forward(self, x):
        # x: (b, c, h, w) -> (b, h x w + 1, d)
        x = self.feature_extractor(x)
        x = self.patch_emb(x)
        return x
        