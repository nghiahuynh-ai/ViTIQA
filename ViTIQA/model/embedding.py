import torch
import torch.nn as nn
from ViTIQA.model.pos_enc import (
    PositionalEncoding2D,
    LearnablePositionalEncoding,
)
from einops import repeat
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
        self.cls_token = nn.Parameter(torch.randn(1, 1, config['d_model']))
        self.pos_emb = LearnablePositionalEncoding(
            d_model=config['d_model'],
            dropout=config['dropout'],
        )
        
    def forward(self, x):
        # x: (b, c, h, w) -> (b, h x w + 1, d)
        b = x.size(0)
        x = self.feature_extractor(x)
        x = self.patch_emb(x)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_emb(x)
        return x
        