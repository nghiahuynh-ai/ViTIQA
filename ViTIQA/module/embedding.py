import torch
import torch.nn as nn
from einops import repeat
from ViTIQA.module.backbone import ResNetBackbone


class IQAEmbedding(nn.Module):
    
    def __init__(self, config):
        super(IQAEmbedding, self).__init__()
        
        self.config = config
        
        self.feature_extractor = ResNetBackbone.from_pretrained(config['backbone'])
        self.conv = nn.Conv2d(
            in_channels=2048, 
            out_channels=config['d_model'], 
            kernel_size=1,
        )
        
        self.spatial_emb = SpatialEmbedding(config['spatial_grid'], config['d_model'])
        self.scale_emb = ScaleEmbedding(config['d_model'])
        self.cls_token = nn.Parameter(torch.randn(1, 1, config['d_model']))
        
        self.proj = nn.Sequential(
            nn.LayerNorm(config['d_model']),
            nn.Linear(config['d_model'], config['d_model']),
            nn.LayerNorm(config['d_model']),
        )
        
    def forward(self, x0, x1, x2):
        # x: (b, c, h, w) -> (b, h x w, d)
        b, c, _, _ = x0.size()
        
        x0 = self.feature_extractor(x0)
        x1 = self.feature_extractor(x1)
        x2 = self.feature_extractor(x2)
        
        x0 = self.conv(x)
        x1 = self.conv(x1)
        x2 = self.conv(x2)
        
        x0 = self.spatial_emb(x0)
        x1 = self.spatial_emb(x1)
        x2 = self.spatial_emb(x2)
        
        x0, x1, x2 = self.scale_emb(x0, x1, x2)
        
        x0 = x0.reshape(b, c, -1)
        x1 = x1.reshape(b, c, -1)
        x2 = x2.reshape(b, c, -1)
        x0 = x0.permute(0, 2, 1)
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)
        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x0, x1, x2), dim=1)
        x = self.proj(x)
        
        return x
    
    
class SpatialEmbedding(nn.Module):
    
    def __init__(self, grid_size, d_model):
        super(SpatialEmbedding, self).__init__()
        
        self.grid_size = grid_size
        self.spatial_emb = nn.Parameter(torch.rand(1, grid_size, grid_size, d_model))
        
    def forward(self, x):
        b, c, h, w = x.size()
        emb = torch.zeros(b, c, h, w).to(x.device)
        for i in range(h):
            for j in range(w):
                t_i = int((i / h) * self.grid_size)
                t_j = int((j / w) * self.grid_size)
                emb[:, :, i, j] = self.spatial_emb[:, t_i, t_j, :]
        x = x + emb
        return x
        
        
class ScaleEmbedding(nn.Module):
    
    def __init__(self, d_model):
        super(ScaleEmbedding, self).__init__()
        
        self.scale_emb = {
            'scale_0': nn.Parameter(torch.rand(1, d_model, 1, 1)),
            'scale_1': nn.Parameter(torch.rand(1, d_model, 1, 1)),
            'scale_2': nn.Parameter(torch.rand(1, d_model, 1, 1)),
        }
        
    def forward(self, x0, x1, x2):
        b = x0.size(0)
        x0 = x0 + repeat(self.scale_emb['scale_0'], '1 c 1 1 -> b c h w', b=b, h=x0.size(2), w=x0.size(3))
        x1 = x1 + repeat(self.scale_emb['scale_1'], '1 c 1 1 -> b c h w', b=b, h=x1.size(2), w=x1.size(3))
        x2 = x2 + repeat(self.scale_emb['scale_2'], '1 c 1 1 -> b c h w', b=b, h=x2.size(2), w=x2.size(3))
        return x0, x1, x2
