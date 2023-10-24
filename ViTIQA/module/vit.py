import torch
import torch.nn as nn
from einops import repeat
from ViTIQA.module.embedding import IQAEmbedding
from ViTIQA.module.pos_enc import LearnablePositionalEncoding


class ViT(nn.Module):
    
    def __init__(self, config):
        super(ViT, self).__init__()
        
        self.embedding = IQAEmbedding(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config['d_model']))
        self.pos_emb = LearnablePositionalEncoding(
            d_model=config['d_model'],
            dropout=config['dropout'],
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=config['d_model'],
                nhead=config['n_heads'],
                dim_feedforward=config['dim_feedforward'],
                dropout=config['dropout'],
                activation=config['act'],
                batch_first=True,
            ),
            num_layers=config['n_layers'],
        )
        self.mlp_head = nn.Linear(config['d_model'], config['n_classes'])
        
    def forward(self, x):
        b = x.size(0)
        x = self.embedding(x)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_emb(x)
        x = self.encoder(x)
        x = x[:, 0] # pick the first feature aka cls_token (B, D)
        x = self.mlp_head(x)
        return x