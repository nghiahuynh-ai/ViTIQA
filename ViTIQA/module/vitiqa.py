import torch.nn as nn
from einops import repeat
from ViTIQA.module.embedding import IQAEmbedding


class ViTIQA(nn.Module):
    
    def __init__(self, config):
        super(ViTIQA, self).__init__()
        
        self.embedding = IQAEmbedding(config['embedding'])

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
        
    def forward(self, x0, x1, x2):
        x = self.embedding(x0, x1, x2)
        x = self.encoder(x)
        x = x[:, 0] # pick the first feature aka cls_token (B, D)
        x = self.mlp_head(x)
        return x
