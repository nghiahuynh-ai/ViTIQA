import torch.nn as nn
from ViTIQA.model.embedding import IQAEmbedding


class IQARegressor(nn.Module):
    
    def __init__(self, config):
        super(IQARegressor, self).__init__()
        
        self.embedding = IQAEmbedding(config['encoder'])
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=config['encoder']['d_model'],
                nhead=config['encoder']['n_heads'],
                dim_feedforward=config['encoder']['dim_feedforward'],
                dropout=config['encoder']['dropout'],
                activation=config['encoder']['act'],
                batch_first=True,
            ),
            num_layers=config['encoder']['n_layers'],
        )
        self.mlp_head =  nn.Linear(config['encoder']['d_model'], config['n_classes'], bias=False)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = x[: 0] # pick the first feature aka cls_token (B, D)
        x = self.mlp_head(x)
        return x