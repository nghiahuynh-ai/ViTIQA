import torch.nn as nn


class ResNetBackbone(nn.Module):
    
    def __init__(self, config):
        super(ResNetBackbone, self).__init__()
        
        self.conv_in = nn.Conv2d(
            in_channels=config['in_channels'],
            out_channels=config['inner_channels'],
            kernel_size=config['kernel_size'],
            stride=config['stride'],
            padding=config['padding'],
        )
        blocks = []
        channels = config['inner_channels']
        for _ in range(config['n_blocks']):
            blocks.append(
                ResBlock(
                    channels=channels,
                    kernel_size=config['kernel_size'],
                    stride=config['stride'],
                    padding=config['padding'],
                )
            )
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=channels*2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.ReLU(),
                )
            )
            channels = channels * 2
        self.blocks = nn.Sequential(*blocks)
        self.conv_out = nn.Conv2d(
            in_channels=channels,
            out_channels=config['out_channels'],
            kernel_size=config['kernel_size'],
            stride=config['stride'],
            padding=config['padding'],
        )
        
    def forward(self, x):
        # x: (b, c, h, w) -> (b, c', h', w')
        x = nn.functional.relu(self.conv_in(x))
        x = self.blocks(x)
        x = nn.functional.relu(self.conv_out(x))
        return x
    
    
class ResBlock(nn.Module):
    
    def __init__(self, channels, kernel_size, stride, padding):
        super(ResBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x):
        return nn.functional.relu(x + self.block(x))