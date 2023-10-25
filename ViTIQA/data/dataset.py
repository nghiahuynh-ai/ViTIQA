import os
import json
import torch
import numpy as np
import PIL.Image as Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F


class IQADataset(Dataset):
    
    def __init__(self, config):

        if not os.path.isfile(config['manifest_path']):
            raise FileNotFoundError(f'Cannot find the manifest file with the given path: {config.manifest_path}')
        
        samples = []
        with open(config['manifest_path'], 'r', encoding='utf8') as f:
            for line in f:
                line = json.loads(line)
                if os.path.isfile(line['path']):
                    # if line['score'] < 0 or line['score'] > 1:
                    #     raise ValueError('Score must be in the range of [0, 1]!')
                    samples.append((line['path'], line['score']))
        self.samples = samples
        
        if config['flip']:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    tuple(config['norm']['mean']), 
                    tuple(config['norm']['std'])
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    tuple(config['norm']['mean']), 
                    tuple(config['norm']['std'])
                ),
            ])
        
        self.loader = DataLoader(
            self, 
            batch_size=config['batch_size'], 
            shuffle=config['shuffle'],
            num_workers=config['num_workers'],
            collate_fn=IQACollate(transform),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        return self.samples[idx]

    
class IQACollate:
    
    def __init__(self, transform):
        
        self.transform = transform
        
    def __call__(self, batch):
        
        h_max, w_max = 0, 0
        samples, scores = [], []
        
        for img_path, score in batch:
            image = Image.open(img_path)
            image = np.array(image.convert('RGB'))
            image = self.transform(image)
            samples.append(image)
            scores.append(score)
            
            h_max = max(h_max, image.size(1))
            w_max = max(w_max, image.size(2))
        
        for idx in range(len(samples)):
            hi = samples[idx].size(1)
            wi = samples[idx].size(2)
            pad = (0, w_max - wi, 0, h_max - hi)
            samples[idx] = F.pad(samples[idx], pad)
            scores[idx] = torch.tensor(scores[idx], dtype=torch.float32)

        samples = torch.stack(samples)
        scores = torch.stack(scores)
        
        return samples, scores
