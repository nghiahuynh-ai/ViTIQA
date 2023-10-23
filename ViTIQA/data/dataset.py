import os
import json
import torch
import numpy as np
import PIL.Image as Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F


class ViTrOCRDataset(Dataset):
    
    def __init__(self, cfg, vocab, parser=None):

        if not os.path.isfile(cfg.manifest_path):
            raise FileNotFoundError(f'Cannot find the manifest file with the given path: {cfg.manifest_path}')
        
        samples = []
        with open(cfg.manifest_path, 'r', encoding='utf8') as f:
            for line in f:
                line = json.loads(line)
                if os.path.isfile(line['path']):
                    samples.append((line['path'], line['text']))
        self.samples = samples
        
        if parser is None:
            parser = Parser(vocab)
        resizer = Resize(height=cfg.height, width=cfg.width)
        augmentor = ImgAugTransform() if cfg.augment else None
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(tuple(cfg.mean), tuple(cfg.std)),
        ])

        self.loader = DataLoader(
            self, 
            batch_size=cfg.batch_size, 
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
            collate_fn=ViTrOCRCollate(parser, resizer, transform, augmentor),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        return self.samples[idx]

    
class ViTrOCRCollate:
    
    def __init__(self, parser, resizer, transform, augmentor):
        
        self.parser = parser
        self.resizer = resizer
        self.transform = transform
        self.augmentor = augmentor
        
    def __call__(self, batch):
        
        w_max, l_max = 0, 0
        samples, samples_length, transcripts, transcripts_length = [], [], [], []
        
        for img_path, text in batch:
            image = Image.open(img_path)
            image = self.resizer(image)
            image = np.array(image.convert('RGB'))
            
            # augment & transform
            if self.augmentor is not None:
                image = self.augmentor(image)
            image = self.transform(image)
            
            # tokenize
            text = self.parser.encode(text)
            
            samples.append(image)
            transcripts.append(text)
            
            w_max = max(w_max, image.size(2))
            l_max = max(l_max, len(text))
        
        for idx in range(len(samples)):
            
            wi = samples[idx].size(2)
            pad = (0, 0, w_max - wi, 0)
            samples[idx] = F.pad(samples[idx], pad)
            samples_length.append(torch.tensor(wi, dtype=torch.long))
            
            li = len(transcripts[idx])
            transcripts[idx] += [self.parser.pad] * (l_max - li)
            transcripts[idx] = torch.tensor(transcripts[idx], dtype=torch.long)
            transcripts_length.append(torch.tensor(li, dtype=torch.long))

        samples = torch.stack(samples)
        samples_length = torch.stack(samples_length)
        transcripts = torch.stack(transcripts)
        transcripts_length = torch.stack(transcripts_length)
        
        return samples, samples_length, transcripts, transcripts_length


class ViTMAEDataset(Dataset):
    
    def __init__(self, cfg):

        if not os.path.isfile(cfg.manifest_path):
            raise FileNotFoundError(f'Cannot find the manifest file with the given path: {cfg.manifest_path}')
        
        samples = []
        with open(cfg.manifest_path, 'r', encoding='utf8') as f:
            for line in f:
                line = json.loads(line)
                if os.path.isfile(line['path']):
                    samples.append(line['path'])
        self.samples = samples
        
        resizer = Resize(height=cfg.height, width=cfg.width)
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.loader = DataLoader(
            self, 
            batch_size=cfg.batch_size, 
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
            collate_fn=ViTMAECollate(resizer, transform),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        return self.samples[idx]

    
class ViTMAECollate:
    
    def __init__(self, resizer, transform):

        self.resizer = resizer
        self.transform = transform
        
    def __call__(self, batch):
        
        samples = []
        
        for img_path in batch:
            image = Image.open(img_path)
            image = self.resizer(image)
            image = np.array(image.convert('RGB'))
            image = self.transform(image)
            samples.append(image)

        samples = torch.stack(samples)
        
        return samples
    