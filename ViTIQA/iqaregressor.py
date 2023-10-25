import torch.nn as nn
import numpy as np
import lightning.pytorch as pl
from omegaconf import DictConfig
from torch.optim import (
    AdamW,
    SGD,
    lr_scheduler,
)
from members.nghiahnh.src.ViTIQA.ViTIQA.module.vitiqa import ViT
from torchvision import transforms
from ViTIQA.util.optim import NoamScheduler
from ViTIQA.data.dataset import IQADataset
from torchmetrics import (
    SpearmanCorrCoef,
    PearsonCorrCoef,
)


class IQARegressor(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(IQARegressor, self).__init__()
        
        self.cfg = cfg
        
        self.model = ViT(config=cfg['encoder'])
        
        self.optimizer = SGD(
            params=self.parameters(),
            lr=cfg['optimizer']['lr'],
            momentum=0.9,
            weight_decay=0,
        )
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=3e4, 
            eta_min=0,
        )
        
        self.criterion = nn.L1Loss()
        
        self.metrics = {
            'SRCC': SpearmanCorrCoef().to(cfg['device']),
            'PLCC': PearsonCorrCoef().to(cfg['device']),
        }
        
        self.train_data = IQADataset(cfg['train_dataset'])
        self.valid_data = IQADataset(cfg['val_dataset'])

    def training_step(self, batch, batch_idx):
        imgs, scores = batch
        
        outputs = self.model(imgs).squeeze(1)
        loss = self.criterion(outputs, scores)
        
        log_dict = {
            "train_loss": {"value": loss, "on_step": True, "on_epoch": True, "prog_bar": True, "logger": True},
            "lr": {
                "value": self.optimizer.param_groups[0]['lr'], 
                "on_step": True, "on_epoch": True, "prog_bar": True, "logger": True
            }
        }
        self._logging(log_dict)

        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs, scores = batch
        
        outputs = self.model(imgs).squeeze(1)
        loss = self.criterion(outputs, scores)
        # srcc = self.metrics['SRCC'](outputs, scores)
        # plcc = self.metrics['PLCC'](outputs, scores)
        
        log_dict = {
            "val_loss": {"value": loss, "on_step": True, "on_epoch": True, "prog_bar": True, "logger": True},
            # "srcc": {"value": srcc, "on_step": False, "on_epoch": True, "prog_bar": True, "logger": True},
            # "plcc": {"value": plcc, "on_step": False, "on_epoch": True, "prog_bar": True, "logger": True},
        }
        self._logging(log_dict)

        return loss
    
    def test_step(self, batch, batch_idx):
        return
    
    def predict(self, imgs=None, imgs_dir=None):
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    tuple(self.cfg['processor']['mean']), 
                    tuple(self.cfg['processor']['std'])
                ),
            ])
        imgs = transform(np.array(imgs.convert('RGB')))
        scores = self.model(imgs.unsqueeze(0))
        return scores
    
    def _logging(self, logs: dict):
        for key in logs:
            self.log(
                key,
                logs[key]['value'],
                logs[key]['on_step'],
                logs[key]['on_epoch'],
                logs[key]['prog_bar'],
                logs[key]['logger'],
            )
        
    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }