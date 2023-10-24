import torch.nn as nn
import lightning.pytorch as pl
from omegaconf import DictConfig
from torch.optim import AdamW
from ViTIQA.module.vit import ViT
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
        
        self.optimizer = AdamW(
            params=self.parameters(),
            lr=cfg['optimizer']['lr'],
            betas=cfg['optimizer']['betas'],
            weight_decay=cfg['optimizer']['weight_decay'],
            eps=1e-9,
        )
        self.scheduler = NoamScheduler(
            self.optimizer,
            factor=cfg['optimizer']['factor'],
            model_size=cfg['optimizer']['model_size'],
            warmup_steps=cfg['optimizer']['warmup_steps'],
        )
        
        self.criterion = nn.L1Loss()
        
        self.metrics = {
            'SRCC': SpearmanCorrCoef(),
            'PLCC': PearsonCorrCoef(),
        }
        
        self.train_data = IQADataset(cfg['train_dataset'])
        self.valid_data = IQADataset(cfg['valid_dataset'])
    
    def training_step(self, batch, batch_idx):
        imgs, scores = batch
        
        outputs = self.model(imgs).sigmoid()
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
        
        outputs = self.model(imgs).sigmoid()
        loss = self.criterion(outputs, scores)
        srcc = self.metrics['SRCC'](outputs, scores)
        plcc = self.metrics['PLCC'](outputs, scores)
        
        log_dict = {
            "train_loss": {"value": loss, "on_step": True, "on_epoch": True, "prog_bar": True, "logger": True},
            "srcc": {"value": srcc, "on_step": False, "on_epoch": True, "prog_bar": True, "logger": True},
            "plcc": {"value": plcc, "on_step": False, "on_epoch": True, "prog_bar": True, "logger": True},
        }
        self._logging(log_dict)

        return loss
    
    def test_step(self, batch, batch_idx):
        return
    
    def predict(self, imgs=None, imgs_dir=None):
        return
    
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
                "monitor": "srcc",
            },
        }