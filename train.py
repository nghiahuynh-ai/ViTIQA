from members.nghiahnh.src.ViTIQA.ViTIQA.iqaregressor import IQARegressor
import lightning.pytorch as pl
from omegaconf import OmegaConf
import torch

accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'

config = OmegaConf.load('ViTIQA/config/base.yaml')
config['device'] = accelerator

model = IQARegressor(config)

trainer = pl.Trainer(
    devices=1, 
    accelerator=accelerator, 
    max_epochs=20,
    enable_checkpointing=True, 
    logger=True,
    log_every_n_steps=1, 
    check_val_every_n_epoch=1,
    default_root_dir='.'
)

train_data=model.train_data.loader
val_data=model.valid_data.loader
trainer.fit(
    model=model,
    train_dataloaders=train_data,
    val_dataloaders=val_data,
)