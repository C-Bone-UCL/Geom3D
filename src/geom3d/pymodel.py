import torch.nn as nn
import torch.optim as optim
import torch
import lightning.pytorch as pl
import torch.nn.functional as Functional
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from pathlib import Path
from geom3d import dataloader
from geom3d.dataloader import load_data, train_val_test_split, load_3d_rpr
from geom3d.dataloaders.dataloaders_GemNet import DataLoaderGemNet
from geom3d import models
from geom3d.models import SchNet, DimeNet, DimeNetPlusPlus, GemNet, SphereNet, SphereNetPeriodic, PaiNN, EquiformerEnergy
from geom3d.utils import database_utils
from geom3d.utils.config_utils import read_config
import importlib


class PrintLearningRate(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        lr = trainer.optimizers[0].param_groups[0]['lr']
        print(f'Learning Rate for Epoch {trainer.current_epoch}: {lr:.5e}')


class Pymodel(pl.LightningModule):
    def __init__(self, model, graph_pred_linear, config):
        super().__init__()
        self.save_hyperparameters(ignore=['graph_pred_linear', 'model'])
        self.molecule_3D_repr = model
        self.graph_pred_linear = graph_pred_linear
        self.config = config

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        with torch.cuda.amp.autocast(enabled=self.trainer.precision == 16): # 16-bit precision for mixed precision training, activated only when self.trainer.precision == 16
            loss = self._get_preds_loss_accuracy(batch)
        
        lr = self.trainer.optimizers[0].param_groups[0]['lr']

        self.log("train_loss", loss, batch_size=batch.size(0))
        self.log('lr', lr, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch.size(0))

        return loss

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        with torch.cuda.amp.autocast(enabled=self.trainer.precision == 16): # 16-bit precision for mixed precision training, activated only when self.trainer.precision == 16
            loss = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("val_loss", loss, batch_size=batch.size(0))
        return loss

    def _get_preds_loss_accuracy(self, batch):
        """convenience function since train/valid/test steps are similar"""
        batch = batch.to(self.device)
        z = self.forward(batch)

        if self.graph_pred_linear is not None:
            loss = Functional.mse_loss(z, batch.y.unsqueeze(1))
        else:
            loss = Functional.mse_loss(z, batch.y)
        return loss

    def configure_optimizers(self):
        # set up optimizer
        # make sure the optimiser step does not reset the val_loss metrics

        config = self.config
        optimizer = torch.optim.Adam(self.parameters(), lr=config["lr"])

        lr_scheduler = None
        monitor = None

        if config["lr_scheduler"] == "CosineAnnealingLR":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, config["max_epochs"]
            )
            print("Apply lr scheduler CosineAnnealingLR")
        elif config["lr_scheduler"] == "CosineAnnealingWarmRestarts":
            lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, config["max_epochs"], eta_min=1e-4
            )
            print("Apply lr scheduler CosineAnnealingWarmRestarts")
        elif config["lr_scheduler"] == "StepLR":
            lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=config["lr_decay_step_size"], gamma=config["lr_decay_factor"]
            )
            print("Apply lr scheduler StepLR")
        else:
            print("lr scheduler {} is not included.")

        return [optimizer], [lr_scheduler]

        # optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        # return optimizer
    
    def forward(self, batch):
        batch = batch.to(self.device)
        model_name = type(self.molecule_3D_repr).__name__
        if model_name == "EquiformerEnergy":
            model_name = "Equiformer"

        if self.graph_pred_linear is not None:
            if model_name == "PaiNN":
                z = self.molecule_3D_repr(batch.x, batch.positions, batch.radius_edge_index, batch.batch).squeeze()
                z = self.graph_pred_linear(z)
            else:
                z = self.molecule_3D_repr(batch.x, batch.positions, batch.batch)
                z = self.graph_pred_linear(z)
        else:
            if model_name == "GemNet":
                z = self.molecule_3D_repr(batch.x, batch.positions, batch).squeeze()
            elif model_name == "Equiformer":
                z = self.molecule_3D_repr(node_atom=batch.x, pos=batch.positions, batch=batch.batch).squeeze()
            else:
                z = self.molecule_3D_repr(batch.x, batch.positions, batch.batch).squeeze()
        return z