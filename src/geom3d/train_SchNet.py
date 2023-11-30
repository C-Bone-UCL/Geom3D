"""
script to train the SchNet model on the STK dataset
created by Mohammed Azzouzi
date: 2023-11-14
"""
import stk
import pymongo
import numpy as np
import os
import pandas as pd
<<<<<<< Updated upstream
=======
import time
>>>>>>> Stashed changes
import wandb
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import lightning.pytorch as pl
import torch.nn.functional as Functional
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
from geom3d.dataloader import load_data, train_val_test_split, load_3d_rpr
from geom3d.models import SchNet
from geom3d.utils import database_utils
from geom3d.utils.config_utils import read_config


def main(config_dir):
<<<<<<< Updated upstream
=======
    start_time = time.time()

>>>>>>> Stashed changes
    config = read_config(config_dir)
    np.random.seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    config["device"] = (
        "cuda" if torch.cuda.is_available() else torch.device("cpu")
    )
    dataset = load_data(config)
    train_loader, val_loader, test_loader = train_val_test_split(
        dataset, config=config
    )
    model_config = config["model"]
    model = SchNet(
        hidden_channels=model_config["emb_dim"],
        num_filters=model_config["SchNet_num_filters"],
        num_interactions=model_config["SchNet_num_interactions"],
        num_gaussians=model_config["SchNet_num_gaussians"],
        cutoff=model_config["SchNet_cutoff"],
        readout=model_config["SchNet_readout"],
        node_class=model_config["node_class"],
    )
    graph_pred_linear = torch.nn.Linear(
        model_config["emb_dim"], model_config["num_tasks"]
    )
    if config["model_path"]:
        model = load_3d_rpr(model, config["model_path"])
    os.chdir(config["running_dir"])
    wandb.login()
<<<<<<< Updated upstream
=======
    wandb.init(settings=wandb.Settings(start_method="fork"))
>>>>>>> Stashed changes
    # model
    #check if chkpt exists
    if os.path.exists(config["pl_model_chkpt"]):
        pymodel_SCHNET = Pymodel.load_from_checkpoint(config["pl_model_chkpt"])
    else:
        pymodel_SCHNET = Pymodel(model, graph_pred_linear)
    wandb_logger = WandbLogger(log_model="all", project="Geom3D", name=config["name"])
    wandb_logger.log_hyperparams(config)

    # train model
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["name"],
        filename="{epoch}-{val_loss:.2f}-{other_metric:.2f}",
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=config["max_epochs"],
        val_check_interval=1.0,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        model=pymodel_SCHNET,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    wandb.finish()

<<<<<<< Updated upstream
=======
    end_time = time.time()  # Record the end time
    total_time = end_time - start_time
    print(f"Total time taken for model training: {total_time} seconds")
    
>>>>>>> Stashed changes
    # load dataframe with calculated data


class Pymodel(pl.LightningModule):
    def __init__(self, model, graph_pred_linear):
        super().__init__()
        self.save_hyperparameters(ignore=['graph_pred_linear', 'model'])
        self.molecule_3D_repr = model
        self.graph_pred_linear = graph_pred_linear

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        loss = self._get_preds_loss_accuracy(batch)

<<<<<<< Updated upstream
        self.log("train_loss", loss)
=======
        self.log("train_loss", loss, batch_size=batch.size(0))
>>>>>>> Stashed changes
        return loss

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
<<<<<<< Updated upstream
        self.log("val_loss", loss)
=======
        self.log("val_loss", loss, batch_size=batch.size(0))
>>>>>>> Stashed changes
        return loss

    def _get_preds_loss_accuracy(self, batch):
        """convenience function since train/valid/test steps are similar"""
        z = self.molecule_3D_repr(batch.x, batch.positions, batch.batch)
        z = self.graph_pred_linear(z)
        loss = Functional.mse_loss(z, batch.y.unsqueeze(1))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer
    
    def forward(self, batch):
        z = self.molecule_3D_repr(batch.x, batch.positions, batch.batch)
        z = self.graph_pred_linear(z)
        return z


if __name__ == "__main__":
    from argparse import ArgumentParser
    root = os.getcwd()
    argparser = ArgumentParser()
    argparser.add_argument(
        "--config_dir",
        type=str,
        default="",
        help="directory to config.json",
    )
    args = argparser.parse_args()
<<<<<<< Updated upstream
    config_dir = root + args.config_dir
=======
    config_dir = args.config_dir
>>>>>>> Stashed changes
    main(config_dir=config_dir)
