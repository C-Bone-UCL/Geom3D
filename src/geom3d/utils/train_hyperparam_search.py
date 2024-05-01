"""
script to train the SchNet model on the STK dataset
created by Mohammed Azzouzi
date: 2023-11-14
"""
import os
import time
from argparse import ArgumentParser

import stk
import pymongo
import numpy as np
import os
import pandas as pd
import time
import wandb
import torch.nn as nn
import torch.optim as optim
import torch
import importlib
import optuna
import logging
import sys

import sklearn.datasets
import sklearn.linear_model
import sklearn.model_selection
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import lightning.pytorch as pl
import torch.nn.functional as Functional
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pathlib import Path
from geom3d import dataloader
from geom3d.dataloader import load_data, train_val_test_split, load_3d_rpr
from geom3d.dataloaders.dataloaders_GemNet import DataLoaderGemNet
from geom3d import models
from geom3d.models import SchNet, DimeNet, DimeNetPlusPlus, GemNet, SphereNet, SphereNetPeriodic, PaiNN, EquiformerEnergy
from geom3d.utils import database_utils
from geom3d.utils import model_setup_utils
from geom3d.utils.config_utils import read_config
from geom3d.utils.model_setup_utils import model_setup, hyperparameter_setup
from geom3d.pymodel import Pymodel, PrintLearningRate

importlib.reload(models)
importlib.reload(dataloader)
importlib.reload(model_setup_utils)


def objective(trial, config_dir):
    """
    Objective function for the hyperparameter search.

    Args:
    - trial (optuna.Trial): Optuna trial object
    - config_dir (str): directory to config.json

    Returns:
    - validation_loss (float): validation loss of the trained model
    """

    config = read_config(config_dir)

    config = hyperparameter_setup(config, trial)

    np.random.seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    config["device"] = "cuda" if torch.cuda.is_available() else torch.device("cpu")

    dataset = load_data(config)
    model, graph_pred_linear = model_setup(config, trial)

    # Initialize distributed training
    if dist.is_initialized():
        world_size, rank = init_distributed_training()
    else:
        world_size = 1
        rank = 0

    # Wrap the model with DistributedDataParallel if more than one GPU is available
    if torch.cuda.device_count() > 1:
        model = DistributedDataParallel(model)

    print("Model loaded: ", config["model_name"])

    effective_batch_size = config["batch_size"] // world_size
    train_loader, val_loader, test_loader = train_val_test_split(
        dataset, config=config, batch_size=effective_batch_size
    )

    if config["model_path"]:
        model = load_3d_rpr(model, config["model_path"])

    os.chdir(config["running_dir"])

    wandb_logger = WandbLogger(
        log_model="all",
        project=f"Geom3D_{config['model_name']}_{config['target_name']}",
        name=config["name"],
    )
    wandb_logger.log_hyperparams(config)

    #check if chkpt exists
    if os.path.exists(config["pl_model_chkpt"]):
        pymodel_SCHNET = Pymodel.load_from_checkpoint(config["pl_model_chkpt"])
    else:
        pymodel_SCHNET = Pymodel(model, graph_pred_linear, config)
    
    wandb_logger = WandbLogger(
        log_model="all",
        project=f"Geom3D_{config['model_name']}_{config['target_name']}",
        name=config["name"],
    )
    wandb_logger.log_hyperparams(config)

    # train model
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["name"],
        filename="{epoch}-{val_loss:.2f}-{other_metric:.2f}",
        monitor="val_loss",
        mode="min",
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,  # Adjust patience as needed
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    if config["mixed_precision"] is True:
        trainer = pl.Trainer(
            logger=wandb_logger,
            max_epochs=config["max_epochs_hp_search"],
            val_check_interval=1.0,
            log_every_n_steps=1,
            callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor, PrintLearningRate()],
            precision=16,  # 16-bit precision for mixed precision training
        )
    else:
        trainer = pl.Trainer(
            logger=wandb_logger,
            max_epochs=config["max_epochs_hp_search"],
            val_check_interval=1.0,
            log_every_n_steps=1,
            callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor, PrintLearningRate()],
        )

    if config["mixed_precision"] is True:
        print("Mixed precision training is activated.")
    else:
        print("Mixed precision training is not activated.")


    for epoch in range(config["max_epochs_hp_search"]):
        trainer.fit(
            model=pymodel_SCHNET,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        torch.cuda.empty_cache()

        intermediate_value = trainer.callback_metrics.get('val_loss')

        if intermediate_value is not None:
            intermediate_value = intermediate_value.item()
            trial.report(intermediate_value, epoch)
            print(f'Epoch {epoch + 1}, Validation Loss: {trainer.callback_metrics["val_loss"]}')
        else:
            print("Validation loss not found in trainer.callback_metrics.")
        
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()

    checkpoint_path = checkpoint_callback.best_model_path

    #retrieve validation loss
    validation_loss = checkpoint_callback.best_model_score.item()

    wandb.finish()
    
    # Print the path of the saved checkpoint
    print(f"Checkpoint saved at: {checkpoint_path}")

    return validation_loss

# function to initialize distributed training
def init_distributed_training():
    rank = int(os.environ.get('RANK', '0'))
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    print(f"Distributed training - GPU {rank + 1}/{torch.cuda.device_count()}")

    return dist.get_world_size(), rank