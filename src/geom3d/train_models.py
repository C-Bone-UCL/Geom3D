"""
This script trains the model using the configuration file provided. 

The script loads the data, initializes the model, and trains the model using the hyperparameters
specified in the configuration file. 
The script also supports hyperparameter optimization using Optuna. 
The trained model is saved in the directory specified in the configuration file. 
The script also logs the training process using Weights and Biases. (WandB)

The script can be run using the following command:
python train_models.py --config_dir <path_to_config.json>

Args:
config_dir (str): directory to the config.json file

"""
import stk
import pymongo
import numpy as np
import os
import pandas as pd
import time
import wandb
import torch
from tqdm import tqdm
import optuna
import importlib
import logging
import sys

import sklearn.datasets
import sklearn.linear_model
import sklearn.model_selection
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import lightning.pytorch as pl
import torch.nn.functional as Functional
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from pathlib import Path


from geom3d import dataloader
from geom3d import models
from geom3d import pymodel
from geom3d.dataloader import load_data, train_val_test_split, load_3d_rpr
from geom3d.dataloaders.dataloaders_GemNet import DataLoaderGemNet
from geom3d.models import SchNet, DimeNet, DimeNetPlusPlus, GemNet, SphereNet, SphereNetPeriodic, PaiNN, EquiformerEnergy
from geom3d.utils import database_utils
from geom3d.utils import train_hyperparam_search
from geom3d.utils import model_setup_utils
from geom3d.utils.config_utils import read_config
from geom3d.utils.model_setup_utils import model_setup
from geom3d.utils.train_hyperparam_search import objective
from geom3d.pymodel import Pymodel, PrintLearningRate

importlib.reload(models)
importlib.reload(dataloader)
importlib.reload(train_hyperparam_search)
importlib.reload(pymodel)
importlib.reload(model_setup_utils)


def main(config_dir):
    """
    Main function to train the model

    Args:
    config_dir (str): directory to the config.json file

    Returns:
    None
    """
    start_time = time.time()

    config = read_config(config_dir)
    np.random.seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    config["device"] = (
        "cuda" if torch.cuda.is_available() else torch.device("cpu")
    )

    dataset = load_data(config)

    if config["hp_search"] is True:
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        if config["model_name"] == "PaiNN":
            # Initialize the hyperparameter optimization
            study = optuna.create_study(
                direction="minimize",
                storage="sqlite:///./hp_search/my_study.db",  # Specify the storage URL here.
                study_name=f"hyperparameter_optimization_{config['model_name']}_{config['target_name']}_2",
                pruner=optuna.pruners.MedianPruner(),
                load_if_exists=True,
            )
        else:
            study = optuna.create_study(
                direction="minimize",
                storage="sqlite:///./hp_search/my_study.db",  # Specify the storage URL here.
                study_name=f"hyperparameter_optimization_{config['model_name']}_{config['target_name']}",
                pruner=optuna.pruners.MedianPruner(),
                load_if_exists=True,
                )
        

        # Run hyperparameter optimization
        study.optimize(lambda trial: objective(trial, config_dir), n_trials=config["n_trials_hp_search"])

        # Retrieve the best hyperparameters
        best_params = study.best_params
        best_value = study.best_value

        print("Best Hyperparameters:", best_params)
        print("Best Validation Loss:", best_value)
     

    else:
        # Initialize distributed training
        if dist.is_initialized():
            world_size, rank = init_distributed_training()
        else:
            world_size = 1
            rank = 0

        model, graph_pred_linear = model_setup(config)

        # Wrap the model with DistributedDataParallel if more than one GPU is available
        if torch.cuda.device_count() > 1:
            model = DistributedDataParallel(model)
        
        print("Model loaded: ", config["model_name"])

        # effective batch size for distributed training
        effective_batch_size = config["batch_size"] // world_size

        train_loader, val_loader, test_loader = train_val_test_split(
            dataset, config=config, batch_size=effective_batch_size
        )

        if config["model_path"]:
            model = load_3d_rpr(model, config["model_path"])
        os.chdir(config["running_dir"])
        wandb.login()
        wandb.init(settings=wandb.Settings(start_method="fork"))
        # model
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

        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        if config["mixed_precision"] is True:
            trainer = pl.Trainer(
                logger=wandb_logger,
                max_epochs=config["max_epochs"],
                val_check_interval=1.0,
                log_every_n_steps=1,
                callbacks=[checkpoint_callback, lr_monitor, PrintLearningRate()],
                precision=16,  # 16-bit precision for mixed precision training
            )
        else:
            trainer = pl.Trainer(
                logger=wandb_logger,
                max_epochs=config["max_epochs"],
                val_check_interval=1.0,
                log_every_n_steps=1,
                callbacks=[checkpoint_callback, lr_monitor, PrintLearningRate()],
            )

        if config["mixed_precision"] is True:
            print("Mixed precision training is activated.")
        else:
            print("Mixed precision training is not activated.")

        for epoch in range(config["max_epochs"]):
            trainer.fit(
                model=pymodel_SCHNET,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )
            torch.cuda.empty_cache()

        checkpoint_path = checkpoint_callback.best_model_path

        wandb.finish()

        # Print the path of the saved checkpoint
        print(f"Checkpoint saved at: {checkpoint_path}")

    end_time = time.time()  # Record the end time
    total_time = end_time - start_time
    print(f"Total time taken for model training: {total_time} seconds")


# function to initialize distributed training
def init_distributed_training():
    rank = int(os.environ.get('RANK', '0'))
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    print(f"Distributed training - GPU {rank + 1}/{torch.cuda.device_count()}")

    return dist.get_world_size(), rank


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
    config_dir = args.config_dir

    # Initialize distributed training
    if dist.is_initialized():
        main(config_dir=config_dir)