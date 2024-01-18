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
import time
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
from geom3d import dataloader
from geom3d.dataloader import load_data, train_val_test_split, load_3d_rpr
from geom3d.dataloaders.dataloaders_GemNet import DataLoaderGemNet
from geom3d import models
from geom3d.models import SchNet, DimeNet, DimeNetPlusPlus, GemNet, SphereNet, SphereNetPeriodic, PaiNN, EquiformerEnergy
from geom3d.utils import database_utils
from geom3d.utils.config_utils import read_config
import importlib

importlib.reload(models)
importlib.reload(dataloader)

def main(config_dir):
    start_time = time.time()

    config = read_config(config_dir)
    np.random.seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    config["device"] = (
        "cuda" if torch.cuda.is_available() else torch.device("cpu")
    )

    dataset = load_data(config)

    model, graph_pred_linear = model_setup(config)
    print("Model loaded: ", config["model_name"])

    train_loader, val_loader, test_loader = train_val_test_split(
        dataset, config=config
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

    checkpoint_path = checkpoint_callback.best_model_path

    wandb.finish()

    end_time = time.time()  # Record the end time
    total_time = end_time - start_time
    print(f"Total time taken for model training: {total_time} seconds")
    
    # Print the path of the saved checkpoint
    print(f"Checkpoint saved at: {checkpoint_path}")

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

        self.log("train_loss", loss, batch_size=batch.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("val_loss", loss, batch_size=batch.size(0))
        return loss

    def _get_preds_loss_accuracy(self, batch):
        """convenience function since train/valid/test steps are similar"""
        batch = batch.to(self.device)
        model_name = type(self.molecule_3D_repr).__name__
        z = self.forward(batch)

        if self.graph_pred_linear is not None:
            loss = Functional.mse_loss(z, batch.y.unsqueeze(1))
        else:
            loss = Functional.mse_loss(z, batch.y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer
    
    def forward(self, batch):
        batch = batch.to(self.device)
        model_name = type(self.molecule_3D_repr).__name__

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
                z = self.molecule_3D_repr(node_atom=batch.x, pos=batch.positions, batch=batch.batch)
            else:
                z = self.molecule_3D_repr(batch.x, batch.positions, batch.batch).squeeze()
        return z


def model_setup(config):
    model_config = config["model"]
    if config["model_name"] == "SchNet":
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

    elif config["model_name"] == "DimeNet":
        model = DimeNet(
            node_class=model_config["node_class"],
            hidden_channels=model_config["hidden_channels"],
            out_channels=model_config["out_channels"],
            num_blocks=model_config["num_blocks"],
            num_bilinear=model_config["num_bilinear"],
            num_spherical=model_config["num_spherical"],
            num_radial=model_config["num_radial"],
            cutoff=model_config["cutoff"],
            envelope_exponent=model_config["envelope_exponent"],
            num_before_skip=model_config["num_before_skip"],
            num_after_skip=model_config["num_after_skip"],
            num_output_layers=model_config["num_output_layers"],
        )
        graph_pred_linear = None

    elif config["model_name"] == "DimeNetPlusPlus":
        model = DimeNetPlusPlus(
            node_class=model_config["node_class"],
            hidden_channels=model_config["hidden_channels"],
            out_channels=model_config["out_channels"],
            num_blocks=model_config["num_blocks"],
            int_emb_size=model_config["int_emb_size"],
            basis_emb_size=model_config["basis_emb_size"],
            out_emb_channels=model_config["out_emb_channels"],
            num_spherical=model_config["num_spherical"],
            num_radial=model_config["num_radial"],
            cutoff=model_config["cutoff"],
            envelope_exponent=model_config["envelope_exponent"],
            num_before_skip=model_config["num_before_skip"],
            num_after_skip=model_config["num_after_skip"],
            num_output_layers=model_config["num_output_layers"],
        )
        graph_pred_linear = None

    elif config["model_name"] == "GemNet":
        model = GemNet(
            node_class=model_config["node_class"],
            num_targets=model_config["num_targets"],
            num_blocks=model_config["num_blocks"],
            emb_size_atom=model_config["emb_size_atom"],
            emb_size_edge=model_config["emb_size_edge"],
            emb_size_trip=model_config["emb_size_trip"],
            emb_size_quad=model_config["emb_size_quad"],
            emb_size_rbf=model_config["emb_size_rbf"],
            emb_size_cbf=model_config["emb_size_cbf"],
            emb_size_sbf=model_config["emb_size_sbf"],
            emb_size_bil_quad=model_config["emb_size_bil_quad"],
            emb_size_bil_trip=model_config["emb_size_bil_trip"],
            num_concat=model_config["num_concat"],
            num_atom=model_config["num_atom"],
            triplets_only=model_config["triplets_only"],
            direct_forces=model_config["direct_forces"],
            extensive=model_config["extensive"],
            forces_coupled=model_config["forces_coupled"],
            cutoff=model_config["cutoff"],
            int_cutoff=model_config["int_cutoff"],
            envelope_exponent=model_config["envelope_exponent"],
            num_spherical=model_config["num_spherical"],
            num_radial=model_config["num_radial"],
            num_before_skip=model_config["num_before_skip"],
            num_after_skip=model_config["num_after_skip"],
        )
        graph_pred_linear = None

    elif config["model_name"] == "SphereNet":
        model = SphereNet(
            energy_and_force=False,
            hidden_channels=model_config["hidden_channels"],
            out_channels=model_config["out_channels"],
            cutoff=model_config["cutoff"],
            num_layers=model_config["num_layers"],
            int_emb_size=model_config["int_emb_size"],
            basis_emb_size_dist=model_config["basis_emb_size_dist"],
            basis_emb_size_angle=model_config["basis_emb_size_angle"],
            basis_emb_size_torsion=model_config["basis_emb_size_torsion"],
            out_emb_channels=model_config["out_emb_channels"],
            num_spherical=model_config["num_spherical"],
            num_radial=model_config["num_radial"],
            envelope_exponent=model_config["envelope_exponent"],
            num_before_skip=model_config["num_before_skip"],
            num_after_skip=model_config["num_after_skip"],
            num_output_layers=model_config["num_output_layers"],
        )
        graph_pred_linear = None

    elif config["model_name"] == "PaiNN":
        model = PaiNN(
            n_atom_basis=model_config["n_atom_basis"],
            n_interactions=model_config["n_interactions"],
            n_rbf=model_config["n_rbf"],
            cutoff=model_config["cutoff"],
            max_z=model_config["max_z"],
            n_out=model_config["n_out"],
            readout=model_config["readout"],
        )
        graph_pred_linear = model.create_output_layers()

    elif config["model_name"] == "Equiformer":
        if config["model"]["Equiformer_hyperparameter"] == 0:
            # This follows the hyper in Equiformer_l2
            model = EquiformerEnergy(
                irreps_in=model_config["Equiformer_irreps_in"],
                max_radius=model_config["Equiformer_radius"],
                node_class=model_config["node_class"],
                number_of_basis=model_config["Equiformer_num_basis"], 
                irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
                irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
                fc_neurons=[64, 64], 
                irreps_feature='512x0e',
                irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
                rescale_degree=False, nonlinear_message=False,
                irreps_mlp_mid='384x0e+192x1e+96x2e',
                norm_layer='layer',
                alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0)
        elif config["model"]["Equiformer_hyperparameter"] == 1:
            # This follows the hyper in Equiformer_nonlinear_bessel_l2_drop00
            model = EquiformerEnergy(
                irreps_in=model_config["Equiformer_irreps_in"],
                max_radius=model_config["Equiformer_radius"],
                node_class=model_config["node_class"],
                number_of_basis=model_config["Equiformer_num_basis"], 
                irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
                irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
                fc_neurons=[64, 64], basis_type='bessel',
                irreps_feature='512x0e',
                irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
                rescale_degree=False, nonlinear_message=True,
                irreps_mlp_mid='384x0e+192x1e+96x2e',
                norm_layer='layer',
                alpha_drop=0.0, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0)
        graph_pred_linear = None
    
    else:
        raise ValueError("Invalid model name")
    
    return model, graph_pred_linear

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
    main(config_dir=config_dir)
