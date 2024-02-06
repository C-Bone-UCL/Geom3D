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

def model_setup(config, trial=None):
    model_config = config["model"]
    
    if trial:
        batch_size = trial.suggest_int('batch_size', low=16, high=128, step=16)
        config["batch_size"] = batch_size
        lr_scheduler = trial.suggest_categorical("lr_scheduler", ["CosineAnnealingLR", "CosineAnnealingWarmRestarts", "StepLR"])
        config["lr_scheduler"] = lr_scheduler

        if config["model_name"] == "SchNet":
            emb_dim = trial.suggest_int("emb_dim", low=16, high=240, step=32)
            SchNet_cutoff = trial.suggest_int("SchNet_cutoff", low=2, high=10, step=1)
            config["model"]["emb_dim"] = emb_dim
            config["model"]["SchNet_cutoff"] = SchNet_cutoff

        elif config["model_name"] == "DimeNet":
            hidden_channels = trial.suggest_categorical("hidden_channels", [128, 300])
            num_output_layers = trial.suggest_int("num_output_layers", 2, 4)
            config["model"]["hidden_channels"] = hidden_channels
            config["model"]["num_output_layers"] = num_output_layers

        elif config["model_name"] == "DimeNetPlusPlus":
            hidden_channels = trial.suggest_categorical("hidden_channels", [128, 300])
            num_output_layers = trial.suggest_int("num_output_layers", 2, 4)
            config["model"]["hidden_channels"] = hidden_channels
            config["model"]["num_output_layers"] = num_output_layers

        elif config["model_name"] == "GemNet":
            num_blocks = trial.suggest_int("num_blocks", 3, 5)
            config["model"]["num_blocks"] = num_blocks

        elif config["model_name"] == "SphereNet":
            hidden_channels = trial.suggest_categorical("hidden_channels", [128, 320])
            num_layers = trial.suggest_int("num_layers", 3, 5)
            out_channels = trial.suggest_int("out_channels", 1, 3)
            config["model"]["hidden_channels"] = hidden_channels
            config["model"]["num_layers"] = num_layers
            config["model"]["out_channels"] = out_channels

        elif config["model_name"] == "PaiNN":
            n_atom_basis = trial.suggest_categorical("n_atom_basis", [32, 64, 128, 256])
            cutoff = trial.suggest_int("cutoff", 2, 10)
            config["model"]["n_atom_basis"] = n_atom_basis
            config["model"]["cutoff"] = cutoff

        elif config["model_name"] == "Equiformer":
            num_layers = trial.suggest_int("Equiformer_num_layers", 3, 5)
            config["model"]["num_layers"] = num_layers

    
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
                irreps_node_embedding='128x0e+64x1e+32x2e', 
                num_layers=6,
                irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
                fc_neurons=[64, 64], 
                irreps_feature='512x0e',
                irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
                rescale_degree=False, nonlinear_message=False,
                irreps_mlp_mid='384x0e+192x1e+96x2e',
                norm_layer='layer',
                alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0)
                # irreps_in=model_config["Equiformer_irreps_in"],
                # max_radius=model_config["Equiformer_radius"],
                # node_class=model_config["node_class"],
                # number_of_basis=model_config["Equiformer_num_basis"], 
                # irreps_node_embedding='64x0e+32x1e+16x2e', 
                # num_layers=4,
                # irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
                # fc_neurons=[32,32], 
                # irreps_feature='256x0e',
                # irreps_head='32x0e+16x1e+8x2e', num_heads=2, irreps_pre_attn=None,
                # rescale_degree=False, nonlinear_message=False,
                # irreps_mlp_mid='192x0e+96x1e+48x2e',
                # norm_layer='layer',
                # alpha_drop=0.3, proj_drop=0.1, out_drop=0.1, drop_path_rate=0.1)
        elif config["model"]["Equiformer_hyperparameter"] == 1:
            # This follows the hyper in Equiformer_nonlinear_bessel_l2_drop00
            model = EquiformerEnergy(
                irreps_in=model_config["Equiformer_irreps_in"],
                max_radius=model_config["Equiformer_radius"],
                node_class=model_config["node_class"],
                number_of_basis=model_config["Equiformer_num_basis"], 
                irreps_node_embedding='128x0e+64x1e+32x2e', 
                num_layers=6,
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