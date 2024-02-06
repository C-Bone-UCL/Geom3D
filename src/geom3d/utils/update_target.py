import torch
from torch_geometric.data import Data
from geom3d.utils import database_utils
import stk
import pymongo
import numpy as np
import os
import pandas as pd
import torch.optim as optim
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import torch.nn.functional as Functional
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
from geom3d.utils import database_utils
from geom3d.dataloaders import dataloaders_GemNet
from geom3d.dataloaders.dataloaders_GemNet import DataLoaderGemNet
from geom3d.dataloaders.dataloaders_GemNetLEP import DataLoaderGemNetLEP
from geom3d.dataloaders.dataloaders_GemNetPeriodicCrystal import DataLoaderGemNetPeriodicCrystal
import importlib
from torch_cluster import radius_graph


def update_target_in_dataset(dataset, new_targets):
    for data, new_target in zip(dataset, new_targets):
        data.y = torch.tensor(new_target, dtype=torch.float32)

def main():
    # Load the original dataset
    original_dataset_path = "/rds/general/user/cb1319/home/GEOM3D/Geom3D/training/80Kdataset_radius.pt"
    dataset = torch.load(original_dataset_path)
    print("Original dataset loaded")

    # Load the new targets
    df_path = Path(
        "/rds/general/user/cb1319/home/GEOM3D/STK_path/", "data/output/Full_dataset/", "df_total_subset_16_11_23.csv"
    )

    df_precursors_path = Path(
        "/rds/general/user/cb1319/home/GEOM3D/STK_path/",
        "data/output/Prescursor_data/",
        "calculation_data_precursor_071123_clean.pkl",
    )

    df_total, df_precursors = database_utils.load_data_from_file(
        df_path, df_precursors_path
    )

    # Extract new targets from df_total_new
    targets_IP = df_total["ionisation potential (eV)"].values
    targets_ES1 = df_total["ES1"].values
    targets_fosc1 = df_total["fosc1"].values

    # Update the target in the dataset for each instance
    print("Updating targets in dataset...")
    update_target_in_dataset(dataset, targets_IP)
    
    # Save the dataset with new targets
    new_dataset_path_IP = "/rds/general/user/cb1319/home/GEOM3D/Geom3D/training/dataset80K_radius_IP.pt"
    torch.save(dataset, new_dataset_path_IP)
    print(f"New dataset saved: {new_dataset_path_IP}")

    # Update the target for ES1
    update_target_in_dataset(dataset, targets_ES1)
    
    # Save the dataset with new targets
    new_dataset_path_ES1 = "/rds/general/user/cb1319/home/GEOM3D/Geom3D/training/dataset80K_radius_ES1.pt"
    torch.save(dataset, new_dataset_path_ES1)
    print(f"New dataset saved: {new_dataset_path_ES1}")

    # Update the target for fosc1
    update_target_in_dataset(dataset, targets_fosc1)
    
    # Save the dataset with new targets
    new_dataset_path_fosc1 = "/rds/general/user/cb1319/home/GEOM3D/Geom3D/training/dataset80K_radius_fosc1.pt"
    torch.save(dataset, new_dataset_path_fosc1)
    print(f"New dataset saved: {new_dataset_path_fosc1}")


if __name__ == "__main__":
    main()
