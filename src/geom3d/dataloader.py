"""
This module contains functions for loading the dataset and splitting it into training, validation, and test sets

Functions:
- load_data(config)
- load_3d_rpr(model, output_model_path)
- load_molecule(InChIKey, target, db)
- generate_dataset(df_total, df_precursors, db, model_name, radius, number_of_molecules=500)
- train_val_test_split(dataset, config, batch_size, smiles_list=None)
"""
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
from geom3d.utils import fragment_scaffold_split
from geom3d.utils.fragment_scaffold_split import *
from geom3d.utils import oligomer_scaffold_split
from geom3d.utils.oligomer_scaffold_split import *
from geom3d.utils import top_target_split
from geom3d.utils.top_target_split import *
from geom3d.utils import smart_data_split
from geom3d.utils.smart_data_split import *
from geom3d.dataloaders import dataloaders_GemNet
from geom3d.dataloaders.dataloaders_GemNet import DataLoaderGemNet
from geom3d.dataloaders.dataloaders_GemNetLEP import DataLoaderGemNetLEP
from geom3d.dataloaders.dataloaders_GemNetPeriodicCrystal import DataLoaderGemNetPeriodicCrystal
import importlib
from torch_cluster import radius_graph

importlib.reload(dataloaders_GemNet)
importlib.reload(fragment_scaffold_split)
importlib.reload(oligomer_scaffold_split)
importlib.reload(top_target_split)
importlib.reload(smart_data_split)

def load_data(config):
    """
    Load the dataset from the database or from a file

    Args:
    - config (dict): dictionary containing the configuration

    Returns:
    - dataset (list): list of Data objects

    """

    if config["load_dataset"]:
        if os.path.exists(config["dataset_path"]):
            print(config["dataset_path"])
            dataset = torch.load(config["dataset_path"])

            return dataset
        else:
            print(config["dataset_path"])
            print("dataset not found")
    df_path = Path(
        config["STK_path"], "data/output/Full_dataset/", config["df_total"]
    )
    df_precursors_path = Path(
        config["STK_path"],
        "data/output/Prescursor_data/",
        config["df_precursor"],
    )
    df_total, df_precursors = database_utils.load_data_from_file(
        df_path, df_precursors_path
    )
    client = pymongo.MongoClient(config["pymongo_client"])
    db = stk.ConstructedMoleculeMongoDb(
        client,
        database=config["database_name"],
    )
    
    if config["model_name"] == "PaiNN":
        dataset = generate_dataset(
            df_total,
            df_precursors,
            db,
            number_of_molecules=config["num_molecules"],
            model_name=config["model_name"],
            radius=config["model"]["cutoff"],
        )
    else:
        dataset = generate_dataset(
            df_total,
            df_precursors,
            db,
            number_of_molecules=config["num_molecules"],
            model_name=config["model_name"],
            radius=None
        )

    print(f"length of dataset: {len(dataset)}")

    # where the new dataset daves
    if config["save_dataset"]:
        os.makedirs(config['dataset_folder'], exist_ok=True)
        if config["model_name"] == "PaiNN":
            torch.save(dataset, f"{config['dataset_folder']}/{len(dataset)}dataset_radius_{config['target_name']}.pt")
            print(f"dataset saved to {config['dataset_folder']}/{len(dataset)}dataset_radius_{config['target_name']}.pt")
        else:    
            torch.save(dataset, f"{config['dataset_folder']}/{len(dataset)}dataset_{config['target_name']}.pt")
            print(f"dataset saved to {config['dataset_folder']}/{len(dataset)}dataset_{config['target_name']}.pt")
    
    return dataset


def load_3d_rpr(model, output_model_path):
    """
    Load the 3D representation model

    Args:
    - model (nn.Module): the model to be loaded
    - output_model_path (str): the path to the saved model

    Returns:
    - model (nn.Module): the loaded model
    """

    saved_model_dict = torch.load(output_model_path)
    model.load_state_dict(saved_model_dict["model"])
    model.eval()
    # check if the function has performed correctly
    print(model)
    return model


def load_molecule(InChIKey, target, db):
    """
    Load a molecule from the database

    Args:
    - InChIKey (str): the InChIKey of the molecule
    - target (float): the target value of the molecule
    - db (stk.ConstructedMoleculeMongoDb): the database

    Returns:
    - molecule (Data): the molecule as a Data object
    """

    polymer = None
    try:
        polymer = db.get({"InChIKey": InChIKey})
        # Print the complete dictionary returned from the database
        print("Database entry for InChIKey:", polymer)
    except KeyError:
        print(f"No key found in the database with a key of: {InChIKey}")
        # Handle the missing key case (e.g., return a default value or raise an exception) 

    if polymer is not None:
        dat_list = list(polymer.get_atomic_positions())
        positions = np.vstack(dat_list)
        positions = torch.tensor(positions, dtype=torch.float)
        atom_types = list(
            [
                atom.get_atom().get_atomic_number()
                for atom in polymer.get_atom_infos()
            ]
        )
        atom_types = torch.tensor(atom_types, dtype=torch.long)
        
        y = torch.tensor(target, dtype=torch.float32)

        molecule = Data(x=atom_types, positions=positions, y=y,
            InChIKey=InChIKey)
        return molecule
    else:
        return None

def generate_dataset(df_total, df_precursors, db, model_name, radius, number_of_molecules=500):
    """
    Generate a dataset of molecules

    Args:
    - df_total (pandas.DataFrame): the dataframe containing the total dataset
    - df_precursors (pandas.DataFrame): the dataframe containing the precursor dataset
    - db (stk.ConstructedMoleculeMongoDb): the database

    Returns:
    - data_list (list): list of Data objects

    """

    molecule_index = np.random.choice(
        len(df_total), number_of_molecules, replace=False
    )
    data_list = []
    for i in molecule_index:
        molecule = load_molecule(
                df_total["InChIKey"][i], df_total["target"][i], db
            )
        if model_name == "PaiNN":
            if molecule is not None:
                radius_edge_index = radius_graph(molecule.positions, r=radius, loop=False)
                molecule.radius_edge_index = radius_edge_index
                data_list.append(molecule)
        else:
            data_list.append(molecule)
    return data_list


def train_val_test_split(dataset, config, batch_size, smiles_list=None):
    """
    Split the dataset into a training, validation, and test set
    Can specify the split method in the config file

    Args:
    - dataset (list): list of Data objects
    - config (dict): dictionary containing the configuration
    - batch_size (int): the batch size
    - smiles_list (list): list of SMILES strings

    Returns:
    - train_loader (DataLoader): the training dataloader
    - val_loader (DataLoader): the validation dataloader
    - test_loader (DataLoader): the test dataloader
    - (train_smiles, valid_smiles, test_smiles) (tuple): tuple of lists of SMILES strings

    """

    seed = config["seed"]
    num_mols = len(dataset)
    np.random.seed(seed)
    smart_num_mols = min(config['smart_dataset_size'], num_mols)

    # Define the file path for storing the dataset split indices
    if config["split"] == "random":
        split_file_path = config["running_dir"] + f"/datasplit_{num_mols}_{config['split']}.npz"
    elif config["split"] == "fragment_scaffold":
        split_file_path = config["running_dir"] + f"/datasplit_{num_mols}_{config['split']}_threshold_{config['fragment_cluster_threshold']}_cluster_{config['test_set_fragment_cluster']}.npz"
    elif config["split"] == "oligomer_scaffold":
        split_file_path = config["running_dir"] + f"/datasplit_{num_mols}_{config['split']}_mincluster_{config['oligomer_min_cluster_size']}_minsample_{config['oligomer_min_samples']}_cluster_{config['test_set_oligomer_cluster']}.npz"
    elif config["split"] == "top_target":
        split_file_path = config["running_dir"] + f"/datasplit_{num_mols}_{config['split']}_target_{config['target_name']}_slice_{config['test_set_target_cluster']}.npz"
    elif config["split"] == "smart":
        split_file_path = config["running_dir"] + f"/datasplit_{smart_num_mols}_{config['split']}.npz"
    else:
        raise ValueError(f"Unknown split method: {config['split']}")

    if os.path.exists(split_file_path):
        # Load split indices from the file if it exists
        print(f"Loading dataset split indices from {split_file_path}")
        split_data = np.load(split_file_path)
        train_idx = split_data['train_idx']
        valid_idx = split_data['valid_idx']
        test_idx = split_data['test_idx']

    else:
        # Perform the split if the file doesn't exist
        if config["split"] == "random":
            print("random split")
            all_idx = np.random.permutation(num_mols)
            Nmols = num_mols
            Ntrain = int(num_mols * config["train_ratio"])
            Nvalid = int(num_mols * config["valid_ratio"])
            Ntest = Nmols - (Ntrain + Nvalid)
            train_idx = all_idx[:Ntrain]
            valid_idx = all_idx[Ntrain : Ntrain + Nvalid]
            test_idx = all_idx[Ntrain + Nvalid :]

        elif config["split"] == "smart":
            print("smart split")
            train_keys, valid_keys, test_keys = smart_data_splitter(dataset, config)
            train_idx = [i for i, data in enumerate(dataset) if data['InChIKey'] in train_keys]
            valid_idx = [i for i, data in enumerate(dataset) if data['InChIKey'] in valid_keys]
            test_idx = [i for i, data in enumerate(dataset) if data['InChIKey'] in test_keys]

            # remove overlappping indices
            train_idx = list(set(train_idx) - set(valid_idx) - set(test_idx))
            valid_idx = list(set(valid_idx) - set(train_idx) - set(test_idx))
            test_idx = list(set(test_idx) - set(train_idx) - set(valid_idx))

        elif config["split"] == "fragment_scaffold":
            print("fragment_scaffold split")
            cluster_keys = fragment_scaffold_splitter(dataset, config)
            print('size of cluster_keys:', len(cluster_keys))
            test_idx = [i for i, data in enumerate(dataset) if data['InChIKey'] in cluster_keys]
            remaining_idx = [i for i in range(len(dataset)) if i not in test_idx]
            np.random.shuffle(remaining_idx)
            split_idx = int(len(remaining_idx) * config["train_ratio"] / (config["train_ratio"] + config["valid_ratio"]))
            train_idx = remaining_idx[:split_idx]
            valid_idx = remaining_idx[split_idx:]

        elif config["split"] == "oligomer_scaffold":
            print("oligomer_scaffold split")
            cluster_keys = oligomer_scaffold_splitter(dataset, config)
            test_idx = [i for i, data in enumerate(dataset) if data['InChIKey'] in cluster_keys]
            remaining_idx = [i for i in range(len(dataset)) if i not in test_idx]
            np.random.shuffle(remaining_idx)
            split_idx = int(len(remaining_idx) * config["train_ratio"] / (config["train_ratio"] + config["valid_ratio"]))
            train_idx = remaining_idx[:split_idx]
            valid_idx = remaining_idx[split_idx:]


        elif config["split"] == "top_target":
            print("top_target split")
            top_target_keys = top_target_splitter(dataset, config)
            test_idx = [i for i, data in enumerate(dataset) if data['InChIKey'] in top_target_keys]
            remaining_idx = [i for i in range(len(dataset)) if i not in test_idx]
            np.random.shuffle(remaining_idx)
            split_idx = int(len(remaining_idx) * config["train_ratio"] / (config["train_ratio"] + config["valid_ratio"]))
            train_idx = remaining_idx[:split_idx]
            valid_idx = remaining_idx[split_idx:]

        else:
            raise ValueError(f"Unknown split method: {config['split']}")
        
        # Save split indices to the file
        np.savez(split_file_path, train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)
        print(f"Dataset split indices saved to {split_file_path}")

    print("train_idx: ", train_idx)
    print("valid_idx: ", valid_idx)
    print("test_idx: ", test_idx)
    
    print(set(train_idx).intersection(set(valid_idx)))
    print(set(valid_idx).intersection(set(test_idx)))
    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(valid_idx).intersection(set(test_idx))) == 0

    if not config["split"] == "smart":
        assert len(train_idx) + len(valid_idx) + len(test_idx) == num_mols

    train_dataset = [dataset[x] for x in train_idx]
    valid_dataset = [dataset[x] for x in valid_idx]
    test_dataset = [dataset[x] for x in test_idx]

    print("Number of training samples:", len(train_dataset))
    print("Number of validation samples:", len(valid_dataset))
    print("Number of test samples:", len(test_dataset))

    if config["model_name"] == "GemNet":
        dataloader_kwargs = {"cutoff": config["model"]["cutoff"], "int_cutoff": config["model"]["int_cutoff"], "triplets_only": config["model"]["triplets_only"]}
        DataLoaderClass = DataLoaderGemNet
    else:
        dataloader_kwargs = {}
        DataLoaderClass = DataLoader

    # Set dataloaders
    train_loader = DataLoaderClass(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["num_workers"],
        **dataloader_kwargs
    )
    val_loader = DataLoaderClass(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["num_workers"],
        **dataloader_kwargs
    )
    test_loader = DataLoaderClass(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["num_workers"],
        **dataloader_kwargs
    )
    if not smiles_list:
        return train_loader, val_loader, test_loader
    else:
        train_smiles = [smiles_list[i] for i in train_idx]
        valid_smiles = [smiles_list[i] for i in valid_idx]
        test_smiles = [smiles_list[i] for i in test_idx]
        return (
            train_loader,
            val_loader,
            test_loader,
            (train_smiles, valid_smiles, test_smiles),
        )
