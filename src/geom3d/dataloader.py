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
from geom3d.dataloaders import dataloaders_GemNet
from geom3d.dataloaders.dataloaders_GemNet import DataLoaderGemNet
from geom3d.dataloaders.dataloaders_GemNetLEP import DataLoaderGemNetLEP
from geom3d.dataloaders.dataloaders_GemNetPeriodicCrystal import DataLoaderGemNetPeriodicCrystal
import importlib
from torch_cluster import radius_graph

importlib.reload(dataloaders_GemNet)
importlib.reload(fragment_scaffold_split)

def load_data(config):
    if config["load_dataset"]:
        if os.path.exists(config["dataset_path"]):
            dataset = torch.load(config["dataset_path"])

            return dataset
        else:
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
        name = config["name"] 
        os.makedirs(name, exist_ok=True)
        torch.save(dataset, "training/"+name+f"/{len(dataset)}dataset.pt")
        print(f"dataset saved to {name}/{len(dataset)}dataset.pt")
    
    return dataset


def load_3d_rpr(model, output_model_path):
    saved_model_dict = torch.load(output_model_path)
    model.load_state_dict(saved_model_dict["model"])
    model.eval()
    # check if the function has performed correctly
    print(model)
    return model


def load_molecule(InChIKey, target, db):
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
    molecule_index = np.random.choice(
        len(df_total), number_of_molecules, replace=False
    )
    data_list = []
    for i in molecule_index:
        # try:
        #     molecule = load_molecule(
        #         df_total["InChIKey"][i], df_total["target"][i], db
        #     )
        #     data_list.append(molecule)
        # except KeyError:
        #     print(f"No key found in the database for molecule at index {i}")
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
    seed = config["seed"]
    num_mols = len(dataset)
    np.random.seed(seed)

    # Define the file path for storing the dataset split indices
    if config["split"] == "random":
        split_file_path = config["running_dir"] + f"/datasplit_{num_mols}_{config['split']}.npz"
    elif config["split"] == "fragment_scaffold":
        split_file_path = config["running_dir"] + f"/datasplit_{num_mols}_{config['split']}_threshold_{config['fragment_cluster_threshold']}_cluster_{config['test_set_fragment_cluster']}.npz"
    elif config["split"] == "oligomer_scaffold":
        split_file_path = config["running_dir"] + f"/datasplit_{num_mols}_{config['split']}_mincluster_{config['oligomer_min_cluster_size']}_minsample_{config['oligomer_min_samples']}_cluster_{config['test_set_oligomer_cluster']}.npz"
    elif config["split"] == "top_target":
        split_file_path = config["running_dir"] + f"/datasplit_{num_mols}_{config['split']}.npz"

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

        elif config["split"] == "fragment_scaffold":
            print("fragment_scaffold split")
            cluster_keys = fragment_scaffold_splitter(dataset, config)
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
            top_target_keys = top_target_split(dataset, config)
            test_idx = [i for i, data in enumerate(dataset) if data['InChIKey'] in top_target_keys]
            remaining_idx = [i for i in range(len(dataset)) if i not in test_idx]
            np.random.shuffle(remaining_idx)
            split_idx = int(len(remaining_idx) * config["train_ratio"] / (config["train_ratio"] + config["valid_ratio"]))
            train_idx = remaining_idx[:split_idx]
            valid_idx = remaining_idx[split_idx:]

        # Save split indices to the file
        np.savez(split_file_path, train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)
        print(f"Dataset split indices saved to {split_file_path}")

    print("train_idx: ", train_idx)
    print("valid_idx: ", valid_idx)
    print("test_idx: ", test_idx)
    
    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(valid_idx).intersection(set(test_idx))) == 0
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

# def train_val_test_split(dataset, config, batch_size, smiles_list=None):
#     seed = config["seed"]
#     num_mols = len(dataset)
#     np.random.seed(seed)

#     if config["split"] == "random":
#         print("random split")

#         all_idx = np.random.permutation(num_mols)

#         Nmols = num_mols
#         Ntrain = int(num_mols * config["train_ratio"])
#         Nvalid = int(num_mols * config["valid_ratio"])
#         Ntest = Nmols - (Ntrain + Nvalid)

#         train_idx = all_idx[:Ntrain]
#         valid_idx = all_idx[Ntrain : Ntrain + Nvalid]
#         test_idx = all_idx[Ntrain + Nvalid :]

#         # np.savez("customized_01", train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)

#     elif config["split"] == "fragment_scaffold":
#         print("fragment_scaffold split")

#         cluster_keys = fragment_scaffold_splitter(dataset, config)

#         # Get the test set indices based on InChIKeys
#         test_idx = [i for i, data in enumerate(dataset) if data['InChIKey'] in cluster_keys]
#         # Get the remaining indices for train and val set
#         remaining_idx = [i for i in range(len(dataset)) if i not in test_idx]
#         # Randomly shuffle the remaining indices
#         np.random.shuffle(remaining_idx)
#         # Split the remaining indices into train and val based on the specified ratio
#         split_idx = int(len(remaining_idx) * config["train_ratio"] / (config["train_ratio"] + config["valid_ratio"]))
#         train_idx = remaining_idx[:split_idx]
#         valid_idx = remaining_idx[split_idx:]

#     elif config["split"] == "oligomer_scaffold":
#         print("oligomer_scaffold split")

#         cluster_keys = oligomer_scaffold_splitter(dataset, config)

#         # Get the test set indices based on InChIKeys
#         test_idx = [i for i, data in enumerate(dataset) if data['InChIKey'] in cluster_keys]
#         # Get the remaining indices for train and val set
#         remaining_idx = [i for i in range(len(dataset)) if i not in test_idx]
#         # Randomly shuffle the remaining indices
#         np.random.shuffle(remaining_idx)
#         # Split the remaining indices into train and val based on the specified ratio
#         split_idx = int(len(remaining_idx) * config["train_ratio"] / (config["train_ratio"] + config["valid_ratio"]))
#         train_idx = remaining_idx[:split_idx]
#         valid_idx = remaining_idx[split_idx:]


#     print("train_idx: ", train_idx)
#     print("valid_idx: ", valid_idx)
#     print("test_idx: ", test_idx)
    
#     assert len(set(train_idx).intersection(set(valid_idx))) == 0
#     assert len(set(valid_idx).intersection(set(test_idx))) == 0
#     assert len(train_idx) + len(valid_idx) + len(test_idx) == num_mols

#     train_dataset = [dataset[x] for x in train_idx]
#     valid_dataset = [dataset[x] for x in valid_idx]
#     test_dataset = [dataset[x] for x in test_idx]

#     print("Number of training samples:", len(train_dataset))
#     print("Number of validation samples:", len(valid_dataset))
#     print("Number of test samples:", len(test_dataset))

#     if config["model_name"] == "GemNet":
#         dataloader_kwargs = {"cutoff": config["model"]["cutoff"], "int_cutoff": config["model"]["int_cutoff"], "triplets_only": config["model"]["triplets_only"]}
#         DataLoaderClass = DataLoaderGemNet
#     else:
#         dataloader_kwargs = {}
#         DataLoaderClass = DataLoader

#     # Set dataloaders
#     train_loader = DataLoaderClass(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=config["num_workers"],
#         **dataloader_kwargs
#     )
#     val_loader = DataLoaderClass(
#         valid_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=config["num_workers"],
#         **dataloader_kwargs
#     )
#     test_loader = DataLoaderClass(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=config["num_workers"],
#         **dataloader_kwargs
#     )
#     if not smiles_list:
#         return train_loader, val_loader, test_loader
#     else:
#         train_smiles = [smiles_list[i] for i in train_idx]
#         valid_smiles = [smiles_list[i] for i in valid_idx]
#         test_smiles = [smiles_list[i] for i in test_idx]
#         return (
#             train_loader,
#             val_loader,
#             test_loader,
#             (train_smiles, valid_smiles, test_smiles),
#         )
