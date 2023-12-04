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
    
    dataset = generate_dataset(
        df_total,
        df_precursors,
        db,
        number_of_molecules=config["num_molecules"],
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

def generate_dataset(df_total, df_precursors, db, number_of_molecules=500):
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
        data_list.append(molecule)
    return data_list


def train_val_test_split(dataset, config, smiles_list=None):
    seed = config["seed"]
    num_mols = len(dataset)
    np.random.seed(seed)
    all_idx = np.random.permutation(num_mols)

    Nmols = num_mols
    Ntrain = int(num_mols * config["train_ratio"])
    Nvalid = int(num_mols * config["valid_ratio"])
    Ntest = Nmols - (Ntrain + Nvalid)

    train_idx = all_idx[:Ntrain]
    valid_idx = all_idx[Ntrain : Ntrain + Nvalid]
    test_idx = all_idx[Ntrain + Nvalid :]

    print("train_idx: ", train_idx)
    print("valid_idx: ", valid_idx)
    print("test_idx: ", test_idx)
    # np.savez("customized_01", train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(valid_idx).intersection(set(test_idx))) == 0
    assert len(train_idx) + len(valid_idx) + len(test_idx) == num_mols
    train_dataset = [dataset[x] for x in train_idx]
    valid_dataset = [dataset[x] for x in valid_idx]
    test_dataset = [dataset[x] for x in test_idx]
    # Set dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
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