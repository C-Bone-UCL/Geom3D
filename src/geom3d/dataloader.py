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
from geom3d.pl_model import Pymodel,model_setup


def load_data(config):
    if config["load_dataset"]:
        if os.path.exists(config["dataset_path"]):
            if "device" in config.keys():
                dataset = torch.load(
                    config["dataset_path"], map_location=config["device"]
                )
            else:
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
        torch.save(dataset, "training/" + name + f"/{len(dataset)}dataset.pt")
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

        molecule = Data(
            x=atom_types, positions=positions, y=y, InChIKey=InChIKey
        )
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


def train_val_split(dataset, config):
    seed = config["seed"]
    num_mols = config["num_molecules"]
    assert num_mols <= len(dataset)
    np.random.seed(seed)
    all_idx = np.random.choice(len(dataset), num_mols, replace=False)
    Ntrain = int(num_mols * config["train_ratio"])
    train_idx = all_idx[:Ntrain]
    valid_idx = all_idx[Ntrain:]
    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    train_dataset = [dataset[x] for x in train_idx]
    valid_dataset = [dataset[x] for x in valid_idx]
    # Set dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
    )
    return train_loader, val_loader


def load_data_frag(config):
    dataset_opt = None
    if config["load_dataset"]:
        if os.path.exists(config["dataset_path_frag"]):
            print(f"loading dataset from {config['dataset_path_frag']}")
            if os.path.exists(config["dataset_path"]):
                if "device" in config.keys():
                    dataset = torch.load(
                        config["dataset_path_frag"],
                        map_location=config["device"],
                    )
                else:
                    dataset = torch.load(config["dataset_path_frag"])

            if os.path.exists(config["model_embedding_chkpt"]):
                chkpt_path = config["model_embedding_chkpt"]
                checkpoint = torch.load(chkpt_path)
                model, graph_pred_linear = model_setup(config)
                print("Model loaded: ", config["model_name"])
                # Pass the model and graph_pred_linear to the Pymodel constructor
                pymodel = Pymodel(model, graph_pred_linear)
                # Load the state dictionary
                pymodel.load_state_dict(state_dict=checkpoint["state_dict"])
                pymodel.freeze()
                pymodel.to(config["device"])
                model = pymodel.molecule_3D_repr
                return dataset, model
            else:
                print("model not found")
                return None, None
        else:
            print("dataset frag not found")
        if os.path.exists(config["dataset_path"]):
            if "device" in config.keys():
                dataset_opt = torch.load(
                    config["dataset_path"], map_location=config["device"]
                )
            else:
                dataset_opt = torch.load(config["dataset_path"])
        else:
            print("opt dataset not found")
        
    if dataset_opt is None:
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
        generate_function = generate_dataset_frag_pd
    else:
        print("loading dataset from org dataset")
        generate_function = generate_dataset_frag_dataset
        df_total = dataset_opt
    client = pymongo.MongoClient(config["pymongo_client"])
    db = stk.ConstructedMoleculeMongoDb(
        client,
        database=config["database_name"],
    )
    # check if model is in the path
    if os.path.exists(config["model_embedding_chkpt"]):
        chkpt_path = config["model_embedding_chkpt"]
        checkpoint = torch.load(chkpt_path)
        model, graph_pred_linear = model_setup(config)
        print("Model loaded: ", config["model_name"])
        # Pass the model and graph_pred_linear to the Pymodel constructor
        pymodel = Pymodel(model, graph_pred_linear)
        # Load the state dictionary
        pymodel.load_state_dict(state_dict=checkpoint["state_dict"])
        pymodel.freeze()
        pymodel.to(config["device"])
        model = pymodel.molecule_3D_repr
        dataset = generate_function(
            df_total,
            model,
            db,
            number_of_molecules=config["num_molecules"],
            number_of_fragement=config["number_of_fragement"],
            device=config["device"],
        )
        if config["save_dataset_frag"]:
            name = (
                config["name"]
                + "_frag_transf_"
                + str(config["number_of_fragement"])
            )
            os.makedirs(name, exist_ok=True)
            torch.save(dataset, name + "/dataset_frag.pt")
        return dataset, model
    else:
        print("model not found")
        return None, None


def generate_dataset_frag_pd(
    df_total,
    model,
    db,
    number_of_molecules=1000,
    number_of_fragement=6,
    device="cuda",
):
    molecule_index = np.random.choice(
        len(df_total), number_of_molecules, replace=False
    )
    data_list = []
    for i in molecule_index:
        moldata = fragment_based_encoding(
            df_total["InChIKey"][i],
            db,
            model,
            number_of_fragement,
            device=device,
        )
        if moldata is not None:
            data_list.append(moldata)
    return data_list


def generate_dataset_frag_dataset(
    dataset,
    model,
    db,
    number_of_molecules=1000,
    number_of_fragement=6,
    device="cuda",
):
    data_list = []
    molecule_index = np.random.choice(
        len(dataset), number_of_molecules, replace=False
    )
    for i in molecule_index:
        moldata = fragment_based_encoding(
            dataset[i]["InChIKey"],
            db,
            model,
            number_of_fragement,
            device=device,
        )
        if moldata is not None:
            data_list.append(moldata)
    return data_list


def fragment_based_encoding(
    InChIKey, db_poly, model, number_of_fragement=6, device=None
):
    # device = "cuda" if torch.cuda.is_available() else torch.device("cpu")
    polymer = db_poly.get({"InChIKey": InChIKey})
    frags = []
    dat_list = list(polymer.get_atomic_positions())
    positions = np.vstack(dat_list)
    positions = torch.tensor(positions, dtype=torch.float, device=device)
    atom_types = list(
        [
            atom.get_atom().get_atomic_number()
            for atom in polymer.get_atom_infos()
        ]
    )
    atom_types = torch.tensor(atom_types, dtype=torch.long, device=device)
    molecule = Data(x=atom_types, positions=positions, device=device)
    if len(list(polymer.get_building_blocks())) == number_of_fragement:
        with torch.no_grad():
            model.eval()

            opt_geom_encoding = model(molecule.x, molecule.positions)
        for molecule_bb in polymer.get_building_blocks():
            dat_list = list(molecule_bb.get_atomic_positions())
            positions = np.vstack(dat_list)
            positions = torch.tensor(
                positions, dtype=torch.float, device=device
            )
            atom_types = list(
                [atom.get_atomic_number() for atom in molecule_bb.get_atoms()]
            )
            atom_types = torch.tensor(
                atom_types, dtype=torch.long, device=device
            )
            molecule_frag = Data(
                x=atom_types,
                positions=positions,
                device=device,
                y=opt_geom_encoding,
                InChIKey=InChIKey,
            )
            frags.append(molecule_frag)
        return frags
