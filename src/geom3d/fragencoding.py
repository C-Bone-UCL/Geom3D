"""
this script is to encode the representation of the oligomer from the representation of the fragments
"""

import stk
import pymongo
import numpy as np
import os
import pandas as pd
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
from geom3d.models import SchNet
from geom3d import Database_utils
from pathlib import Path
from geom3d.config_utils import read_config

def main(config_dir):
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
    os.chdir(config["dir"])
    wandb.login()
    # model
    EncodingModel = FragEncoding(config['embedding_dim'], config["number_of_fragement"])
    name=config["name"]+"_frag_"+str(config["number_of_fragement"])
    wandb_logger = WandbLogger(log_model="all", project="Geom3D_fragencoding", name=name)
    wandb_logger.log_hyperparams(config)

    # train model
    checkpoint_callback = ModelCheckpoint(
        dirpath=name,
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
        model=EncodingModel,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    wandb.finish()

class FragEncoding(pl.LightningModule):
    def __init__(self, embedding_dim=128,num_fragment=6):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Linear(num_fragment*embedding_dim, embedding_dim*2), nn.ReLU(), nn.Linear(embedding_dim*2, embedding_dim))
        self.decoder =  nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim*2), nn.ReLU(), nn.Linear(embedding_dim*2, num_fragment*embedding_dim)
        )

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        loss = self._get_preds_loss_accuracy(batch)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("val_loss", loss)
        return loss

    def _get_preds_loss_accuracy(self, batch):
        """convenience function since train/valid/test steps are similar"""
        x, y = batch
        x = x.view(x.size(0), -1)
        y_pred = self.encoder(x)
        z = self.decoder(z)
        loss = Functional.mse_loss(z, x)+Functional.mse_loss(y, y_pred)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer


def load_data(config):
    df_path = Path(
        config["STK_path"], "data/output/Full_dataset/", config["df_total"]
    )
    df_precursors_path = Path(
        config["STK_path"],
        "data/output/Prescursor_data/",
        config["df_precursor"],
    )
    df_total, df_precursors = Database_utils.load_data_from_file(
        df_path, df_precursors_path
    )
    client = pymongo.MongoClient(config["pymongo_client"])
    db = stk.ConstructedMoleculeMongoDb(
        client,
        database=config["database_name"],
    )
    #check if model is in the path
    if os.path.exists(config["model_path"]):
        model = torch.jit.load(config["model_path"])
        dataset = generate_dataset_frag(
            df_total,
            model,
            db,
            number_of_molecules=config["num_molecules"],
            number_of_fragement=config["number_of_fragement"],
        )
        if config["save_dataset"]:
            torch.save(dataset, "dataset_frag.pt")
        return dataset
    else:
        print("model not found")
        return None

def generate_dataset_frag(df_total, model, db, number_of_molecules=1000,number_of_fragement):
    molecule_index = np.random.choice(
        len(df_total), number_of_molecules, replace=False
    )
    data_list = []
    for i in molecule_index:
        molecule = fragment_based_encoding(df_total["InChIKey"][i], db, model,number_of_fragement)
        if molecule is not None:
            data_list.append(molecule)
    return data_list


def fragment_based_encoding(InChIKey, db_poly, model, number_of_fragement=6):
    polymer = db_poly.get({"InChIKey": InChIKey})
    frags = []
    if len(list(polymer.get_building_blocks())) == number_of_fragement:
        for molecule in polymer.get_building_blocks():

            dat_list = list(molecule.get_atomic_positions())
            positions = np.vstack(dat_list)
            positions = torch.tensor(positions, dtype=torch.float)
            atom_types = list(
                [atom.get_atomic_number() for atom in molecule.get_atoms()]
            )
            atom_types = torch.tensor(atom_types, dtype=torch.long)
            molecule = Data(
                x=atom_types,
                positions=positions,
            )
            frags.append(molecule)
        batch = Batch.from_data_list(frags)
        original_encoding = model(batch.x, batch.positions, batch.batch)
        original_encoding = original_encoding.reshape((-1,))
        return original_encoding

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
    config_dir = root + args.config_dir
    main(config_dir=config_dir)
