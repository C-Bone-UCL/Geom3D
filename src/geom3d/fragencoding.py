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
from geom3d.test_train import Pymodel
from lightning.pytorch.callbacks import LearningRateMonitor

def main(config_dir):
    config = read_config(config_dir)
    os.chdir(config["running_dir"])
    np.random.seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    config["device"] = (
        "cuda" if torch.cuda.is_available() else torch.device("cpu")
    )
    #model_config = config["model"]

    dataset = load_data(config)
    train_loader, val_loader, test_loader = train_val_test_split(
        dataset, config=config
    )
    wandb.login()
    # model
    if os.path.exists(config["model_VAE_chkpt"]):
        EncodingModel = FragEncoding.load_from_checkpoint(config["model_VAE_chkpt"])
    else:
        EncodingModel = FragEncoding(
            config["emb_dim"], config["number_of_fragement"]
        )
    name = config["name"] + "_frag_" + str(config["number_of_fragement"])
    wandb_logger = WandbLogger(
        log_model="all", project="Geom3D_fragencoding", name=name
    )
    wandb_logger.log_hyperparams(config)

    # train model
    checkpoint_callback = ModelCheckpoint(
        dirpath=name,
        filename="{epoch}-{val_loss:.2f}-{other_metric:.2f}",
        monitor="val_loss",
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=config["max_epochs"],
        val_check_interval=1.0,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, lr_monitor],
    )
    trainer.fit(
        model=EncodingModel,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    wandb.finish()


class FragEncoding(pl.LightningModule):
    def __init__(self, embedding_dim=128, num_fragment=6):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Linear(num_fragment * embedding_dim, embedding_dim * 3),
            nn.ReLU(),
            nn.Linear(embedding_dim * 3, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim * 3),
            nn.ReLU(),
            nn.Linear(embedding_dim * 3, num_fragment * embedding_dim),
        )

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        lossvae, loss_encoder = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("train_loss_vae", lossvae)
        self.log("train_loss_encoder", loss_encoder)
        self.log("train_loss", lossvae+loss_encoder)
        return loss_encoder 

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        lossvae, loss_encoder = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("val_loss_vae", lossvae)
        self.log("val_loss_encoder", loss_encoder)
        self.log("val_loss", lossvae+loss_encoder)
        return loss_encoder 

    def _get_preds_loss_accuracy(self, batch):
        """convenience function since train/valid/test steps are similar"""
        x, y = batch.x, batch.y
        #x = x.view(x.size(0), -1)
        y_pred = self.encoder(x)
        z = self.decoder(y_pred)
        loss_encoder = Functional.mse_loss(y, y_pred)
        lossvae = Functional.mse_loss(z, x)
        #loss = Functional.mse_loss(z, x) + Functional.mse_loss(y, y_pred)
        return lossvae, loss_encoder

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
           # optimizer, self.epochs
        #)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return [optimizer], [lr_scheduler]


def load_data(config):
    if config["load_dataset"]:
        if os.path.exists(config["dataset_path_frag"]):
            dataset = torch.load(config["dataset_path_frag"])
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
    df_total, df_precursors = Database_utils.load_data_from_file(
        df_path, df_precursors_path
    )
    client = pymongo.MongoClient(config["pymongo_client"])
    db = stk.ConstructedMoleculeMongoDb(
        client,
        database=config["database_name"],
    )
    # check if model is in the path
    if os.path.exists(config["model_embedding_chkpt"]):
        pymodel = Pymodel.load_from_checkpoint(config["model_embedding_chkpt"])
        pymodel.freeze()
        pymodel.to(config["device"])
        model = pymodel.molecule_3D_repr
        dataset = generate_dataset_frag(
            df_total,
            model,
            db,
            number_of_molecules=config["num_molecules"],
            number_of_fragement=config["number_of_fragement"],
        )
        if config["save_dataset"]:
            name = config["name"] + "_frag_" + str(config["number_of_fragement"])
            os.makedirs(name, exist_ok=True)
            torch.save(dataset, name+"/dataset_frag.pt")
        return dataset
    else:
        print("model not found")
        return None


def load_3d_rpr(model, output_model_path):
    saved_model_dict = torch.load(output_model_path)
    model.load_state_dict(saved_model_dict["model"])
    model.eval()
    return model


def generate_dataset_frag(
    df_total, model, db, number_of_molecules=1000, number_of_fragement=6
):
    molecule_index = np.random.choice(
        len(df_total), number_of_molecules, replace=False
    )
    data_list = []
    for i in molecule_index:
        moldata = fragment_based_encoding(
            df_total["InChIKey"][i], db, model, number_of_fragement
        )
        if moldata is not None:
            data_list.append(moldata)
    return data_list


def fragment_based_encoding(InChIKey, db_poly, model, number_of_fragement=6):
    device = (
        "cuda" if torch.cuda.is_available() else torch.device("cpu")
    )
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
        for molecule_bb in polymer.get_building_blocks():
            dat_list = list(molecule_bb.get_atomic_positions())
            positions = np.vstack(dat_list)
            positions = torch.tensor(positions, dtype=torch.float, device=device)
            atom_types = list(
                [atom.get_atomic_number() for atom in molecule_bb.get_atoms()]
            )
            atom_types = torch.tensor(atom_types, dtype=torch.long, device=device)
            molecule_frag = Data(
                x=atom_types,
                positions=positions,
                device=device,
            )
            frags.append(molecule_frag)
        
        with torch.no_grad():
            model.eval()
            batch = Batch.from_data_list(frags).to(device)
            original_encoding = model(batch.x, batch.positions, batch.batch)
            original_encoding = original_encoding.reshape((-1,))
            original_encoding = original_encoding.unsqueeze(0)
            opt_geom_encoding = model(molecule.x, molecule.positions)
        return Data(x=original_encoding, y=opt_geom_encoding, InChIKey=InChIKey)


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
