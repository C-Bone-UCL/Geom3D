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
from geom3d.transformer_utils import TransformerPredictor


def main(config_dir):
    config = read_config(config_dir)
    os.chdir(config["running_dir"])
    np.random.seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    config["device"] = (
        "cuda" if torch.cuda.is_available() else torch.device("cpu")
    )
    # model_config = config["model"]

    dataset, model = load_data(config)
    train_loader, val_loader, test_loader = train_val_test_split(
        dataset, config=config
    )
    wandb.login()
    # initilize model
    model_config = config["model"]
    model = SchNet(
        hidden_channels=model_config["emb_dim"],
        num_filters=model_config["SchNet_num_filters"],
        num_interactions=model_config["SchNet_num_interactions"],
        num_gaussians=model_config["SchNet_num_gaussians"],
        cutoff=model_config["SchNet_cutoff"],
        readout=model_config["SchNet_readout"],
        node_class=model_config["node_class"],
    )
    # model = model.to(device)
    # model
    EncodingModel = Fragment_encoder(
        input_dim=config["emb_dim"] * config["number_of_fragement"],
        model_dim=config["emb_dim"],
        num_heads=1,
        num_classes=model_config["emb_dim"],
        num_layers=1,
        dropout=0.0,
        lr=5e-4,
        warmup=50,
        max_iters=config["max_epochs"] * len(train_loader),
    )

    EncodingModel.add_encoder(model)

    if os.path.exists(config["model_transformer_chkpt"]):
        print("loading model from checkpoint")
        state_dict = torch.load(
            config["model_transformer_chkpt"], map_location=config["device"]
        )
        EncodingModel.load_state_dict(state_dict["state_dict"])
    name = (
        config["name"] + "_frag_transf_" + str(config["number_of_fragement"])
    )
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
    lr_monitor = LearningRateMonitor(logging_interval="step")
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


class Fragment_encoder(TransformerPredictor):
    def add_encoder(self, model_encoder):
        self.model_encoder = model_encoder

    def forward(self, batch, mask=None, add_positional_encoding=True):
        """
        Args:
            x: Input features of shape [Batch, SeqLen, input_dim]
            mask: Mask to apply on the attention outputs (optional)
            add_positional_encoding: If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        if self.model_encoder is not None:
            x = []
            for b in batch:
                x.append(self.model_encoder(b.x, b.positions, b.batch))
            x = torch.cat(x, dim=1)
        else:
            x = batch.x
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = self.output_net(x)
        return x

    def _calculate_loss(self, batch, mode="train"):
        # Fetch data and transform categories to one-hot vectors
        inp_data, labels = batch, batch[0].y.squeeze()

        # inp_data = F.one_hot(inp_data, num_classes=self.hparams.num_classes).float()

        # Perform prediction and calculate loss and accuracy
        preds = self.forward(inp_data, add_positional_encoding=True)
        loss = Functional.mse_loss(preds.view(-1, preds.size(-1)), labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logging
        self.log("%s_loss" % mode, loss)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")


def load_data(config):
    if config["load_dataset"]:
        if os.path.exists(config["dataset_path_frag"]):
            print(f"loading dataset from {config['dataset_path_frag']}")
            dataset = torch.load(config["dataset_path_frag"])
            if os.path.exists(config["model_embedding_chkpt"]):
                pymodel = Pymodel.load_from_checkpoint(
                    config["model_embedding_chkpt"]
                )
                pymodel.freeze()
                pymodel.to(config["device"])
                model = pymodel.molecule_3D_repr
                return dataset, model
            else:
                print("model not found")
                return None, None
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
    device = "cuda" if torch.cuda.is_available() else torch.device("cpu")
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

    # print("train_idx: ", train_idx)
    # print("valid_idx: ", valid_idx)
    # print("test_idx: ", test_idx)
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
        drop_last=True,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
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
