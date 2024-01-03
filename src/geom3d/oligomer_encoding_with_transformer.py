"""
this script is to encode the representation of the oligomer from the representation of the fragments
"""

import numpy as np
import os

import wandb

import torch
import lightning.pytorch as pl
import torch.nn.functional as Functional
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from geom3d.models import SchNet
from lightning.pytorch.callbacks import LearningRateMonitor
from geom3d.transformer_utils import TransformerPredictor
from geom3d.utils.config_utils import read_config
from geom3d.dataloader import load_data_frag, train_val_split


def main(config_dir):
    config = read_config(config_dir)
    os.chdir(config["running_dir"])
    np.random.seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    config["device"] = (
        "cuda" if torch.cuda.is_available() else torch.device("cpu")
    )
    # model_config = config["model"]

    dataset, model = load_data_frag(config)
    train_loader, val_loader = train_val_split(dataset, config=config)
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
        log_model="all",
        project=f"Geom3D_frag_{config['model_name']}_{config['target_name']}",
        name=name,
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
