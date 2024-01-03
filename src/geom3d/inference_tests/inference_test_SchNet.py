"""
Script to test the how the model performs (model inference) on a single batch of data
Loads a trained PyTorch model checkpoint and plots the predicted values against the true values for three different datasets:
training set, validation set, and test set
"""
import os
import torch
import copy
from geom3d.train_models import model_setup, Pymodel, read_config
from geom3d.dataloader import (
    load_data,
    train_val_split,
    load_data_frag,
    generate_dataset_frag_dataset,
)
from torch_geometric.loader import DataLoader


def plot_training_results(chkpt_path, config_dir):
    """
    Function to plot the training results
    """
    import numpy as np

    config = read_config(config_dir)
    np.random.seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.manual_seed(config["seed"])

    config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("checkpoint used:", chkpt_path)

    # try:
    #     pymodel = Pymodel.load_from_checkpoint(chkpt_path)
    #     pymodel.freeze()
    # except (TypeError, KeyError):

    # to get the try and except start indent here
    checkpoint = torch.load(chkpt_path)
    model, graph_pred_linear = model_setup(config)
    print("Model loaded: ", config["model_name"])

    # Pass the model and graph_pred_linear to the Pymodel constructor
    pymodel = Pymodel(model, graph_pred_linear)

    # Load the state dictionary
    pymodel.load_state_dict(state_dict=checkpoint["state_dict"])

    # Set the model to evaluation mode
    pymodel.eval()

    # end indent here

    dataset = load_data(config)
    np.random.seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    print("y_true", dataset[0].y)
    pymodel_cpu = copy.deepcopy(pymodel).to("cpu")
    print("y_pred_cpu", pymodel_cpu(dataset[0].to("cpu")))

    # Move pymodel to the same device as the input data
    pymodel = pymodel.to(config["device"])
    print("y_pred", pymodel(dataset[0].to(config["device"])))

    train_loader, val_loader = train_val_split(dataset, config=config)
    test_path = config["test_dataset_path"]
    test_dataset = torch.load(test_path, map_location=config["device"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    # %matplotlib inline
    import matplotlib.pyplot as plt

    print("pymodel device", pymodel.device)
    fig, axis = plt.subplots(1, 3, figsize=(15, 5))
    for id, loader in enumerate([train_loader, val_loader, test_loader]):
        axis[id].set_ylabel("y_pred")
        axis[id].set_xlabel("y_true")

        for x in loader:
            with torch.no_grad():
                Y_pred = pymodel(x.to(config["device"]))
            break
        axis[id].scatter(
            x.y.to("cpu"),
            Y_pred.to("cpu").detach(),
        )
        axis[id].plot(x.y.to("cpu"), x.y.to("cpu"))
        axis[id].set_title(["train set", "validation set", "test set"][id])
        # add R2 score and MSE and MAE
        from sklearn.metrics import (
            mean_squared_error,
            mean_absolute_error,
            r2_score,
        )

        score = r2_score(x.y.to("cpu"), Y_pred.to("cpu").detach())
        axis[id].text(
            0.05, 0.95, f"R2 score: {score:.2f}", transform=axis[id].transAxes
        )
        score = mean_squared_error(x.y.to("cpu"), Y_pred.to("cpu").detach())
        axis[id].text(
            0.05, 0.90, f"MSE: {score:.2f}", transform=axis[id].transAxes
        )
        score = mean_absolute_error(x.y.to("cpu"), Y_pred.to("cpu").detach())
        axis[id].text(
            0.05, 0.85, f"MAE: {score:.2f}", transform=axis[id].transAxes
        )

    plt.show()
    plt.savefig("training_results.png")

    return train_loader


def plot_training_results_frag(
    chkpt_path, chkpt_path_frag, config_dir, plot_test=True
):
    import pymongo
    import stk
    from geom3d.oligomer_encoding_with_transformer import Fragment_encoder

    config = read_config(config_dir)
    # to get the try and except start indent here
    checkpoint = torch.load(chkpt_path)
    model, graph_pred_linear = model_setup(config)
    print("Model loaded: ", config["model_name"])

    # Pass the model and graph_pred_linear to the Pymodel constructor
    pymodel = Pymodel(model, graph_pred_linear)

    # Load the state dictionary
    pymodel.load_state_dict(state_dict=checkpoint["state_dict"])

    # Set the model to evaluation mode
    pymodel.eval()
    pymodel = pymodel.to(config["device"])
    EncodingModel = Fragment_encoder(
        input_dim=config["emb_dim"] * config["number_of_fragement"],
        model_dim=config["emb_dim"],
        num_heads=1,
        num_classes=config["emb_dim"],
        num_layers=1,
        dropout=0.0,
        lr=5e-4,
        warmup=50,
        max_iters=config["max_epochs"],
    )

    EncodingModel.add_encoder(model)
    state_dict = torch.load(chkpt_path_frag, map_location=torch.device(config["device"]))
    EncodingModel.load_state_dict(state_dict["state_dict"])
    EncodingModel.eval()
    EncodingModel = EncodingModel.to(config["device"])
    dataset, model = load_data_frag(config)
    dataset_org = load_data(config)
    
    train_loader, val_loader = train_val_split(dataset, config=config)
    if plot_test:
        test_path = config["test_dataset_path"]
        from pathlib import Path

        if Path(config["test_dataset_frag_path"]).exists():
            test_dataset_frag = torch.load(
                config["test_dataset_frag_path"], map_location=config["device"]
            )
            test_loader = DataLoader(
                test_dataset_frag,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=config["num_workers"],
            )
        elif Path(test_path).exists():
            print("generating test dataset")
            test_dataset = torch.load(test_path, map_location=config["device"])
            client = pymongo.MongoClient(config["pymongo_client"])
            db = stk.ConstructedMoleculeMongoDb(
                client,
                database=config["database_name"],
            )
            test_dataset_frag = generate_dataset_frag_dataset(
                test_dataset,
                model,
                db,
                number_of_molecules=len(test_dataset),
                number_of_fragement=6,
                device="cuda",
            )
            torch.save(
                test_dataset_frag,
                config["test_dataset_frag_path"],
            )
            test_loader = DataLoader(
                test_dataset_frag,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=config["num_workers"],
            )
        else:
            raise ValueError("No test dataset found")
        list_loader = [train_loader, val_loader, test_loader]
    else:
        list_loader = [train_loader, val_loader]

    # %matplotlib inline
    import matplotlib.pyplot as plt

    print("pymodel device", pymodel.device)
    fig, axis = plt.subplots(1, 3, figsize=(15, 5))
    for id, loader in enumerate(list_loader):
        axis[id].set_ylabel("y_pred")
        axis[id].set_xlabel("y_true")

        for x in loader:
            with torch.no_grad():
                representation = EncodingModel(x)
                representation = representation.squeeze()
                #representation = representation.transpose(0, 1)
                print("representation", representation.shape)
                Y_pred = pymodel.graph_pred_linear(representation.to(config["device"]))
                #add y_pred from org representation
                print("x[0].y", x[0].y.shape)
                y_pred_org = pymodel.graph_pred_linear(x[0].y.to(config["device"]))
                print("y_pred_org", y_pred_org.shape)
            break
        y_true = []
        for elm in x[0]["InChIKey"]:
            for data_org in dataset_org:
                if elm == data_org["InChIKey"]:
                    y_true.append(data_org["y"].to("cpu"))
                    break
        print("y_true", len(y_true))
        print("y_pred", Y_pred.to("cpu").detach().shape)
        axis[id].scatter(
            y_true,
            Y_pred.to("cpu").detach(),
        )
        axis[id].scatter(
            y_true,
            y_pred_org.to("cpu").detach(),
        )
        axis[id].plot(y_true, y_true)
        axis[id].set_title(["train set", "validation set", "test set"][id])
        # add R2 score and MSE and MAE
        from sklearn.metrics import (
            mean_squared_error,
            mean_absolute_error,
            r2_score,
        )
        try:
            score = r2_score(y_true, Y_pred.squeeze().to("cpu").detach())
            axis[id].text(
                0.05, 0.95, f"R2 score: {score:.2f}", transform=axis[id].transAxes
            )
            score = mean_squared_error(y_true, Y_pred.squeeze().to("cpu").detach())
            axis[id].text(
                0.05, 0.90, f"MSE: {score:.2f}", transform=axis[id].transAxes
            )
            score = mean_absolute_error(y_true, Y_pred.squeeze().to("cpu").detach())
            axis[id].text(
                0.05, 0.85, f"MAE: {score:.2f}", transform=axis[id].transAxes
            )
        except ValueError as e:
            print(e)
            print("ValueError")
            pass

    plt.show()
    plt.savefig("training_results.png")

    return train_loader


if __name__ == "__main__":
    from argparse import ArgumentParser

    root = os.getcwd()
    chkpt_path = ""
    config_dir = ""
    train_loader = plot_training_results(chkpt_path, config_dir)
