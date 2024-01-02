"""
Script to test the how the model performs (model inference) on a single batch of data
Loads a trained PyTorch model checkpoint and plots the predicted values against the true values for three different datasets:
training set, validation set, and test set
"""
import os
import torch
import copy
from geom3d.train_SchNet import SchNet, Pymodel, read_config, load_data, train_val_test_split


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

    print('checkpoint used:', chkpt_path)

    # try:
    #     pymodel = Pymodel.load_from_checkpoint(chkpt_path)
    #     pymodel.freeze()
    # except (TypeError, KeyError):

    # to get the try and except start indent here
    model_config = config["model"]
    checkpoint = torch.load(chkpt_path)

    model = SchNet(
        hidden_channels=model_config["emb_dim"],
        num_filters=model_config["SchNet_num_filters"],
        num_interactions=model_config["SchNet_num_interactions"],
        num_gaussians=model_config["SchNet_num_gaussians"],
        cutoff=model_config["SchNet_cutoff"],
        readout=model_config["SchNet_readout"],
        node_class=model_config["node_class"],
    )

    graph_pred_linear = torch.nn.Linear(
        model_config["emb_dim"], model_config["num_tasks"]
    )

    # Pass the model and graph_pred_linear to the Pymodel constructor
    pymodel = Pymodel(model, graph_pred_linear)

    # Load the state dictionary
    pymodel.load_state_dict(state_dict=checkpoint['state_dict'])

    # Set the model to evaluation mode
    pymodel.eval()

    # end indent here

    dataset = load_data(config)
    np.random.seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    print('y_true', dataset[0].y)
    pymodel_cpu = copy.deepcopy(pymodel).to('cpu')
    print('y_pred_cpu', pymodel_cpu(dataset[0].to('cpu')))

    # Move pymodel to the same device as the input data
    pymodel = pymodel.to(config["device"])
    print('y_pred', pymodel(dataset[0].to(config["device"])))

    train_loader, val_loader, test_loader = train_val_test_split(
        dataset, config=config
    )
    #%matplotlib inline
    import matplotlib.pyplot as plt
    print("pymodel device", pymodel.device)
    fig, axis = plt.subplots(1,3, figsize=(15,5))
    for id,loader in enumerate([train_loader, val_loader, test_loader]):
        axis[id].set_ylabel('y_pred')
        axis[id].set_xlabel('y_true')
        
        for x in loader:
            with torch.no_grad():
                Y_pred = pymodel(x.to(config["device"]))
            break
        axis[id].scatter( x.y.to('cpu'),Y_pred.to('cpu').detach(),)
        axis[id].plot(x.y.to('cpu'),x.y.to('cpu'))
        axis[id].set_title(['train set', 'validation set', 'test set'][id])
    plt.show()
    plt.savefig('training_results.png')
    
    return train_loader

if __name__ == "__main__":
    from argparse import ArgumentParser
    root = os.getcwd()
    chkpt_path = ""
    config_dir = ""
    train_loader = plot_training_results(chkpt_path, config_dir)