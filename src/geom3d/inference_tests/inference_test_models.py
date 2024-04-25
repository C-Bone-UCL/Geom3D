"""
Script to test the how the model performs (model inference) on a single batch of data
Loads a trained PyTorch model checkpoint and plots the predicted values against the true values for three different datasets:
training set, validation set, and test set
"""
import os
import torch
import copy
from geom3d import train_models
from geom3d.train_models import SchNet, DimeNet, DimeNetPlusPlus, GemNet, SphereNet, SphereNetPeriodic, PaiNN, Pymodel
from geom3d.train_models import read_config, load_data, train_val_test_split, model_setup
import importlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

importlib.reload(train_models)

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
    checkpoint = torch.load(chkpt_path)
    
    model, graph_pred_linear = model_setup(config)
    print("Model loaded: ", config["model_name"])
    
    # Pass the model and graph_pred_linear to the Pymodel constructor
    pymodel = Pymodel(model, graph_pred_linear, config)

    # Load the state dictionary
    pymodel.load_state_dict(state_dict=checkpoint['state_dict'])
    
    # Set the model to evaluation mode
    pymodel.eval()

    # end indent here
    dataset = load_data(config)
    np.random.seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    print('y_true', dataset[0].y)

    # removed here because these bade the code error
    # pymodel_cpu = copy.deepcopy(pymodel).to('cpu')
    print(dataset[0].to('cpu').y)
    # print('y_pred_cpu', pymodel_cpu(dataset[0].to('cpu')))

    # Move pymodel to the same device as the input data
    pymodel = pymodel.to(config["device"])
    
    # removed code here because it made the code error
    # print('y_pred', pymodel(dataset[0].to(config["device"])))

    train_loader, val_loader, test_loader = train_val_test_split(
        dataset, config=config, batch_size=config["batch_size"]
    )
    
    #%matplotlib inline

    print("pymodel device", pymodel.device)
    print("Target: ", config["target_name"])

    # Get the y values from the dataset for setting plot axes
    y_values = [data.y for data in dataset]
    y_min = min(y_values)
    y_max = max(y_values)


    fig, axis = plt.subplots(1, 3, figsize=(15, 5))
    for id, loader in enumerate([train_loader, val_loader, test_loader]):
        axis[id].set_ylabel('y_pred')
        axis[id].set_xlabel('y_true')
        axis[id].set_xlim(y_min, y_max)  # Set x-axis limits based on min and max y values
        axis[id].set_ylim(y_min, y_max)  # Set y-axis limits based on min and max y values

        for x in loader:
            with torch.no_grad():
                Y_pred = pymodel(x.to(config["device"]))
            break
        axis[id].scatter(x.y.to('cpu'), Y_pred.to('cpu').detach())
        axis[id].plot(x.y.to('cpu'), x.y.to('cpu'))
        axis[id].set_title(['train set', 'validation set', 'test set'][id])
    plt.show()

    # calculate the mean absolute error
    y_true = []
    y_pred = []
    for x in test_loader:
        with torch.no_grad():
            Y_pred = pymodel(x.to(config["device"]))
        y_true.append(x.y.to('cpu'))
        y_pred.append(Y_pred.to('cpu').detach())
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    print('Mean Absolute Error (MAE) on test_set:', mae)

    # calculate the mean squared error
    mse = mean_squared_error(y_true, y_pred)
    print('Mean Squared Error (MSE) on test_set:', mse)
    
    return train_loader, mae, mse

if __name__ == "__main__":
    from argparse import ArgumentParser
    root = os.getcwd()
    chkpt_path = ""
    config_dir = ""
    train_loader = plot_training_results(chkpt_path, config_dir)