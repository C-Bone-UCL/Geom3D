import os
import torch
import copy
import argparse
from geom3d import train_models
from geom3d.train_models import Pymodel
from geom3d.train_models import read_config, load_data, train_val_test_split, model_setup
import importlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pandas as pd

def main(target, cluster_numbers, model_names):
    '''
    Function to run model inference on the test set for a given target and cluster number
    
    Args:
    - target (str): target name
    - cluster_numbers (list): list of cluster numbers
    - model_names (list): list of model names

    Returns:
    - None
    '''
    running_dir = "/rds/general/user/cb1319/home/GEOM3D/Geom3D/fragment_experiment_2"
    for cluster in cluster_numbers:
        for model_name in model_names:
            csv_path = running_dir + f"/{cluster}_cluster_{target}_model_inferences.csv"
            csv_path_2 = running_dir + f"/{cluster}_cluster_{target}_model_inferences_totalset.csv"

            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, index_col=0)
            else:
                df = pd.DataFrame()
            if os.path.exists(csv_path_2):
                df_2 = pd.read_csv(csv_path_2, index_col=0)
            else:
                df_2 = pd.DataFrame()
            config_dir = running_dir + f"{target}_experiment/{model_name}_opt_{target}_80000_{cluster}"
            for file in os.listdir(config_dir):
                if file.endswith(".ckpt"):
                    chkpt_path = os.path.join(config_dir, file)
                    print(chkpt_path)
                    break
            if os.path.exists(chkpt_path):
                y_true, y_pred, InChIKey, y_true_val, y_pred_val, InChIKey_val, y_true_train, y_pred_train, InChIKey_train = test_model(chkpt_path, config_dir, cluster)
                df_temp = pd.DataFrame({f'{model_name}_true_{target}': y_true, 
                                        f'{model_name}_pred_{target}': y_pred,
                                        'set': 'test'}, 
                                        index=InChIKey)
                df = pd.concat([df, df_temp], axis=1)

                if target == 'ES1':
                    print('skipping saving cluster files for est set in ES1')
                else:
                    df.to_csv(csv_path)
                    print(f'{model_name} predictions for {cluster} cluster saved')

                df_temp_train = pd.DataFrame({f'{model_name}_true_{target}': y_true_train, 
                                            f'{model_name}_pred_{target}': y_pred_train,
                                            'set': 'train'}, 
                                            index=InChIKey_train)

                df_temp_val = pd.DataFrame({f'{model_name}_true_{target}': y_true_val,
                                            f'{model_name}_pred_{target}': y_pred_val,
                                            'set': 'val'}, 
                                            index=InChIKey_val)
                
                df_2_temp = pd.concat([df_temp_train, df_temp_val, df_temp], axis=0)
                df_2 = pd.concat([df_2, df_2_temp], axis=1)
                print('saving df')
                df_2.to_csv(csv_path_2)
                print(f'{model_name} predictions for {cluster} cluster total saved')
            else:
                print(f'Checkpoint file {chkpt_path} does not exist')
    return

def test_model(chkpt_path, config_dir, cluster):
    config = read_config(config_dir)
    np.random.seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.manual_seed(config["seed"])
    config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    config['test_set_fragment_cluster'] = cluster


    print('checkpoint used:', chkpt_path)
    checkpoint = torch.load(chkpt_path)
    model, graph_pred_linear = model_setup(config)
    print("Model loaded: ", config["model_name"])

    # Pass the model and graph_pred_linear to the Pymodel constructor
    pymodel = Pymodel(model, graph_pred_linear, config)
    # Load the state dictionary
    pymodel.load_state_dict(state_dict=checkpoint['state_dict'])
    # Set the model to evaluation mode
    pymodel.eval()

    dataset = load_data(config)
    np.random.seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    pymodel = pymodel.to(config["device"])

    train_loader, val_loader, test_loader = train_val_test_split(
        dataset, config=config, batch_size=config["batch_size"]
    )

    print("pymodel device", pymodel.device)
    print("Target: ", config["target_name"])

    y_true = []
    y_pred = []
    InChIKey = []

    y_true_val = []
    y_pred_val = []
    InChIKey_val = []

    y_true_train = []
    y_pred_train = []
    InChIKey_train = []

    for x in test_loader:
        with torch.no_grad():
            Y_pred = pymodel(x.to(config["device"]))
        y_true.append(x.y.to('cpu').numpy())
        y_pred.append(Y_pred.to('cpu').detach().numpy().flatten())
        # retrieve InChIKey of the test set
        InChIKey.append(x.InChIKey)
    # Flatten the lists
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    InChIKey = [item for sublist in InChIKey for item in sublist]

    for x in val_loader:
        with torch.no_grad():
            Y_pred = pymodel(x.to(config["device"]))
        y_true_val.append(x.y.to('cpu').numpy())
        y_pred_val.append(Y_pred.to('cpu').detach().numpy().flatten())
        # retrieve InChIKey of the validation set
        InChIKey_val.append(x.InChIKey)
    # flatten the lists
    y_true_val = np.concatenate(y_true_val)
    y_pred_val = np.concatenate(y_pred_val)
    InChIKey_val = [item for sublist in InChIKey_val for item in sublist]

    for x in train_loader:
        with torch.no_grad():
            Y_pred = pymodel(x.to(config["device"]))
        y_true_train.append(x.y.to('cpu').numpy())
        y_pred_train.append(Y_pred.to('cpu').detach().numpy().flatten())
        # retrieve InChIKey of the training set
        InChIKey_train.append(x.InChIKey)

    # flatten the lists
    y_true_train = np.concatenate(y_true_train)
    y_pred_train = np.concatenate(y_pred_train)
    InChIKey_train = [item for sublist in InChIKey_train for item in sublist]

    return y_true, y_pred, InChIKey, y_true_val, y_pred_val, InChIKey_val, y_true_train, y_pred_train, InChIKey_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference on HPC.")
    parser.add_argument("--target", type=str, help="Target name")
    parser.add_argument("--cluster_numbers", nargs="+", type=int, help="List of cluster numbers")
    parser.add_argument("--model_names", nargs="+", type=str, help="List of model names")
    args = parser.parse_args()

    main(args.target, args.cluster_numbers, args.model_names)

