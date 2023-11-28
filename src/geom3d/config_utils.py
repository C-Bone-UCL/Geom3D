import json
import os


def read_config(dir):
    if os.path.exists(dir + "/config.json"):
        config = load_config(dir)
        print("config loaded from", dir)
    else:
# Set parameters
        config = dict()
        config["seed"] = 42
        config["save_dataset"] = False
        config["name"] = "SchNet_target_1K_TEST_5e4lr"
        config["pymongo_client"] = "mongodb://129.31.66.201/"
        config["database_name"] = "stk_mohammed_BO"
        config["STK_path"] = "/home/ma11115/STK_search/"  # "/rds/general/user/ma11115/home/STK_Search/STK_search"
        config["running_dir"] = "/home/ma11115/Geom3D/training"  # "/rds/general/user/ma11115/home/Geom3D/Geom3D/training"
        config["batch_size"] = 128
        config["df_total"] = "df_total_new2023_08_20.csv"
        config["df_precursor"] = "calculation_data_precursor_310823_clean.pkl"
        config["num_molecules"] = 100
        config["num_workers"] = 0
        config["num_tasks"] = 1
        config["emb_dim"] = 128
        config["model"] = dict()
        config["model"]["node_class"] = 119
        config["model"]["edge_class"] = 5
        config["model"]["num_tasks"] = 1
        config["model"]["emb_dim"] = 128
        config["model"]["SchNet_num_filters"] = 128
        config["model"]["SchNet_num_interactions"] = 6
        config["model"]["SchNet_num_gaussians"] = 51
        config["model"]["SchNet_cutoff"] = 10
        config["model"]["SchNet_readout"] = "mean"
        config["max_epochs"] = 3
        config['train_ratio'] = 0.8
        config['valid_ratio'] = 0.1
        config['number_of_fragement'] = 6
        config["model_embedding_chkpt"] = ""
        config["model_VAE_chkpt"] = ""
        config["load_dataset"] = False
        config["dataset_path"] = ""
        config["dataset_path_frag"] = ""
        save_config(config, dir)
        print("config saved to", dir)

    return config


def save_config(config, dir):
    os.makedirs(dir, exist_ok=True)
    #save config to json
    with open(dir + "/config.json", "w") as f:
        json.dump(config, f, indent=4,
            separators=(',', ': '), sort_keys=True)
    return None


def load_config(dir):
    #load config from json
    with open(dir + "/config.json", "r") as f:
        config = json.load(f)
    return config

if __name__ == "__main__":
    from argparse import ArgumentParser
    root = os.getcwd()
    argparser = ArgumentParser()
    argparser.add_argument(
        "--dir",
        type=str,
        default="",
        help="directory to config.json",
    )
    args = argparser.parse_args()
    dir = root + args.dir
    read_config(dir=dir)
