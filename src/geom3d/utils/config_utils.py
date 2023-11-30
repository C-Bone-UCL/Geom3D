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
<<<<<<< Updated upstream
        config["database_name"] = "stk_cyprien_BO"
=======
        config["database_name"] = "stk_mohammed_BO"
>>>>>>> Stashed changes
        config["STK_path"] = "/rds/general/user/cb1319/home/GEOM3D/STK_path/"
        config["running_dir"] = "/rds/general/user/cb1319/home/GEOM3D/Geom3D/training/"
        config["batch_size"] = 128
        config["df_total"] = "df_total_subset_16_11_23.csv"
        config["df_precursor"] = "calculation_data_precursor_071123_clean.pkl"
        config["num_molecules"] = 100
        config["num_workers"] = 0
        config["num_tasks"] = 1
        config["emb_dim"] = 128
        config["max_epochs"] = 3
        config['train_ratio'] = 0.8
        config['valid_ratio'] = 0.1
        config['number_of_fragement'] = 6
        config["model_path"] = ""
        config["pl_model_chkpt"] = ""
        config["load_dataset"] = False
        config["dataset_path"] = ""
        config["dataset_path_frag"] = ""

<<<<<<< Updated upstream
        config["SchNet_model"] = dict()
        config["SchNet_model"]["node_class"] = 119
        config["SchNet_model"]["edge_class"] = 5
        config["SchNet_model"]["num_tasks"] = 1
        config["SchNet_model"]["emb_dim"] = 128
        config["SchNet_model"]["SchNet_num_filters"] = 128
        config["SchNet_model"]["SchNet_num_interactions"] = 6
        config["SchNet_model"]["SchNet_num_gaussians"] = 51
        config["SchNet_model"]["SchNet_cutoff"] = 10
        config["SchNet_model"]["SchNet_readout"] = "mean"

        config["DimeNet_model"] = dict()
        config["DimeNet_model"]["node_class"] = 119
        config["DimeNet_model"]["hidden_channels"] = 128
        config["DimeNet_model"]["out_channels"] = 1
        config["DimeNet_model"]["num_blocks"] = 6
        config["DimeNet_model"]["num_bilinear"] = 8
        config["DimeNet_model"]["num_spherical"] = 7
        config["DimeNet_model"]["num_radial"] = 6
        config["DimeNet_model"]["cutoff"] = 5.0
        config["DimeNet_model"]["envelope_exponent"] = 5
        config["DimeNet_model"]["num_before_skip"] = 1
        config["DimeNet_model"]["num_after_skip"] = 2
        config["DimeNet_model"]["num_output_layers"] = 3
        config["DimeNet_model"]["act"] = "swish"
=======
        # prompt the user to enter model name
        config["model_name"] = input("Enter model name: ") 

        if config["model_name"] == "SchNet":
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

        elif config["model_name"] == "DimeNet":
            config["model"] = dict()
            config["model"]["node_class"] = 119
            config["model"]["hidden_channels"] = 300
            config["model"]["out_channels"] = 1
            config["model"]["num_blocks"] = 6
            config["model"]["num_bilinear"] = 8
            config["model"]["num_spherical"] = 7
            config["model"]["num_radial"] = 6
            config["model"]["cutoff"] = 10.0
            config["model"]["envelope_exponent"] = 5
            config["model"]["num_before_skip"] = 1
            config["model"]["num_after_skip"] = 2
            config["model"]["num_output_layers"] = 3
>>>>>>> Stashed changes

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
