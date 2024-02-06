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
        config["target_name"] = "combined"
        config["model_name"] = "SchNet"
        config["mixed_precision"] = False

        config["fragment_cluster_threshold"] = 0.55
        config["test_set_fragment_cluster"] = 6

        config["oligomer_min_cluster_size"] = 750,
        config["oligomer_min_samples"] = 50,
        config["test_set_oligomer_cluster"] = 6

        config["lr"] = 5e-4
        config["lr_scale"] = 1
        config["decay"] = 0
        config["lr_scheduler"] = "CosineAnnealingLR"
        config["lr_decay_factor"] = 0.5
        config["lr_decay_step_size"] = 100
        config["lr_decay_patience"] = 50
        config["min_lr"] = 1e-6

        config["split"] = "random"

        config["hp_search"] = False
        config["max_epochs_hp_search"] = 1
        config["n_trials_hp_search"] = 1

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
