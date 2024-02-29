"""
Script to run models with different parameters

For the different splits, perform the analysis in a jupyter notbook using the functions in each script
Then tune the parameters for the best performance (cluster threshold, min_cluster_size, min_samples, etc.)
Then need to chose a cluster to put in the test set, and then run the training with the chosen parameters

Perform hyperparameter search for the best performance by setting hp_search to True
Can set the number of trials and the maximum number of epochs for the hyperparameter search

Currently implemented parameters:
model_name: SchNet, DimeNet, DimeNetPlusPlus, SphereNet, PaiNN, Equiformer
target_name: combined, IP, ES1, fosc1
lr_scheduler: CosineAnnealingLR, CosineAnnealingWarmRestartsm, StepLR
split: random, fragment_scaffold, oligomer_scaffold, target_cluster

Example usage:
python run_training.py --model_name SchNet --num_molecules 500 --target_name combined 
"""

from geom3d.utils.config_utils import save_config, read_config
import argparse
from geom3d import train_models
import os
import json
import importlib

importlib.reload(train_models)


def run_training(
    model_name,
    num_molecules,
    target_name,
    lr_scheduler,
    batch_size,
    max_epochs,
    split,
    hp_search,
    max_epochs_hp_search,
    n_trials_hp_search, 
    test_set_fragment_cluster,
    test_set_oligomer_cluster,
    test_set_target_cluster,
    running_dir="/rds/general/user/cb1319/home/GEOM3D/Geom3D/fragment_experiment",
    dataset_folder="/rds/general/user/cb1319/home/GEOM3D/Geom3D/datasets"
):
    """
    Run the training with the given hyperparameters
    """
    config_dir = os.path.join(running_dir, f"{model_name}_opt_{target_name}_{num_molecules}_{test_set_fragment_cluster}")

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    config = read_config(config_dir)
    config["STK_path"] = "/rds/general/user/cb1319/home/GEOM3D/STK_path/"
    config["df_precursor"] = "calculation_data_precursor_071123_clean.pkl"
    config["df_total"] = "df_total_subset_16_11_23.csv"
    config["max_epochs"] = max_epochs
    config["load_dataset"] = True
    config["model_name"] = model_name
    config["dataset_path_frag"] = ""
    config["dataset_folder"] = dataset_folder
    config["save_dataset"] = True
    config["num_molecules"] = num_molecules
    config["running_dir"] = running_dir
    config["train_ratio"] = 0.8
    config["name"] = f"{model_name}_opt_{target_name}_{num_molecules}_{test_set_fragment_cluster}"
    config["target_name"] = target_name
    config["batch_size"] = batch_size

    config["lr_scheduler"] = lr_scheduler

    config["fragment_cluster_threshold"] = 0.067
    config["test_set_fragment_cluster"] = test_set_fragment_cluster

    config["oligomer_min_cluster_size"] = 750
    config["oligomer_min_samples"] = 50
    config["test_set_oligomer_cluster"] = test_set_oligomer_cluster

    config["test_set_target_cluster"] = test_set_target_cluster

    config["mixed_precision"] = True

    config["split"] = split

    config["hp_search"] = hp_search
    config["max_epochs_hp_search"] = max_epochs_hp_search
    config["n_trials_hp_search"] = n_trials_hp_search


    if model_name == 'PaiNN':
        config["dataset_path"] = f"{dataset_folder}/{str(num_molecules)}dataset_radius_{target_name}.pt"
    else:
        config["dataset_path"] = f"{dataset_folder}/{str(num_molecules)}dataset_{target_name}.pt"
    
    if config["model_name"] == "SchNet":
        config["model"] = dict()
        config["model"]["node_class"] = 119
        config["model"]["edge_class"] = 5
        config["model"]["num_tasks"] = 1
        config["model"]["emb_dim"] = 176
        config["model"]["SchNet_num_filters"] = 128
        config["model"]["SchNet_num_interactions"] = 8
        config["model"]["SchNet_num_gaussians"] = 51
        config["model"]["SchNet_cutoff"] = 6
        config["model"]["SchNet_readout"] = "mean"

    elif config["model_name"] == "DimeNet":
        config["model"] = dict()
        config["model"]["node_class"] = 119
        config["model"]["hidden_channels"] = 128
        config["model"]["out_channels"] = 1
        config["model"]["num_blocks"] = 6
        config["model"]["num_bilinear"] = 8
        config["model"]["num_spherical"] = 7
        config["model"]["num_radial"] = 6
        config["model"]["cutoff"] = 4.0
        config["model"]["envelope_exponent"] = 5
        config["model"]["num_before_skip"] = 1
        config["model"]["num_after_skip"] = 2
        config["model"]["num_output_layers"] = 3

    elif config["model_name"] == "DimeNetPlusPlus":
        config["model"] = dict()
        config["model"]["node_class"] = 119
        config["model"]["hidden_channels"] = 32
        config["model"]["out_channels"] = 1
        config["model"]["num_blocks"] = 9
        config["model"]["int_emb_size"] = 64
        config["model"]["basis_emb_size"] = 8
        config["model"]["out_emb_channels"] = 64
        config["model"]["num_spherical"] = 7
        config["model"]["num_radial"] = 6
        config["model"]["cutoff"] = 6.0
        config["model"]["envelope_exponent"] = 5
        config["model"]["num_before_skip"] = 1
        config["model"]["num_after_skip"] = 2
        config["model"]["num_output_layers"] = 3

    elif config["model_name"] == "GemNet":
        config["model"] = dict()
        config["model"]["node_class"] = 119
        config["model"]["num_spherical"] = 7
        config["model"]["num_radial"] = 6
        config["model"]["num_blocks"] = 4
        config["model"]["emb_size_atom"] = 64
        config["model"]["emb_size_edge"] = 64
        config["model"]["emb_size_trip"] = 64
        config["model"]["emb_size_quad"] = 32
        config["model"]["emb_size_rbf"] = 16
        config["model"]["emb_size_cbf"] = 16
        config["model"]["emb_size_sbf"] = 32
        config["model"]["emb_size_bil_quad"] = 32
        config["model"]["emb_size_bil_trip"] = 64
        config["model"]["num_before_skip"] = 1
        config["model"]["num_after_skip"] = 1
        config["model"]["num_concat"] = 1
        config["model"]["num_atom"] = 2
        config["model"]["cutoff"] = 5.0
        config["model"]["int_cutoff"] = 10.0
        config["model"]["triplets_only"] = 1
        config["model"]["direct_forces"] = 0
        config["model"]["envelope_exponent"] = 5
        config["model"]["extensive"] = 1
        config["model"]["forces_coupled"] = 0
        config["model"]["num_targets"] = 1

    elif config["model_name"] == "Equiformer":
        config["model"] = dict()
        config["model"]["Equiformer_radius"] = 7.0
        config["model"]["Equiformer_irreps_in"] = "5x0e"
        config["model"]["Equiformer_num_basis"] = 64
        config["model"]["Equiformer_hyperparameter"] = 0
        config["model"]["Equiformer_num_layers"] = 3
        config["model"]["node_class"] = 64
        config["model"]["irreps_node_embedding"] = "64x0e+16x1e+16x2e"

    elif config["model_name"] == "PaiNN":
        config["model"] = dict()
        config["model"]["n_atom_basis"] = 64
        config["model"]["n_interactions"] = 6
        config["model"]["n_rbf"] = 20
        config["model"]["cutoff"] = 4.0
        config["model"]["max_z"] = 93
        config["model"]["n_out"] = 1
        config["model"]["readout"] = "add"

    elif config["model_name"] == "SphereNet":
        config["model"] = dict()
        config["model"]["hidden_channels"] = 128
        config["model"]["out_channels"] = 1
        config["model"]["cutoff"] = 6.0
        config["model"]["num_layers"] = 5
        config["model"]["int_emb_size"] = 64
        config["model"]["basis_emb_size_dist"] = 8
        config["model"]["basis_emb_size_angle"] = 8
        config["model"]["basis_emb_size_torsion"] = 8
        config["model"]["out_emb_channels"] = 256
        config["model"]["num_spherical"] = 3
        config["model"]["num_radial"] = 6
        config["model"]["envelope_exponent"] = 5
        config["model"]["num_before_skip"] = 1
        config["model"]["num_after_skip"] = 2
        config["model"]["num_output_layers"] = 3

    save_config(config, config_dir)
    print("config saved at ", config_dir)

    # train_hyperparam_search.main(config_dir)
    train_models.main(config_dir)

# can add diff lr here later
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="SphereNet")
    parser.add_argument("--num_molecules", type=int, default=500)
    parser.add_argument("--target_name", type=str, default="combined")
    parser.add_argument("--lr_scheduler", type=str, default="CosineAnnealingLR")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--split", type=str, default="random")
    parser.add_argument("--hp_search", type=bool, default=False)
    parser.add_argument("--max_epochs_hp_search", type=int, default=5)
    parser.add_argument("--n_trials_hp_search", type=int, default=5)
    parser.add_argument("--test_set_fragment_cluster", type=int, default=1)
    parser.add_argument("--test_set_oligomer_cluster", type=int, default=1)
    parser.add_argument("--test_set_target_cluster", type=int, default=1)
    
    args = parser.parse_args()
    run_training(
        args.model_name, 
        args.num_molecules, 
        args.target_name, 
        args.lr_scheduler, 
        args.batch_size, 
        args.max_epochs, 
        args.split, 
        args.hp_search, 
        args.max_epochs_hp_search, 
        args.n_trials_hp_search,
        args.test_set_fragment_cluster,
        args.test_set_oligomer_cluster,
        args.test_set_target_cluster
        )