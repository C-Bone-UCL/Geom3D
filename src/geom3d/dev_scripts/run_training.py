from geom3d.utils.config_utils import save_config, read_config
import argparse
from geom3d import train_models


def run_training(
    target_name,
    aim,
    num_molecules,
    max_epochs=5,
    running_dir="/rds/general/user/ma11115/home/Geom3D/Geom3D/training",
):
    config_dir = running_dir + f"/SchNet/{target_name}_{num_molecules}_{aim}/"
    config = read_config(config_dir, model_name="SchNet")
    config["STK_path"] = "/rds/general/user/ma11115/home/STK_Search/STK_search"
    config["df_precursor"] = "calculation_data_precursor_190923_clean.pkl"
    config["df_total"] = "df_total_subset_16_11_23.csv"
    config["max_epochs"] = max_epochs
    config["load_dataset"] = True
    config[
        "dataset_path"
    ] = f"/rds/general/user/ma11115/home/Geom3D/Geom3D/dataset/opt_geom3d/dataset_not_test_{target_name}_aim_{aim}.pt"
    config["dataset_path_frag"] = ""
    config["save_dataset"] = False
    config["num_molecules"] = num_molecules
    config["running_dir"] = running_dir + "/SchNet"
    config["train_ratio"] = 0.8
    config[
        "test_dataset_path"
    ] = f"/rds/general/user/ma11115/home/Geom3D/Geom3D/dataset/opt_geom3d/test_dataset_{target_name}_aim_{aim}.pt"
    config["save_dataset_frag"] = True
    config["name"] = f"{target_name}_{num_molecules}_{aim}/"
    config['target_name'] = target_name
    save_config(config, config_dir)
    train_models.main(config_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="fosc1")
    parser.add_argument("--aim", type=float, default=0)
    parser.add_argument("--num_molecules", type=int, default=1000)
    parser.add_argument("--max_epochs", type=int, default=5)
    args = parser.parse_args()
    run_training(args.target, args.aim, args.num_molecules, args.max_epochs)
