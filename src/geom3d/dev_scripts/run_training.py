from geom3d.utils.config_utils import save_config, read_config
import argparse
from geom3d import train_models


def run_training(
    model_name,
    num_molecules,
    max_epochs=5,
    running_dir="/rds/general/user/cb1319/home/Geom3D/Geom3D/training",
):
    config_dir = running_dir + f"/{model_name}_target_80K_TEST_5e4lr"
    config = read_config(config_dir)
    config["STK_path"] = "/rds/general/user/cb1319/home/STK_Search/STK_search"
    config["df_precursor"] = "calculation_data_precursor_190923_clean.pkl"
    config["df_total"] = "df_total_subset_16_11_23.csv"
    config["max_epochs"] = max_epochs
    config["load_dataset"] = True
    config[
        "dataset_path"
    ] = f"/rds/general/user/cb1319/home/GEOM3D/Geom3D/training/{model_name}_target_80K_TEST_5e4lr/80000dataset_radius.pt"
    config["model_name"] = model_name
    config["dataset_path_frag"] = ""
    config["save_dataset"] = True
    config["num_molecules"] = num_molecules
    config["running_dir"] = running_dir
    config["train_ratio"] = 0.8
    config["name"] = f"{model_name}_target_80K_TEST_5e4lr"
    save_config(config, config_dir)
    print("config saved")
    print(config)

# can add diff lr here later
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="SphereNet")
    parser.add_argument("--num_molecules", type=int, default=1000)
    parser.add_argument("--max_epochs", type=int, default=5)
    args = parser.parse_args()
    run_training(args.model_name, args.num_molecules, args.max_epochs)