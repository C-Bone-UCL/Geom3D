""" 
script to turn a a dataset into custom target based split that puts the 10% best performing molecules in the test set 
"""

import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display, HTML
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw, rdFMCS
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
from tqdm import tqdm

from geom3d.utils import database_utils
from geom3d.utils.oligomer_scaffold_split import load_dataframes, check_data_exists

def top_target_splitter(dataset, config):
    """
    Split a dataset into a training and test set based on the target value.
    The 10% best performing molecules are put in the test set.

    Args:
    - dataset (list): list of dictionaries containing the data
    - config (dict): dictionary containing the configuration

    Returns:
    - test_set_inchikeys (list): list of InChIKeys of the test set

    """
    
    if config["target_name"] == "combined":
        ideal_value = 0
    elif config["target_name"] == "IP":
        ideal_value = 5.5
    elif config["target_name"] == "ES1":
        ideal_value = 3
    elif config["target_name"] == "fosc1":
        ideal_value = 10
    else:
        raise ValueError(f"Unknown target_name: {config['target_name']}")

    print(f"splitting group {config['test_set_target_cluster']} 10% of dataset into equal val and test set. The target value is {config['target_name']}.")

    # same logic but look at how far the value is from 0 and take the 10% closest to 10
    target = np.array([data['y'] for data in dataset])
    distance = np.abs(target - ideal_value)
    sorted_indices = np.argsort(distance)

    chosen_set = sorted_indices[int(len(sorted_indices) * 0.1 * (config["test_set_target_cluster"] - 1)):int(len(sorted_indices) * 0.1 * config["test_set_target_cluster"])]

    np.random.shuffle(chosen_set)

    test_set = [dataset[i] for i in chosen_set]

    test_set_inchikeys = [data["InChIKey"] for data in test_set]
    
    return test_set_inchikeys

def target_plot(dataset, config):
    """Plot the 2D PCA space of the dataset, highlighting the top 10% of oligomers wrt target value.

    Args:
    - dataset (list): list of dictionaries containing the data
    - config (dict): dictionary containing the configuration

    """
    
    df_total, df_precursors, df_path, df_path_2, df_precursors_path = load_dataframes(dataset, config)
    check_data_exists(df_total, dataset, config)
    test_set_inchikeys = top_target_splitter(dataset, config)

    # Plot all clusters
    plt.figure(figsize=(10, 10))
    plt.scatter(df_total['2d_tani_pca_1'], df_total['2d_tani_pca_2'], alpha=0.7, label='All oligomers', c='lightgrey')
    # Highlight the samples where df_total[InChIKey] is in test_set_inchikeys
    plt.scatter(df_total.loc[df_total['InChIKey'].isin(test_set_inchikeys), '2d_tani_pca_1'], df_total.loc[df_total['InChIKey'].isin(test_set_inchikeys), '2d_tani_pca_2'], c='red', label='Test set', alpha=0.9)

    plt.legend()
    plt.title("Top 10% of oligomers wrt target value in 2D PCA space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

    return

def substructure_analysis_top_target(dataset, config):
    """
    Generate common substructures for the top 10% of oligomers wrt target value.
    
    Args:
    - dataset (list): list of dictionaries containing the data
    - config (dict): dictionary containing the configuration

    """
    
    df_total, df_precursors, df_path, df_path_2, df_precursors_path = load_dataframes(dataset, config)
    
    X_frag_mol = df_precursors['mol_opt'].values
    X_frag_inch = df_precursors['InChIKey'].values
    keys_6mer = df_total['InChIKey'].values
    
    selected_cluster_keys = top_target_splitter(dataset, config)
    print('target_name:', config['target_name'])
    # Generate common substructures for each molecule in the cluster
    common_substructures = []
    counter = 0

    # Loop through the oligomers in the cluster
    for oligomer_key in tqdm(selected_cluster_keys, desc=f"Generating substructures for top 10% wrt target"):
        # Extract InChIKeys from columns InChIKeys_0 to InChIKeys_5
        inchikeys = [df_total.loc[df_total['InChIKey'] == oligomer_key, f'InChIKey_{i}'].values[0] for i in range(6)]

        # Get the RDKit molecules for the corresponding InChIKeys
        fragments = [X_frag_mol[X_frag_inch == inchikey][0] for inchikey in inchikeys if inchikey in X_frag_inch]

        # Combine the individual fragments into a single molecule, stepwise because can only take 2 rdkit molecules at a time
        combined_molecule = Chem.CombineMols(fragments[0], fragments[1])
        for i in range(2, len(fragments)):
            combined_molecule = Chem.CombineMols(combined_molecule, fragments[i])

        # Convert the combined oligomer molecule to SMILES
        oligomer_smiles = Chem.MolToSmiles(combined_molecule)

        # Check if there's only one molecule in the cluster
        if len(selected_cluster_keys) < 2:
            print(f"Oligomer {oligomer_key}: Not enough fragments for comparison.")
        else:
            # Find the common substructure in the combined oligomer
            common_substructure = rdFMCS.FindMCS([combined_molecule, combined_molecule])
            common_substructure = Chem.MolFromSmarts(common_substructure.smartsString)
            common_substructures.append(common_substructure)


        #visualise only one combined molecule in the cluster in 2D, so its easier to see
        if len(fragments) == 6 and counter == 0:
            print(f'representative oligomer in top 10%')
            mol = Chem.MolFromSmiles(oligomer_smiles)
            img = Draw.MolToImage(mol)
            display(img)
            counter += 1


    # Count the occurrences of each substructure
    substructure_counts = Counter([Chem.MolToSmarts(sub) for sub in common_substructures])

    # Rank substructures based on frequency
    ranked_substructures = sorted(substructure_counts.items(), key=lambda x: x[1], reverse=True)

    # Display the top N substructures and their occurrences
    top_n = min(3, len(ranked_substructures))  # Choose the smaller of 3 and the actual number of substructures
    for i, (substructure, count) in enumerate(ranked_substructures[:top_n]):
        print(f"Top {i + 1} Substructure (Frequency: {count} oligomers):")
        img = Draw.MolToImage(Chem.MolFromSmarts(substructure))
        display(img)