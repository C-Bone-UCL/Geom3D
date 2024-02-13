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

def cluster_target_splitter(dataset, config):
    """
    Split a dataset into a training and test set based on the target value.
    The 10% best performing molecules are put in the test set.
    """
    print("top 10% in the test set. The target value is the combined score.")
    # Get the target column 'y' from each dictionary in the list
    target = np.array([data['y'] for data in dataset])

    # Cluster the target values using HDBSCAN
    clusterer = HDBSCAN(min_cluster_size=5)
    clusters = clusterer.fit_predict(target.reshape(-1, 1))

    # Choose one cluster to put in the test set (e.g., the cluster with the highest number of samples)
    test_cluster = config["test_set_target_cluster"]
    print(f"Test set cluster: {test_cluster}")

    # Get the indices of the molecules in the chosen cluster
    test_indices = np.where(clusters == test_cluster)[0]

    # find the InChIKeys of the test set
    test_set = [dataset[i] for i in test_indices]
    test_set_inchikeys = [data["InChIKey"] for data in test_set]

    return test_set_inchikeys

# function to visualize the clusters showing the assigned number for each cluster, the axes should both be the target value
def visualize_clusters(dataset, config):
    """
    Visualize the clusters of the target values.
    """
    # Get the target column 'y' from each dictionary in the list
    target = np.array([data['y'] for data in dataset])

    # Cluster the target values using HDBSCAN
    clusterer = HDBSCAN(min_cluster_size=5)
    clusters = clusterer.fit_predict(target.reshape(-1, 1))

    # Plot the clusters with reduced number of pixels
    plt.figure(figsize=(8, 6))
    plt.scatter(target, target, c=clusters, cmap="viridis", hue=clusters, s=10)
    plt.xlabel("Target value")
    plt.ylabel("target")
    plt.title("Clusters of target values")
    plt.show()

    return None

def substructure_analysis_top_target(dataset, config):
    df_path = Path(
        config["STK_path"], "data/output/Full_dataset/", config["df_total"]
    )
    df_precursors_path = Path(
        config["STK_path"],
        "data/output/Prescursor_data/",
        config["df_precursor"],
    )

    df_total, df_precursors = database_utils.load_data_from_file(
        df_path, df_precursors_path
    )
    
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

