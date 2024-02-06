""" script to turn a a dataset into custom fragment scaffold split"""

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


def oligomer_scaffold_splitter(dataset, config):
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

    y_IP = df_total['ionisation potential (eV)'].values
    X_6mer_inch = df_total['BB'].values
    X_frag_mol = df_precursors['mol_opt'].values
    X_frag_inch = df_precursors['InChIKey'].values
    keys_6mer = df_total['InChIKey'].values

    # Generate Morgan fingerprints for the dataset
    morgan_fingerprints = generate_morgan_fingerprints(X_frag_mol)

    # make a list of the InChiKeys in the dataset
    morgan_fingerprints = [list(morgan_fingerprints[i]) for i in range(len(X_frag_mol))]

    # Number of components you want to retain after PCA
    n_components = 2  # You can change this based on your requirements

    # Perform PCA on the Morgan fingerprints
    pca = PCA(n_components=n_components)
    pca_scores = pca.fit_transform(morgan_fingerprints)

    # Create a list to store average PCA scores for each row in df_total
    average_pca_scores = []

    # Define the total number of iterations
    total_iterations = len(df_total)

    # Create a tqdm instance
    for index, row in tqdm(df_total.iterrows(), total=total_iterations, desc="Calculating the average PCA score for each oligomer"):
        row_pca_scores = []
        error_keys = []
        error = 0

        # Extract InChIKeys from columns InChIKeys_0 to InChIKeys_5
        for i in range(6):
            inchkey = row[f'InChIKey_{i}']
            # Check if the InChIKey is not None (assuming there are no missing values)
            if inchkey is not None:
                # Find the corresponding PCA score for the InChIKey if it exists
                if inchkey in X_frag_inch:
                    fragment_index = np.where(X_frag_inch == inchkey)[0][0]
                    pca_score = pca_scores[fragment_index]
                    row_pca_scores.append(pca_score)
                else:
                    # Handle the case where the InChIKey is not in X_frag_inch
                    row_pca_scores.append(None)
                    if inchkey not in error_keys:
                        error_keys.append(inchkey)
        
        # Ensure that all arrays in row_pca_scores have the same shape
        row_pca_scores = [score for score in row_pca_scores if score is not None and score.shape[0] == n_components]

        # Calculate the average PCA score for the row
        if row_pca_scores:
            average_pca_score = np.mean(row_pca_scores, axis=0)
            average_pca_scores.append(average_pca_score)
        else:
            # Handle the case where there are no valid InChIKeys for the row
            average_pca_scores.append(None)
            error += 1

    # Convert the list of average PCA scores to a NumPy array
    average_pca_scores_array = np.array(average_pca_scores)
    print('Problematic keys:', error_keys)
    print('Number of Oligomers not converted:', error)

    # Define HDBSCAN parameters (adjust as needed)
    min_cluster_size = config["oligomer_min_cluster_size"]  # Minimum size for a cluster to be considered valid
    min_samples = config["oligomer_min_samples"]  # Minimum number of points required to form a core point

    print('Clustering with min_cluster_size =', min_cluster_size, 'and min_samples =', min_samples)

    # Create a HDBSCAN instance
    hdb_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)

    # Fit the model to the average PCA scores
    cluster_labels = hdb_model.fit_predict(average_pca_scores_array)

    # Access the cluster assignments for each data point and keys_6mer
    cluster_assignments = dict(zip(keys_6mer, cluster_labels))

    chosen_cluster = config["test_set_oligomer_cluster"]  # Choose the cluster you want to use for the test set

    cluster_keys = []
    for key, value in cluster_assignments.items():
        if value == chosen_cluster:
            cluster_keys.append(key)

    print(f"Length of Cluster {chosen_cluster}: {len(cluster_keys)}")

    return cluster_keys


# Function to generate Morgan fingerprints
def generate_morgan_fingerprints(molecules, radius=2, n_bits=2048):
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) for mol in molecules]
    return fingerprints


# Function to generate ECFP fingerprints
def generate_ecfp_fingerprints(molecules, radius=2, n_bits=2048):
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) for mol in molecules]
    return fingerprints


# Function to calculate Tanimoto similarity between fingerprints
def calculate_tanimoto_similarity(fingerprint1, fingerprint2):
    return DataStructs.TanimotoSimilarity(fingerprint1, fingerprint2)

def prepare_oligomer_plot(dataset, config):
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

    y_IP = df_total['ionisation potential (eV)'].values
    X_6mer_inch = df_total['BB'].values
    X_frag_mol = df_precursors['mol_opt'].values
    X_frag_inch = df_precursors['InChIKey'].values
    keys_6mer = df_total['InChIKey'].values

    # Generate Morgan fingerprints for the dataset
    morgan_fingerprints = generate_morgan_fingerprints(X_frag_mol)

    # make a list of the InChiKeys in the dataset
    morgan_fingerprints = [list(morgan_fingerprints[i]) for i in range(len(X_frag_mol))]

    # Number of components you want to retain after PCA
    n_components = 2  # You can change this based on your requirements

    # Perform PCA on the Morgan fingerprints
    pca = PCA(n_components=n_components)
    pca_scores = pca.fit_transform(morgan_fingerprints)

    # Create a list to store average PCA scores for each row in df_total
    average_pca_scores = []

    # Define the total number of iterations
    total_iterations = len(df_total)

    # Create a tqdm instance
    for index, row in tqdm(df_total.iterrows(), total=total_iterations, desc="Calculating the average PCA score for each oligomer"):
        row_pca_scores = []
        error_keys = []
        error = 0

        # Extract InChIKeys from columns InChIKeys_0 to InChIKeys_5
        for i in range(6):
            inchkey = row[f'InChIKey_{i}']
            # Check if the InChIKey is not None (assuming there are no missing values)
            if inchkey is not None:
                # Find the corresponding PCA score for the InChIKey if it exists
                if inchkey in X_frag_inch:
                    fragment_index = np.where(X_frag_inch == inchkey)[0][0]
                    pca_score = pca_scores[fragment_index]
                    row_pca_scores.append(pca_score)
                else:
                    # Handle the case where the InChIKey is not in X_frag_inch
                    row_pca_scores.append(None)
                    if inchkey not in error_keys:
                        error_keys.append(inchkey)
        
        # Ensure that all arrays in row_pca_scores have the same shape
        row_pca_scores = [score for score in row_pca_scores if score is not None and score.shape[0] == n_components]

        # Calculate the average PCA score for the row
        if row_pca_scores:
            average_pca_score = np.mean(row_pca_scores, axis=0)
            average_pca_scores.append(average_pca_score)
        else:
            # Handle the case where there are no valid InChIKeys for the row
            average_pca_scores.append(None)
            error += 1

    # Convert the list of average PCA scores to a NumPy array
    average_pca_scores_array = np.array(average_pca_scores)
    print('Problematic keys:', error_keys)
    print('Number of Oligomers not converted:', error)
    print('PCA scores converted')

    return average_pca_scores_array, keys_6mer


def cluster_analysis(dataset, config, min_cluster_size=750, min_samples=50):

    average_pca_scores_array, keys_6mer  = prepare_oligomer_plot(dataset, config)

    # min_cluster_size = 750  # Minimum size for a cluster to be considered valid
    # min_samples = 50  # Minimum number of points required to form a core point

    print('Clustering with min_cluster_size =', min_cluster_size, 'and min_samples =', min_samples)

    # Create a HDBSCAN instance
    hdb_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)

    # Fit the model to the average PCA scores
    cluster_labels = hdb_model.fit_predict(average_pca_scores_array)

    counter = Counter(cluster_labels)

    print("Cluster split and amount of Oligomers in each:", counter)

    return cluster_labels


def pca_plot(dataset, config):
    
    # Calculate average PCA scores and cluster labels
    average_pca_scores_array, keys_6mer  = prepare_oligomer_plot(dataset, config)

    min_cluster_size = config["oligomer_min_cluster_size"]  # Minimum size for a cluster to be considered valid
    min_samples = config["oligomer_min_samples"]  # Minimum number of points required to form a core point

    print('Clustering with min_cluster_size =', min_cluster_size, 'and min_samples =', min_samples)
    # Create a HDBSCAN instance
    hdb_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)

    # Fit the model to the average PCA scores
    cluster_labels = hdb_model.fit_predict(average_pca_scores_array)

    selected_cluster = config["test_set_oligomer_cluster"]  # Choose the cluster you want to use for the test set

    # Filter out the labels in the selected cluster
    df_selected_cluster = pd.DataFrame({
        'PCA1': average_pca_scores_array[:, 0],
        'PCA2': average_pca_scores_array[:, 1],
        'Cluster': cluster_labels
    })

    # Plot all clusters
    plt.figure(figsize=(10, 10))
    plt.scatter(df_selected_cluster['PCA1'], df_selected_cluster['PCA2'], c=df_selected_cluster['Cluster'], cmap='viridis', alpha=0.7)

    # Highlight the specific cluster
    df_cluster_spec = df_selected_cluster[df_selected_cluster['Cluster'] == selected_cluster]
    plt.scatter(df_cluster_spec['PCA1'], df_cluster_spec['PCA2'], c='red', label=f'Cluster {selected_cluster}', alpha=0.9)

    plt.legend()
    plt.title("Clusters of oligomers based on average PCA scores")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


# still to do for oligomer
def substructure_analysis_oligomers(dataset, config, selected_cluster=6, min_cluster_size=750, min_samples=50):
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
    
    # Prepare data for oligomer analysis
    average_pca_scores_array, keys_6mer = prepare_oligomer_plot(dataset, config)
    print('PCA scores converted')

    # Clustering
    min_cluster_size = config["oligomer_min_cluster_size"]
    min_samples = config["oligomer_min_samples"]

    print('Clustering with min_cluster_size =', min_cluster_size, 'and min_samples =', min_samples)

    hdb_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    cluster_labels = hdb_model.fit_predict(average_pca_scores_array)
    cluster_assignments = dict(zip(keys_6mer, cluster_labels))

    selected_cluster = config["test_set_oligomer_cluster"]
    # Filter out the data points in the specified cluster
    selected_cluster_keys = [oligomer_key for oligomer_key, cluster_id in cluster_assignments.items() if cluster_id == selected_cluster]

    print(f"Length of Cluster {selected_cluster}: {len(selected_cluster_keys)}")
    print('Clustered')

    print('Performing substructure analysis for Cluster', selected_cluster)

    # Generate common substructures for each molecule in the cluster
    common_substructures = []
    counter = 0

    # Loop through the oligomers in the cluster
    for oligomer_key in tqdm(selected_cluster_keys, desc=f"Generating substructures for Cluster {selected_cluster}"):
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
            print(f"Oligomer {oligomer_key} (Cluster {selected_cluster}): Not enough fragments for comparison.")
        else:
            # Find the common substructure in the combined oligomer
            common_substructure = rdFMCS.FindMCS([combined_molecule, combined_molecule])
            common_substructure = Chem.MolFromSmarts(common_substructure.smartsString)
            common_substructures.append(common_substructure)


        #visualise only one combined molecule in the cluster in 2D, so its easier to see
        if len(fragments) == 6 and counter == 0:
            print(f'representative oligomer in cluster {selected_cluster}')
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

