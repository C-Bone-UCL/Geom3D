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

from geom3d.utils import database_utils


def fragment_scaffold_splitter(dataset, config):
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

    X_frag_mol = df_precursors['mol_opt_2'].values
    X_InChIKey = df_precursors['InChIKey'].values

    # Generate Morgan fingerprints for the dataset
    morgan_fingerprints = generate_morgan_fingerprints(X_frag_mol)

    # make a list of the InChiKeys in the dataset
    morgan_fingerprints = [list(morgan_fingerprints[i]) for i in range(len(X_frag_mol))]

    # Calculate the linkage matrix for hierarchical clustering, 
    morgan_matrix = linkage(morgan_fingerprints, method='average', metric='jaccard', optimal_ordering=True)

    # Cut the dendrogram to obtain clusters
    threshold = config["fragment_cluster_threshold"]  # Adjust the threshold based on the dendrogram
    clusters_morgan = fcluster(morgan_matrix, threshold, criterion='distance')

    #Maka a list the InChiKeys present in Morgan cluster
    morgan_keys = {}
    for i in range(len(clusters_morgan)):
        morgan_keys[X_InChIKey[i]] = clusters_morgan[i]

    #make a table of the different InChiKeys and their cluster, naming both the columns
    morgan_keys = pd.DataFrame(morgan_keys.items(), columns=['InChIKey', 'Cluster'])
    morgan_keys

    test_cluster = config["test_set_fragment_cluster"]

    oligomer_has_frag_in_cluster_keys = []

    # Columns to check
    columns_to_check = [f'InChIKey_{i}' for i in range(6)]

    # Iterate through the keys in morgan_keys['Cluster']
    for key in morgan_keys['InChIKey'][morgan_keys['Cluster'] == test_cluster]:
        # Initialize a list to store associated df_total['InChIKey'] values
        associated_keys = []
        # Iterate through the columns to check
        for column in columns_to_check:
            # Check if the key is present in the current column
            if key in df_total[column].values:
                # Append the associated df_total['InChIKey'] values to the list
                associated_keys.extend(df_total['InChIKey'][df_total[column] == key].values)
        # If any associated keys were found, append them to the oligomer_has_frag_in_cluster_keys list
        if associated_keys:
            oligomer_has_frag_in_cluster_keys.extend(associated_keys)
        else:
            print('no')

    #how many oligomer_has_frag_in_cluster_keys are in the dataset and remove duplicates
    oligomer_has_frag_in_cluster_keys = list(set(oligomer_has_frag_in_cluster_keys))

    #make a list of the InChiKeys in the dataset
    dataset_keys = [dataset[i]['InChIKey'] for i in range(len(dataset))]

    #make a list of the InChiKeys in the dataset that are in the cluster
    dataset_keys_in_cluster = []
    for key in dataset_keys:
        if key in oligomer_has_frag_in_cluster_keys:
            dataset_keys_in_cluster.append(key)

    print('Number of Oligomers that have a fragment in the cluster:', len(dataset_keys_in_cluster))

    return dataset_keys_in_cluster


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

def prepare_frag_plot(dataset, config):
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

    X_frag_mol = df_precursors['mol_opt_2'].values

    # Generate Morgan fingerprints for the dataset
    morgan_fingerprints = generate_morgan_fingerprints(X_frag_mol)

    # Combine Morgan and ECFP fingerprints for clustering
    morgan_fingerprints = [list(morgan_fingerprints[i]) for i in range(len(X_frag_mol))]

    # Calculate the linkage matrix for hierarchical clustering
    morgan_matrix = linkage(morgan_fingerprints, method='average', metric='jaccard')

    # Cut the dendrogram to obtain clusters
    threshold = config["fragment_cluster_threshold"]  # Adjust the threshold based on the dendrogram
    clusters_morgan = fcluster(morgan_matrix, threshold, criterion='distance')

    return X_frag_mol, morgan_fingerprints, morgan_matrix, clusters_morgan

# Plot the dendrograms next to eachother
def plot_dendrograms(dataset, config):
    
    X_frag_mol, morgan_fingerprints, morgan_matrix, clusters_morgan = prepare_frag_plot(dataset, config)

    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Morgan fingerprints")
    dendrogram(morgan_matrix, ax=ax1, labels=clusters_morgan, orientation='left')

    plt.tight_layout()
    plt.show()


def cluster_analysis(dataset, config, threshold=0.5):

    X_frag_mol, morgan_fingerprints, morgan_matrix, clusters_morgan = prepare_frag_plot(dataset, config)

    # Adjust the threshold based on the dendrogram
    clusters_morgan = fcluster(morgan_matrix, threshold, criterion='distance')

    #number of molecules in each cluster
    unique, counts = np.unique(clusters_morgan, return_counts=True)
    dict(zip(unique, counts))
    print("Number of molecules in each cluster for morgan fp:", dict(zip(unique, counts)))


def pca_plot(dataset, config, selected_cluster=1, threshold=0.5):

    X_frag_mol, morgan_fingerprints, morgan_matrix, clusters_morgan = prepare_frag_plot(dataset, config)

    # Adjust the threshold based on the dendrogram
    clusters_morgan = fcluster(morgan_matrix, threshold, criterion='distance')

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(morgan_fingerprints)

    # Create a DataFrame for visualization
    df_pca = pd.DataFrame({'PCA1': pca_result[:, 0], 'PCA2': pca_result[:, 1], 'PCA3': pca_result[:, 2], 'Cluster': clusters_morgan})

    # Plot the PCA in 3D
    ax = plt.axes(projection='3d')
    ax.scatter3D(df_pca['PCA1'], df_pca['PCA2'], df_pca['PCA3'], c=df_pca['Cluster'], cmap='viridis', alpha=0.2)

    # Filter the DataFrame to include only selected cluster
    df_cluster_spec = df_pca[df_pca['Cluster'] == selected_cluster]

    # Plot only the values for cluster 5 with a different color
    ax.scatter3D(df_cluster_spec['PCA1'], df_cluster_spec['PCA2'], df_cluster_spec['PCA3'], c='red', label=f'Cluster {selected_cluster}', alpha=0.9)

    # Move the legend to the right of the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.title(f'PCA Plot with Clusters, Highlighted Cluster {selected_cluster}')
    plt.show()

def substructure_analysis(dataset, config, selected_cluster=1, threshold=0.5):

    X_frag_mol, morgan_fingerprints, morgan_matrix, clusters_morgan = prepare_frag_plot(dataset, config)

    # Adjust the threshold based on the dendrogram
    clusters_morgan = fcluster(morgan_matrix, threshold, criterion='distance')

    #length of cluster
    print('Length of cluster:', len([i for i, cluster_id in enumerate(clusters_morgan) if cluster_id == selected_cluster]))
    print(f"Cluster {selected_cluster} representative molecule:")

    # Find common substructure for the specified cluster 
    representative_molecules = [X_frag_mol[i] for i, cluster_id in enumerate(clusters_morgan) if cluster_id == selected_cluster]
    cluster_smiles = [Chem.MolToSmiles(X_frag_mol[j]) for j, cluster_id in enumerate(clusters_morgan) if cluster_id == selected_cluster]

    # Display one molecule from the cluster
    img = Draw.MolToImage(representative_molecules[0])
    display(img)

    # Generate common substructures for each molecule in the cluster
    common_substructures = []
    for smiles in cluster_smiles:
        mol = Chem.MolFromSmiles(smiles)
        mcs = rdFMCS.FindMCS([mol] + representative_molecules)
        common_substructure = Chem.MolFromSmarts(mcs.smartsString)
        common_substructures.append(common_substructure)

    # Check if there's only one molecule in the cluster
    if len(representative_molecules) < 2:
        print(f"Cluster {selected_cluster}: Not enough molecules for comparison.")
    else:
        # Count the occurrences of each substructure
        substructure_counts = Counter([Chem.MolToSmarts(sub) for sub in common_substructures])

        # Rank substructures based on frequency
        ranked_substructures = sorted(substructure_counts.items(), key=lambda x: x[1], reverse=True)

        # Display the top N substructures and their occurrences
        top_n = min(3, len(ranked_substructures))  # Choose the smaller of 3 and the actual number of substructures
        for i, (substructure, count) in enumerate(ranked_substructures[:top_n]):
            print(f"Top {i + 1} Substructure (Frequency: {count} molecules):")
            img = Draw.MolToImage(Chem.MolFromSmarts(substructure))
            display(img)