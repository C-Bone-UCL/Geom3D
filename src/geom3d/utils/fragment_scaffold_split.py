""" script to turn a a dataset into custom fragment scaffold split"""

import os
from collections import Counter
from pathlib import Path
from tqdm import tqdm

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
    num_mols = len(dataset)
    df_total, df_precursors, X_frag_mol, X_InChIKey = load_dataset(dataset, config)

    morgan_keys = check_if_dataset_exists(dataset, config, config["fragment_cluster_threshold"])

    test_cluster = config["test_set_fragment_cluster"]

    print('Chosen cluster:', test_cluster)
    print('Number of molecules in the cluster:', len(morgan_keys['InChIKey'][morgan_keys['Cluster'] == test_cluster]))

    oligomer_has_frag_in_cluster_keys = []

    # Columns to check
    columns_to_check = [f'InChIKey_{i}' for i in range(6)]

    # Iterate through the keys in morgan_keys['Cluster'] with tqdm
    for key in tqdm(morgan_keys['InChIKey'][morgan_keys['Cluster'] == test_cluster], desc="analysing how many oliogomers have chosen fragments"):
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


# Plot the dendrograms next to eachother
def plot_dendrograms(dataset, config):

    morgan_matrix, morgan_keys = prepare_frag_plot(dataset, config)

    # Plot the dendrogram
    plt.figure(figsize=(15, 5))
    plt.title('Hierarchical Clustering Dendrogram')
    dendrogram(morgan_matrix, no_labels=True)
    plt.show()



def cluster_analysis(dataset, config, threshold):

    morgan_matrix, morgan_keys = prepare_frag_plot(dataset, config)

    print(f"Clustering dataset with threshold {threshold}")

    clusters_morgan = fcluster(morgan_matrix, threshold, criterion='distance')

    morgan_keys['Cluster'] = clusters_morgan
    
    # show the first 5 rows of the dataframe
    print(morgan_keys.columns)

    #number of molecules in each cluster
    unique, counts = np.unique(morgan_keys['Cluster'], return_counts=True)
    dict(zip(unique, counts))
    print("Number of molecules in each cluster for morgan fp:", dict(zip(unique, counts)))

    # save the cluster assignments to a file
    split_file_path = config["running_dir"] + f"/datasplit_{len(dataset)}_{config['split']}_threshold_{threshold}.csv"
    morgan_keys.to_csv(split_file_path, index=False)  # Set index=False to exclude row indices from the saved file
    print(f"Dataset cluster assignments saved to {split_file_path}")

    return morgan_keys


def pca_plot(dataset, config, selected_cluster=1, threshold=0.5):
    
    morgan_keys = check_if_dataset_exists(dataset, config, threshold)

    morgan_fingerprints = morgan_keys['Morgan_Fingerprint']
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(morgan_fingerprints.to_list())

    # Create a DataFrame for visualization
    df_pca = pd.DataFrame({'PCA1': pca_result[:, 0], 'PCA2': pca_result[:, 1], 'PCA3': pca_result[:, 2], 'Cluster': morgan_keys['Cluster']})

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
    df_total, df_precursors, X_frag_mol, X_InChIKey = load_dataset(dataset, config)
    
    morgan_keys = check_if_dataset_exists(dataset, config, threshold)

    #length of cluster
    print('Length of cluster:', len([i for i, cluster_id in enumerate(morgan_keys['Cluster']) if cluster_id == selected_cluster]))
    print(f"Cluster {selected_cluster} representative molecule:")

    # Find common substructure for the specified cluster 
    representative_molecules = [X_frag_mol[i] for i, cluster_id in enumerate(morgan_keys['Cluster']) if cluster_id == selected_cluster]
    cluster_smiles = [Chem.MolToSmiles(X_frag_mol[j]) for j, cluster_id in enumerate(morgan_keys['Cluster']) if cluster_id == selected_cluster]

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


def calculate_morgan_fingerprints(mols):
    morgan_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mols]
    return morgan_fps

def calculate_tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def prepare_frag_plot(dataset, config):
    df_total, df_precursors, X_frag_mol, X_InChIKey = load_dataset(dataset, config)

    # Generate Morgan fingerprints for the dataset
    morgan_fingerprints = calculate_morgan_fingerprints(X_frag_mol)

    # tanimoto_sim = np.zeros((len(X_frag_mol), len(X_frag_mol)))
    # for i in range(len(X_frag_mol)):
    #     for j in range(len(X_frag_mol)):
    #         tanimoto_sim[i,j] = calculate_tanimoto_similarity(morgan_fingerprints[i], morgan_fingerprints[j])
    #         tanimoto_sim[j,i] = tanimoto_sim[i,j]

    # Calculate the linkage matrix for hierarchical clustering, 
    morgan_matrix = linkage(morgan_fingerprints, method='average', metric='jaccard', optimal_ordering=True)
        
    morgan_keys = pd.DataFrame({'InChIKey': X_InChIKey, 'Morgan_Fingerprint': morgan_fingerprints})
    
    # show the first 5 rows of the dataframe
    print(morgan_keys.columns)

    return morgan_matrix, morgan_keys

def load_dataset(dataset, config):
    seed = config["seed"]
    np.random.seed(seed)
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

    return df_total, df_precursors, X_frag_mol, X_InChIKey

def check_if_dataset_exists(dataset, config, threshold):
    split_file_path = config["running_dir"] + f"/datasplit_{len(dataset)}_{config['split']}_threshold_{threshold}.csv"

    if os.path.exists(split_file_path):
        # load the dictionary from the file
        print(f"Loading dataset indices from {split_file_path}")
        morgan_keys = pd.read_csv(split_file_path)

    else:
        morgan_keys = cluster_analysis(dataset, config, threshold)

    return morgan_keys
