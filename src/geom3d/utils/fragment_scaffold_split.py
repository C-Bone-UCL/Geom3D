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
import plotly.graph_objs as go


from geom3d.utils import database_utils


def fragment_scaffold_splitter(dataset, config):
    """
    Split a dataset into a training and test set based on the fragment scaffold.
    The 10% best performing molecules are put in the test set.

    Args:
    - dataset (list): list of dictionaries containing the molecules and their properties
    - config (dict): dictionary containing the configuration parameters

    Returns:
    - test_set_inchikeys (list): list of InChIKeys of the molecules in the test set
    """

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
    """
    Plot the dendrogram for the dataset.

    Args:
    - dataset (list): list of dictionaries containing the molecules and their properties
    - config (dict): dictionary containing the configuration parameters
    """

    morgan_matrix, morgan_keys = prepare_frag_plot(dataset, config)

    # Plot the dendrogram
    plt.figure(figsize=(15, 5))
    plt.title('Hierarchical Clustering Dendrogram')
    dendrogram(morgan_matrix, no_labels=True)
    plt.show()



def cluster_analysis(dataset, config, threshold):
    """
    Cluster the dataset based on the fragment fingerprints.

    Args:
    - dataset (list): list of dictionaries containing the molecules and their properties
    - config (dict): dictionary containing the configuration parameters
    - threshold (float): threshold for clustering

    Returns:
    - morgan_keys (pd.DataFrame): dataframe containing the InChIKeys and the cluster assignments

    """

    # Load the dataset
    morgan_matrix, morgan_keys = prepare_frag_plot(dataset, config)
    print(f"Clustering dataset with threshold {threshold}")
    clusters_morgan = fcluster(morgan_matrix, threshold, criterion='distance')

    print("Number of clusters:", len(np.unique(clusters_morgan)))

    # Merge clusters with less than 40 molecules with nearby clusters
    unique_clusters, counts = np.unique(clusters_morgan, return_counts=True)
    print("Number of molecules in each cluster for morgan fp:", dict(zip(unique_clusters, counts)))

    # while min(counts) < 30:
    for cluster, count in zip(unique_clusters, counts):
        if count < 30:
            # Find adjacent clusters
            adjacent_clusters = [
                c for c in unique_clusters
                if c != cluster and abs(c - cluster) <= 1
            ]

            if adjacent_clusters:
                # Find the adjacent cluster with the lowest count
                nearest_cluster = min(adjacent_clusters, key=lambda x: counts[np.where(unique_clusters == x)])
                # Merge the current cluster into the nearest one
                clusters_morgan[clusters_morgan == cluster] = nearest_cluster
            else:
                print("No adjacent clusters found.")

    # Relabel clusters to ensure consecutive numbering
    unique_clusters, counts = np.unique(clusters_morgan, return_counts=True)
    new_cluster_mapping = dict(zip(unique_clusters, range(1, len(unique_clusters) + 1)))
    clusters_morgan = np.array([new_cluster_mapping[c] for c in clusters_morgan])  

    # Update InChIKey DataFrame with new cluster assignments
    morgan_keys['Cluster'] = clusters_morgan

    # Number of molecules in each cluster
    unique_clusters, counts = np.unique(morgan_keys['Cluster'], return_counts=True)
    print("Number of molecules in each cluster after merging small clusters:", dict(zip(unique_clusters, counts)))

    # # save the cluster assignments to a file
    # split_file_path = config["running_dir"] + f"/datasplit_{len(dataset)}_{config['split']}_threshold_{threshold}.csv"

    # morgan_keys.to_csv(split_file_path, index=False)  # Set index=False to exclude row indices from the saved file
    # print(f"Dataset cluster assignments saved to {split_file_path}")
    # morgan_keys = resplit_cluster(morgan_keys)
    
    return morgan_keys


def pca_plot(dataset, config, selected_cluster=1, threshold=0.5):
    """
    Plot the PCA of the dataset, highlighting the specified cluster.

    Args:
    - dataset (list): list of dictionaries containing the molecules and their properties
    - config (dict): dictionary containing the configuration parameters
    - selected_cluster (int): cluster to highlight
    - threshold (float): threshold for clustering

    """
    morgan_keys = check_if_dataset_exists(dataset, config, threshold)
    morgan_fingerprints = np.array(morgan_keys['Morgan_Fingerprint'].to_list())

    # Apply PCA to reduce dimensionality to 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(morgan_fingerprints)

    # Create 3D scatter plot
    scatter = go.Scatter3d(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        z=pca_result[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=morgan_keys['Cluster'],
            colorscale='Viridis',
            opacity=0.8
        )
    )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='PCA1'),
            yaxis=dict(title='PCA2'),
            zaxis=dict(title='PCA3'),
        ),
        title='3D PCA Plot with Clusters'
    )

    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()

def substructure_analysis(dataset, config, selected_cluster=1, threshold=0.5):
    """
    Generate common substructures for the specified cluster.

    Args:
    - dataset (list): list of dictionaries containing the molecules and their properties
    - config (dict): dictionary containing the configuration parameters
    - selected_cluster (int): cluster to analyze
    - threshold (float): threshold for clustering
    
    """
    
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
    morgan_fps_np = [np.array(fp) for fp in morgan_fps]

    return morgan_fps_np

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
    # possible metrics: 'euclidean', 'jaccard', 'hamming', 'dice', 'matching', 'yule', 'kulsinski', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath'

    metric = 'rogerstanimoto'
    morgan_matrix = linkage(morgan_fingerprints, method='average', metric=metric, optimal_ordering=True)
    print('clustering done with metric:', metric)
    
    # Create a DataFrame with the InChIKeys and the Morgan fingerprints (save the fingerprints as np arrays)
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

    # if os.path.exists(split_file_path):
    #     # load the dictionary from the file
    #     print(f"Loading dataset indices from {split_file_path}")
    #     morgan_keys = pd.read_csv(split_file_path)

    # else:
    morgan_keys = cluster_analysis(dataset, config, threshold)

    return morgan_keys

def resplit_cluster(morgan_keys):
    morgan_fingerprints = morgan_keys[morgan_keys["Cluster"] == 5]["Morgan_Fingerprint"]
    morgan_inchikeys = morgan_keys[morgan_keys["Cluster"] == 5]["InChIKey"]

    # make a list of the morgan fingerprints
    morgan_fingerprints = list(morgan_fingerprints)

    metric = 'rogerstanimoto'
    morgan_matrix = linkage(morgan_fingerprints, method='average', metric=metric, optimal_ordering=True)
    print('splitting the cluster 5 into 2 smaller clusters with:', metric)

    clusters_morgan = fcluster(morgan_matrix, 0.051, criterion='distance')

    # Merge clusters with less than 40 molecules with nearby clusters
    unique_clusters, counts = np.unique(clusters_morgan, return_counts=True)

    # make a dataframe with the InChiKeys and the clusters
    df_morgan_cluster_5 = pd.DataFrame({"InChIKey": morgan_inchikeys, "Cluster": clusters_morgan})

    # merge clusters 1 and 2 together into cluster 5, and then 3,4,5 together into cluster 7
    df_morgan_cluster_5['Cluster'] = df_morgan_cluster_5['Cluster'].replace([1, 2], 8)
    df_morgan_cluster_5['Cluster'] = df_morgan_cluster_5['Cluster'].replace([3, 4, 5], 7)
    for index, row in df_morgan_cluster_5.iterrows():
        inchikey = row["InChIKey"]
        cluster = row["Cluster"]
        morgan_keys.loc[morgan_keys["InChIKey"] == inchikey, "Cluster"] = cluster

    # find the clusters that are greater than 5, and then subtract 1 from the cluster value
    morgan_keys.loc[morgan_keys["Cluster"] > 5, "Cluster"] = morgan_keys.loc[morgan_keys["Cluster"] > 5, "Cluster"] - 1

    # see how many samples are in each cluster for the morgan_keys dataframe
    print(morgan_keys["Cluster"].value_counts())

    return morgan_keys




