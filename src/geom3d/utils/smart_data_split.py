import importlib
import torch
from geom3d.utils import oligomer_scaffold_split
from geom3d.utils.oligomer_scaffold_split import cluster_analysis
import os
import random

importlib.reload(oligomer_scaffold_split)
def smart_data_splitter(dataset, config):
    df_total = cluster_analysis(dataset, config)

    smart_dataset_size = min(config['smart_dataset_size'], len(df_total))

    smart_dataset_keys = []

    # Get the value counts of the 'Cluster' column to determine the number of oligomers in each cluster
    cluster_counts = df_total['Cluster'].value_counts()
    dataset_InChIKeys = set([data['InChIKey'] for data in dataset])
    print(f'Cluster counts: {cluster_counts}')

    # take the rows of the dataframe that have InChIKeys that are in the dataset_InChIKeys
    df_total = df_total[df_total['InChIKey'].isin(dataset_InChIKeys)]

    # Initialize a counter to keep track of the total number of samples added
    total_samples_added = 0
    # iterate through each cluster in the dataframe, and make a list of InChIKeys that has an equal amount of samples from each cluster, and randomly sample from the list for the % of smart_dataset_size / number of clusters
    for cluster in cluster_counts.index:
        # Get the InChIKeys of the samples in the current cluster
        cluster_samples = df_total[df_total['Cluster'] == cluster]
        cluster_samples = cluster_samples.sample(frac=1)
        cluster_samples = cluster_samples.reset_index(drop=True)
        cluster_samples = cluster_samples.head(int(smart_dataset_size / len(cluster_counts)))
        # Add the InChIKeys to the smart_dataset_keys list if the InChIKey is present in the dataset and is not already in the list
        if cluster_samples['InChIKey'].isin(dataset_InChIKeys).all():
            smart_dataset_keys.extend(cluster_samples['InChIKey'])
            total_samples_added += len(cluster_samples)

    if total_samples_added < smart_dataset_size:
        # add random samples from the dataset to the smart_dataset_keys list until the smart_dataset_keys list has the desired number of samples
        random_samples = sorted(dataset_InChIKeys, key=lambda x: random.random())[:smart_dataset_size - total_samples_added]
        smart_dataset_keys.extend(random_samples)
        total_samples_added += len(random_samples)

    print(f'Total samples added: {total_samples_added}')
    print(f'Smart dataset size: {len(smart_dataset_keys)}')

    # Look at the corresponding cluster of each InChIKey in the smart_dataset_keys list and make a new set of InChIKeys for each train, val, and test set
    # Use the ratios set by config['train_ratio'], config['valid_ratio'] to distribute the InChIKeys into the train, val, and test sets by having an equal amount of samples from each cluster in each set

    # Initialize dictionaries to store keys for each cluster
    keys_by_cluster = {cluster: [] for cluster in cluster_counts.index}

    # Populate keys_by_cluster
    for key in smart_dataset_keys:
        key_cluster = df_total[df_total['InChIKey'] == key]['Cluster'].values[0]
        keys_by_cluster[key_cluster].append(key)

    print('done')

    train_keys = []
    valid_keys = []
    test_keys = []

    # Distribute keys into train, validation, and test sets for each cluster
    for cluster, keys in keys_by_cluster.items():
        random.shuffle(keys)

        length = len(keys)
        train_length = int(length * config['train_ratio'])
        valid_length = int(length * config['valid_ratio'])
        print(f'Cluster {cluster} length: {length}')
        print(f'Train length: {train_length}')
        print(f'Valid length: {valid_length}')
        
        # Add the keys to the train, validation, and test sets making sure there is no overlap between the sets
        train_keys.extend(keys[:train_length])
        valid_keys.extend(keys[train_length:train_length + valid_length])
        test_keys.extend(keys[train_length + valid_length:])

    print(f'Train set size: {len(train_keys)}')
    print(f'Validation set size: {len(valid_keys)}')
    print(f'Test set size: {len(test_keys)}')

    return train_keys, valid_keys, test_keys
