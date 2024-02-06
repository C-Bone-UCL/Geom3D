""" script to turn a a dataset into custom target based split that puts the 10% best performing molecules in the test set """

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

def top_target_split(dataset):
    """
    Split a dataset into a training and test set based on the target value.
    The 10% best performing molecules are put in the test set.
    """
    # Get the target column
    target = dataset["target"]
    # Sort the target values and get the indices
    sorted_indices = np.argsort(target)
    # Get the indices of the 10% best performing molecules
    test_indices = sorted_indices[-int(len(sorted_indices) * 0.1) :]

    # find the InChIKeys of the test set
    test_set = dataset.iloc[test_indices]
    test_set_inchikeys = test_set["inchikey"].values

    return test_set_inchikeys
