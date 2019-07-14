
import time
from dataset_util import * #pylint: disable=W0614
import numpy as np
import scipy.sparse
from tqdm import tqdm
from sklearn import preprocessing
import networkx as nx
import pandas as pd
import warnings

np.random.seed(4453)

def get_root(graph):

    root = [n for n in graph.nodes() if len(list(graph.predecessors(n)))==0][0]
    return root

def get_level_nodes(graph, others):

    d = {}
    root = get_root(graph)

    for i, j in enumerate(others):
        p = nx.shortest_path_length(graph, source=root, target = j)
        if p not in d:
            d[p] = [j]
        else:
            d[p].append(j)
    
    h = sorted(d.items(), key=lambda x: x[0], reverse=True) # [(2, [3, 4, 5, 6]), (1, [1, 2]), (0, [0])]
    
    top_level = root # (0, [0]) -> [0]
    
    internal_levels = [j for i, j in h] # [(2, [3, 4, 5, 6]), (1, [1, 2])] -> [[3, 4, 5, 6], [1, 2]] by default in a bottom-up level order
    
    return top_level, internal_levels


def init_recursive_regularizer(graph, features):

    w_dict = {}

    for node in list(graph.nodes()):
        if node not in w_dict:
            w_dict[node] = np.random.randn(features,)
    
    return w_dict


def non_leaf_update(parents, children, w_dict, features):
    
    sum_pi = np.zeros(features,)
    sum_c = np.zeros(features,)
    
    len_cn = len(children)
    
    if len(parents) > 0:
            
        for p in parents:
            if isinstance(w_dict[p], np.ndarray):
                sum_pi+=w_dict[p]
            else:
                sum_pi+=0
            
        for c in children:
            if isinstance(w_dict[c], np.ndarray):
                sum_c = sum_c.squeeze()
                w_dict[c] = w_dict[c].squeeze()
                sum_c+=w_dict[c]
            else:
                sum_c+=0
                
            
    else:
        for c in children:
            if isinstance(w_dict[c], np.ndarray):
                sum_c = sum_c.squeeze()
                w_dict[c] = w_dict[c].squeeze()
                sum_c+=w_dict[c]
            else:
                sum_c+=0
    
    return len_cn, sum_c, sum_pi