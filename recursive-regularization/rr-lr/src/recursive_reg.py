
import time
from dataset_util import * #pylint: disable=W0614
import numpy as np
import scipy.sparse
from tqdm import tqdm
from sklearn import preprocessing
import networkx as nx
import pandas as pd
import warnings

np.random.seed(12345678)

def get_root(graph):

    in_degree = nx.in_degree_centrality(graph)

    root = []

    min_v = min(in_degree.values())
    for k, v in in_degree.items():
        if v == min_v:
            root.append(k)

    return root

def get_level_nodes(graph, train_node_list):
    
    temp = []
    top_level = []
    internal_levels = []
    
    while len(train_node_list) > 0:
        pred = []
        for leaves in train_node_list:
            if len(list(graph.predecessors(leaves)))>0:
                level_up = list(graph.predecessors(leaves))
                for n in level_up:
                    if n not in pred:
                        pred.append(n)
            else:
                if leaves not in top_level:
                    top_level.append(leaves)            
        train_node_list = pred
        temp.append(train_node_list)


    set_r = set(top_level)
    for level in temp:
        new_l = set(level) - set_r
        if len(new_l) > 0:
            internal_levels.append(list(new_l))         

    return top_level, internal_levels


def init_recursive_regularizer(graph, features):

    w_dict = {}

    for node in list(graph.nodes()):
        if node not in w_dict:
            w_dict[node] = np.random.randn(features,)
    
    return w_dict


def non_leaf_update(parents, children, w_dict):
    
    sum_pi = 0
    sum_c = 0
    
    len_cn = len(children)
    
    if len(parents) > 0:
            
        for p in parents:
            sum_pi+=w_dict[p]
            
        for c in children:
            sum_c+=w_dict[c]
            
    else:
        for c in children:
            sum_c+=w_dict[c]
    
    return len_cn, sum_c, sum_pi