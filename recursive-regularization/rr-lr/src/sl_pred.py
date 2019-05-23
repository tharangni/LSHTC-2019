'''
Test hierarchical flat classifier
using cost sensitive learning based on hierarchical costs

REF:
Anveshi Charuvaka and Huzefa Rangwala "HierCost: Improving Large Scale
Hierarchical Classification with Cost Sensitive Learning"  European Conference
on Machine Learning and Principles and Practice of Knowledge Discovery in
Databases, 2015
'''
import os
import numpy as np
import networkx as nx
from sklearn.metrics import f1_score
from dataset_util import *
from tqdm import tqdm
import scipy.sparse
import warnings


def get_leaf_nodes(graph_path):
    '''
    Get list of leaf nodes.

    Returns:
        np.ndarray[]:int : list of leaf nodes in the graph.
    '''

    if "graphml" not in graph_path:
        graph = safe_read_graph(graph_path)
    else:
        graph = safe_read_graphml(graph_path)

    # graph = nx.read_edgelist(graph_path, create_using=nx.DiGraph(),nodetype=int)
    leaf_nodes = np.array([ n for n in graph.nodes()  if len(list(graph.successors(n))) == 0],dtype=int)
    leaf_nodes = np.array(graph.nodes())
    return leaf_nodes

def pred_multiclass(X_test, model_dir, leaf_nodes):
    '''
    Predict class labels for test set.

    Args:
        X_test (np.ndarray[num_examples x num_features]:float): test dataset features.
        model_dir (str): Directory containing pickled model files (ending with *.p)
                         belonging to class LogisticCost, with one *.p file per leaf node.
        leaf_nodes (np.ndarray[]:int):list of leaf nodes in the graph.

    Returns:
        np.ndarray[num_examples]: predicted labels for test dataset.

    '''
    num_examples = X_test.shape[0]
    y_pred = np.zeros(num_examples, int)
    best_scores = np.zeros(num_examples)

    print(len(leaf_nodes))
    for idx, node in enumerate(tqdm(leaf_nodes)):
        model_save_path = '{}/model_h_{}.p'.format(model_dir, node)
        try:
            node_model = safe_pickle_load(model_save_path)
        except:
            node_model = None
            
        if node_model != None:
            print(node, " exists")
            node_score = node_model.decision_function(X_test)
            if idx == 0:
                y_pred[:] = node
                best_scores = node_score
            else:
                select_index = node_score > best_scores
                y_pred[select_index] = node
                best_scores[select_index] = node_score[select_index]
        else:
            pass
#             print("node model {}".format(node), "not found. empty predict")

    return y_pred


def main(args):
    '''
    Driver function to
        - parse command line argumnets.
        - obtain predictions for test set and write them to a file.
    '''
    
    if "npy" not in args.dataset:
        X_test, y_test = safe_read_svmlight_file_multilabel(
                 args.dataset, args.features)
    else:
        fe, ex = os.path.splitext(args.dataset)
        temp = "".join(fe.split("_")[0])
        label_path = "{}_labels{}".format(temp, ex)
        X_test, y_test = read_embeddings(args.dataset, label_path)
    
    leaf_nodes = get_leaf_nodes(args.hierarchy)
    y_pred = pred_multiclass(X_test, args.model_dir, leaf_nodes)
    np.savetxt(args.pred_path, y_pred, fmt="%d")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # suppress UndefinedMetricWarning for macro_f1
        micro_f1 = f1_score(y_test, y_pred, average='micro')
        macro_f1 = f1_score(y_test, y_pred, average='macro')

        print("Micro-F1 = {:.5f}".format(micro_f1))
        print("Macro-F1 = {:.5f}".format(macro_f1))

