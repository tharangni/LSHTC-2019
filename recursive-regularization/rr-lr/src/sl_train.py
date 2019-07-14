'''
Train hierarchical flat classifier
using cost sensitive learning based on hierarchical costs

REF:
Anveshi Charuvaka and Huzefa Rangwala "HierCost: Improving Large Scale
Hierarchical Classification with Cost Sensitive Learning"  European Conference
on Machine Learning and Principles and Practice of Knowledge Discovery in
Databases, 2015
'''
import warnings
warnings.simplefilter("ignore")

from tqdm import tqdm
import sys, os, pickle, time
from dataset_util import *
import numpy as np
import scipy.sparse
from slcost import *
from recursive_reg import *
from logcost import LogisticCost

def train_and_output_model(X, class_labels, train_node, graph, 
    cost_type, imbalance, rho, outpath, w_pi, w_n, sum_c, mod_cn):
    '''
    Train model and save model to a pickled file.

    Args:
        X (np.ndarray[num_instances x num_features]:float): Training dataset
        class_labels (np.ndarray[num_instances]:int): class labels.
        train_node (int): positive training label.
        cost_type (str): cost type in ["lr", "nca", "trd", "etrd"]
        imbalance (bool): Include imbalance cost?
        rho (float): Regularization parameter
        outpath (str): output path of the model

    Returns:
        None
    '''
    print("Training Model {} :  ".format(train_node))
    start = time.time()

    y = 2*(class_labels == train_node).astype(int) - 1
    cost_vector = compute_treecost_vector(train_node, class_labels, graph,
            cost_type=cost_type, imbalance=imbalance)
    model = LogisticCost(rho=rho, w_n=w_n, w_pi=w_pi, 
        children=sum_c, mod_cn=mod_cn)
    model.fit(X, y, cost_vector)

    # save model
    safe_pickle_dump(model, outpath)
    end = time.time()
    print(" time= {:.3f} sec".format(end-start))


def main(args):
    '''
    Driver function to
        - parse command line argumnets.
        - train models for all input nodes.
    '''

    mkdir_if_not_exists(args.model_dir)

    if "graphml" not in args.hierarchy:
        graph = safe_read_graph(args.hierarchy)
    else:
        graph = safe_read_graphml(args.hierarchy)

    if "npy" not in args.dataset:
        X_train, labels_train = safe_read_svmlight_file_multilabel(
                 args.dataset, args.features)
    else:
        fe, ex = os.path.splitext(args.dataset)
        temp = "".join(fe.split("_")[0])
        label_path = "{}_labels{}".format(temp, ex)
        X_train, labels_train = read_embeddings(args.dataset, label_path)

    if args.nodes:
        train_node_list = [int(n) for n in args.nodes.split(",")]
    else:
        # all leaf nodes
        train_node_list = [n for n in graph.nodes() if len(list(graph.successors(n)))==0]
    
    features = X_train.shape[1]+1
    w_dict = init_recursive_regularizer(graph, features)
    
    labels_train = [i[0] for i in labels_train]
    set_train = list(set(labels_train))
    
    leaves = list(set(train_node_list).intersection(set_train))
    
    others = list(set(set_train).difference(leaves))
    
    top_level, internal_leaves = get_level_nodes(graph, others)
    top_level = [top_level]
    
    print("Root: {}\nInternal nodes: {}\nLeaves: {}".format(top_level, len(others), len(leaves)))
    
    labels_train = np.array(labels_train)
    
    print("Leaves: ")
    for train_node in tqdm(leaves):
        try:
            model_save_path = '{}/model_h_{}.p'.format(args.model_dir, train_node)
            
            w_n = w_dict[train_node]

            sum_c = np.zeros(2)

            mod_cn = np.zeros(2)

            for p in list(graph.predecessors(train_node)):
                w_pi = w_dict[p]

            updated_Wn = train_and_output_model(X_train, labels_train, train_node, graph,
                args.cost_type, args.imbalance, args.rho, model_save_path,
                w_pi, w_n, sum_c, mod_cn)

            w_dict[train_node] = updated_Wn

        except:
            print(train_node, "has no samples (X)")

    print("non-leaves")
    
    for level in internal_leaves:
        for node in tqdm(level):
            try:
                model_save_path = '{}/model_h_{}.p'.format(args.model_dir, node)
                
                parent = list(graph.predecessors(node))
                children = list(graph.successors(node))

                w_n = w_dict[node]

                mod_cn, sum_c, sum_pi = non_leaf_update(parent, children, w_dict, features)

                updated_Wn = train_and_output_model(X_train, labels_train, node, graph,
                args.cost_type, args.imbalance, args.rho, model_save_path,
                sum_pi, w_n, sum_c, mod_cn)

                w_dict[node] = updated_Wn
            except:
                    print(node, "has no samples (X)")

    
    print("root")
    for root in top_level:
        try:
            model_save_path = '{}/model_h_{}.p'.format(args.model_dir, root)

            parent = list(graph.predecessors(root))
            children = list(graph.successors(root))

            w_n = w_dict[root]

            mod_cn, sum_c, sum_pi = non_leaf_update(parent, children, w_dict, features)

            updated_Wn = train_and_output_model(X_train, labels_train, root, graph,
                args.cost_type, args.imbalance, args.rho, model_save_path,
                sum_pi, w_n, sum_c, mod_cn)

            w_dict[root] = updated_Wn
        except:
            print(root, "has no samples (X)")