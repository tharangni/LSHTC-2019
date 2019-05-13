'''
Train hierarchical flat classifier
using cost sensitive learning based on hierarchical costs
for hierarchical multi-label classification.

REF:
Anveshi Charuvaka and Huzefa Rangwala "HierCost: Improving Large Scale
Hierarchical Classification with Cost Sensitive Learning"  European Conference
on Machine Learning and Principles and Practice of Knowledge Discovery in
Databases, 2015
'''

import time
from dataset_util import * #pylint: disable=W0614
from recursive_reg import *
import numpy as np
import scipy.sparse
from joblib import Parallel, delayed
from sklearn.preprocessing import MultiLabelBinarizer
from logscut import LogisticScut
from mlcost import *

def train_and_output_model(X, y, class_labels, train_node, graph, cost_type,
    imbalance, rho, outpath, w_pi, w_n, sum_c, mod_cn):

    print("Training Model {} :  ".format(train_node))
    start = time.time()

    cost_vector = compute_treecost_vector(train_node, class_labels, 
        graph, cost_type=cost_type, imbalance=imbalance)
    model = LogisticScut(rho=rho, w_n=w_n, w_pi=w_pi, 
        children=sum_c, mod_cn=mod_cn)
    model.fit(X, y, cost_vector)

    # save model
    safe_pickle_dump(model, outpath)
    end = time.time()
    print(" time= {:.3f} sec".format(end-start))

    updated_Wn = model.W

    return updated_Wn

def training_main(train_node, label_matrix, lbin, X, labels_train, 
    graph, cost_type, imb, rho, model_dir):
    try:
        model_save_path = '{}/model_h_{}.p'.format(model_dir, train_node)
        y_node = label_matrix[:, lbin.classes_ == train_node].toarray().flatten()
        train_and_output_model(X, y_node, labels_train, train_node, 
                               graph, cost_type, imb, rho, 
                               model_save_path) 
    except:
        print(train_node, "has no samples (X)")


def main(args):

    mkdir_if_not_exists(args.model_dir)
    graph = safe_read_graph(args.hierarchy)
    X_train, labels_train = safe_read_svmlight_file_multilabel(
             args.dataset, args.features)
    lbin = MultiLabelBinarizer(sparse_output=True)
    label_matrix = lbin.fit_transform(labels_train)

    if args.nodes:
        train_node_list = [int(n) for n in args.nodes.split(",")]
    else:
        # all leaf nodes
        train_node_list = [n for n in graph.nodes() if len(list(graph.successors(n)))==0]

    features = X_train.shape[1]+1
    w_dict = init_recursive_regularizer(graph, features)

    # Parallel(n_jobs=4, prefer="threads")(delayed(training_main)(train_node, 
    #     label_matrix, lbin, X_train, labels_train, 
    #     graph, args.cost_type, args.imbalance, args.rho, 
    #     args.model_dir) for train_node in (train_node_list))    
    
    for train_node in train_node_list:
        try:
            model_save_path = '{}/model_h_{}.p'.format(args.model_dir, train_node)
            y_node = label_matrix[:, lbin.classes_ == train_node].toarray().flatten()
            
            w_n = w_dict[train_node]

            sum_c = np.zeros(2)

            mod_cn = np.zeros(2)

            for p in list(graph.predecessors(train_node)):
                print('Parent of {} node is: {}'.format(train_node, p))
                w_pi = w_dict[p]

            updated_Wn = train_and_output_model(X_train, y_node, 
                labels_train, train_node, graph, args.cost_type, 
                args.imbalance, args.rho, model_save_path,
                 w_pi, w_n, sum_c, mod_cn)

            w_dict[train_node] = updated_Wn

        except:
            print(train_node, "has no samples (X)")

    top_level, internal_leaves = get_level_nodes(graph, train_node_list)
    
    for level in internal_leaves:
        for node in level:
            try:
                model_save_path = '{}/model_h_{}.p'.format(args.model_dir, node)
                y_node = label_matrix[:, lbin.classes_ == node].toarray().flatten()
                parent = list(graph.predecessors(node))
                children = list(graph.successors(node))

                w_n = w_dict[node]

                mod_cn, sum_c, sum_pi = non_leaf_update(parent, children, w_dict)
                
                updated_Wn = train_and_output_model(X_train, y_node, 
                    labels_train, node, graph, args.cost_type, 
                    args.imbalance, args.rho, model_save_path,
                     sum_pi, w_n, sum_c, mod_cn)
                
                w_dict[node] = updated_Wn
            except:
                    print(node, "has no samples (X)")


    for root in top_level:
        try:
            model_save_path = '{}/model_h_{}.p'.format(args.model_dir, root)
            y_node = label_matrix[:, lbin.classes_ == root].toarray().flatten()

            parent = list(graph.predecessors(root))
            children = list(graph.successors(root))

            w_n = w_dict[root]

            mod_cn, sum_c, sum_pi = non_leaf_update(parent, children, w_dict)
            
            updated_Wn = train_and_output_model(X_train, y_node, 
                labels_train, root, graph, args.cost_type, 
                args.imbalance, args.rho, model_save_path,
                 sum_pi, w_n, sum_c, mod_cn)

            w_dict[root] = updated_Wn
        except:
            print(root, "has no samples (X)")
