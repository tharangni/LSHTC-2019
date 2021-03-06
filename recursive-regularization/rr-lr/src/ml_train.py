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
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.preprocessing import MultiLabelBinarizer
from logscut import LogisticScut
from mlcost import *


def train_and_output_model(X, y, class_labels, train_node, graph, cost_type,
    imbalance, rho, outpath, w_pi, w_n, sum_c, mod_cn):

    
    if not os.path.isfile(outpath):
        print("Training Model {} :  ".format(train_node))
        start = time.time()

        cost_vector = np.ones(len(class_labels))
        model = LogisticScut(rho=rho, w_n=w_n, w_pi=w_pi, 
            children=sum_c, mod_cn=mod_cn)
        model.fit(X, y, cost_vector)

        safe_pickle_dump(model, outpath)
        end = time.time()
        print(" time= {:.3f} sec".format(end-start))
    else:
        model = safe_pickle_load(outpath)
        print("Loading existing model: {}".format(train_node))
        
    updated_Wn = model.W

    return updated_Wn

  
def train_fn(train_node, args, label_matrix, graph, X_train, labels_train, lbin):
    model_save_path = '{}/model_h_{}.p'.format(args.model_dir, train_node)
    model_dict_name = 'model_{}'.format(train_node)
    y_node = label_matrix[:, lbin.classes_ == train_node].toarray().flatten()

    w_n = w_dict[train_node]

    sum_c = np.zeros(2)

    mod_cn = np.zeros(2)

    for p in list(graph.predecessors(train_node)):
        w_pi = w_dict[p]

    updated_Wn = train_and_output_model(X_train, y_node, 
        labels_train, train_node, graph, args.cost_type, 
        args.imbalance, args.rho, model_save_path,
         w_pi, w_n, sum_c, mod_cn)

    w_dict[train_node] = updated_Wn
    


def main(args):
    
    global w_dict 
    
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

    lbin = MultiLabelBinarizer(sparse_output=True)
    label_matrix = lbin.fit_transform(labels_train)

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

    # top_level, internal_leaves = [], []
    
    print("Features: {}\nRoot: {}\nInternal nodes: {}\nLeaves: {}\nAll: {}".format(features-1, top_level, len(others), len(leaves), len(set_train)))

    
    Parallel(n_jobs=8, prefer="threads")(delayed(train_fn)(train_node, args, 
                                                            label_matrix, graph, X_train, 
                                                            labels_train, lbin) for train_node in tqdm(leaves))
    

    print("internal level")
    
    Parallel(n_jobs=8, prefer="threads")(delayed(train_fn)(node, args, 
                                                           label_matrix, graph, X_train, 
                                                           labels_train, lbin) for level in tqdm(internal_leaves) for node in level)