'''
Test hierarchical flat classifier
using cost sensitive learning based on hierarchical costs
for hierarchical multi-label classification.

REF:
Anveshi Charuvaka and Huzefa Rangwala "HierCost: Improving Large Scale
Hierarchical Classification with Cost Sensitive Learning"  European Conference
on Machine Learning and Principles and Practice of Knowledge Discovery in
Databases, 2015
'''

import numpy as np
import networkx as nx
from sklearn.metrics import f1_score, classification_report
from dataset_util import *
from tqdm import tqdm
import scipy.sparse
from sklearn.preprocessing import MultiLabelBinarizer
import warnings

def pred_multilabel(X_test, model_dir, target_names):
    '''
    Predict class labels for test set for multi-label classification.

    Args:
        X_test (np.ndarray[num_examples x num_features]:float): test dataset features.
        model_dir (str): Directory containing pickled model files (ending with *.p)
                         belonging to class LogisticCost, with one *.p file per leaf node.
        target_names (np.ndarray[]:int):list of terminal class labels in graph.

    Returns:
        np.ndarray[num_examples x len(target_names)]: predicted labels for test dataset.

    '''

    num_examples = X_test.shape[0]
    y_pred = scipy.sparse.dok_matrix((num_examples, len(target_names)))
    target_names = target_names.astype(int)

    for idx, node in enumerate(target_names):
        model_save_path = '{}/model_h_{}.p'.format(
            model_dir, node)
        try:
            node_model = safe_pickle_load(model_save_path)
        except:
            node_model = None

        if node_model != None:
            node_pred = node_model.predict(X_test)
            y_pred[node_pred != 0, idx] = 1
        else:
            print("node model {}".format(node), "not found. empty predict")
                   
    return y_pred

def write_labels(out_path, labels):
    '''Output multi-label predictions'''
    with open(out_path,'w') as fout:
        for lbl in labels:
            out_str = ",".join([str(int(l)) for l in lbl])
            fout.write(out_str + "\n")

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
    
    lbin = MultiLabelBinarizer(sparse_output=True)
    y_test_mat = lbin.fit_transform(y_test)

    y_pred_mat = pred_multilabel(X_test, args.model_dir, lbin.classes_)
    y_pred = lbin.inverse_transform(y_pred_mat)
    write_labels(args.pred_path, y_pred)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # suppress UndefinedMetricWarning for macro_f1
        y_pred_mat = y_pred_mat.astype('bool').toarray()
        y_test_mat = y_test_mat.astype('bool').toarray()
        micro_f1 = f1_score(y_test_mat, y_pred_mat, average='micro')
        macro_f1 = f1_score(y_test_mat, y_pred_mat, average='macro')
        print("Micro-F1 = {:.5f}".format(micro_f1))
        print("Macro-F1 = {:.5f}".format(macro_f1))
        # print(classification_report(y_test_mat, y_pred_mat))