import warnings
warnings.simplefilter("ignore")

import logging
logging.basicConfig(level=logging.INFO)

import os
import ast
import pandas as pd
import networkx as nx

from tqdm import tqdm
from gensim.parsing.preprocessing import *
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score

def get_true_pred(true_file, pred_file, suffix):

    logging.info("---Processing prediction outputs---")
    true_list, pred_list = [], []

    with open(pred_file, "r") as fmain:
        reader = fmain.readlines()

    with open(true_file, "rb") as tmain:
        checker = tmain.readlines()

    strip_fn = lambda x: strip_multiple_whitespaces(strip_numeric(x).
        replace("(--) [.]", "").
        replace("(++) [.]", "").
        replace("__label__","")).replace(" ", "")

    rep_fn = lambda x: x.replace("__label__", "").replace("\n", "").replace(" ", "")

    for i, line in enumerate(tqdm(checker)):
        instance = line.decode("utf-8").split(" ")
        true_labels = [s for s in instance if "__label__" in s]
        true = list(map(rep_fn, true_labels))
        true_list.append(true)
    
    true_list.pop(8207)

    for i, line in enumerate(tqdm(reader)):
        if "Predictions" in line:
            pred = list(map(strip_fn, reader[i+1:i+6]))
            pred_list.append(pred)

    print(pred_list[0])
    print(true_list[0])
    assert len(true_list) == len(pred_list), "incorrect: true {} != pred len {}".format(len(true_list), len(pred_list))

    csv_file = "{}/true_pred-{}.csv".format(os.path.dirname(pred_file), suffix)
    if not os.path.isfile(csv_file):
        predictions_df = pd.DataFrame(columns=["true", "pred"])
        predictions_df["true"] = true_list
        predictions_df["pred"] = pred_list
        predictions_df.to_csv(csv_file, index=False)
    
    return csv_file


def get_mapper(gml):

    int_gpath = "{}/cat_hier_int.txt".format(os.path.dirname(gml))
    
    read_g = nx.read_graphml(gml)

    str_nodes = list(read_g.nodes())
    int_g = nx.convert_node_labels_to_integers(read_g)
    int_nodes = list(int_g.nodes())

    assert len(str_nodes) == len(int_g), "str to int conversion incorrect"

    flag = nx.faster_could_be_isomorphic(read_g, int_g)
    logging.info("Isomorphic check: {}".format(flag) )

    mapper = {}

    for i, node in enumerate(str_nodes):
        mapper[node] = i

    if not os.path.isfile(int_gpath):
        nx.write_edgelist(int_g, int_gpath)

        with open(int_gpath, "r") as fmain:
            reader = fmain.readlines()

        file_str = ""

        for i, lines in enumerate(reader):
            line = lines.strip().replace("{'weight': 1}", "")
            line = lines.strip().replace("{}", "")

            file_str += "{}\n".format(line)

        with open(int_gpath, "w") as fmain:
            fmain.write(file_str)

    return mapper


def create_true_pred_output(mapper, csv_file, suffix):
    
    pred_dir = os.path.dirname(csv_file)

    list_converter = lambda x: mapper[x]

    predictions_df = pd.read_csv(csv_file)

    predictions_df["pred"] = predictions_df["pred"].apply(lambda x: ast.literal_eval(x)) 
    predictions_df["true"] = predictions_df["true"].apply(lambda x: ast.literal_eval(x)) 

    true_list = list(predictions_df["true"])
    pred_list = list(predictions_df["pred"])
    
    logging.info("---Writing to files for {}---".format(suffix))

    # mapping string to int
    pred_int_list, true_int_list = [], []
    for tlist in true_list:
        temp = list(map(list_converter, tlist))
        true_int_list.append(temp)

    for plist in pred_list:
        temp = list(map(list_converter, plist))
        pred_int_list.append(temp)

    # Multilabel binarizer for uF1 and mF1
    # lol = list of list 

    mlb = MultiLabelBinarizer()

    true_lol = true_int_list
    pred_lol = pred_int_list

    pred_mat = mlb.fit_transform(pred_lol)
    true_mat = mlb.transform(true_lol)

    truecat = "{}/true-{}.txt".format(pred_dir, suffix)
    predcat = "{}/pred-{}.txt".format(pred_dir, suffix)
    evalcat = "{}/eval-{}.txt".format(pred_dir, suffix)
    
    tfile = open(truecat, "w+")
    pfile = open(predcat, "w+")
    efile = open(evalcat, "w+")

    uF1 = f1_score(true_mat, pred_mat, average="micro")
    mF1 = f1_score(true_mat, pred_mat, average="macro")
    p = precision_score(true_mat, pred_mat, average="samples")
    r = recall_score(true_mat, pred_mat, average="samples")

    print("Micro F1 score: {}\nMacro F1 score: {}\nPrecision score: {}\nRecall score: {}".
        format(uF1, mF1, p, r))

    efile.write("Precision score: {}\nRecall score: {}\nMicro F1 score: {}\nMacro F1 score: {}".
        format(p, r, uF1, mF1))

    for t in true_int_list:
        tfile.write("{} \n".format(t))

    for preds in pred_int_list:
        string = ""
        for p in preds:
            string += "{} ".format(p)
        pfile.write("{}\n".format(string))

    efile.close()
    tfile.close()
    pfile.close()
    logging.info("---Fininshed!---")


if __name__ == '__main__':

    true_f = "C:/Users/harshasivajit/Documents/Starspace/data/oms/text/oms-valid.txt"
    gml = "C:/Users/harshasivajit/Documents/Starspace/data/oms/cat_hier_dag2tree.graphml"
    mapper = get_mapper(gml)

    dict_of_items = {
    # "d64h" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d128v-h-pred.txt",
    "d128-init-h" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d128-init-htc-pred.txt",
    "d128-init-hless" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d128-init-hless-pred.txt",
    
    }

    for suffix, file in dict_of_items.items():
        csv_file = get_true_pred(true_f, file, suffix)
        create_true_pred_output(mapper, csv_file, suffix)


    