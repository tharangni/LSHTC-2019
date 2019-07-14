import warnings
warnings.simplefilter("ignore")

import logging
logging.basicConfig(level=logging.INFO)

import os
import ast
import pandas as pd
import networkx as nx

from tqdm import tqdm
from fuzzywuzzy import fuzz
from gensim.parsing.preprocessing import *
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

def get_true_pred(true_file, pred_file, suffix):

    logging.info("---Processing prediction outputs---")
    true_list, pred_list = [], []
    true_text, pred_text = [], []

    with open(pred_file, "r") as fmain:
        reader = fmain.readlines()

    with open(true_file, "rb") as tmain:
        checker = tmain.readlines()

    strip_fn = lambda x: strip_multiple_whitespaces(strip_numeric(x).
        replace("(--) [.]", "").
        replace("(++) [.]", "").
        replace("__label__","")).replace(" ", "")

    strip_int_fn = lambda x: int(strip_multiple_whitespaces(x.
    replace("(--) [.]", "").
    replace("(++) [.]", "").
    replace("__label__","").replace(" ", "").replace("(--)", "").split("]")[-1]))

    
    rep_fn = lambda x: x.replace("__label__", "").replace("\n", "").replace(" ", "")
    
    rep_int_fn = lambda x: int(x.replace("__label__", "").replace("\n", "").replace(" ", ""))


    if "swiki" in pred_file:
        strip_fn = strip_int_fn
        rep_fn = rep_int_fn


    for i, line in enumerate(tqdm(checker)):
    
        instance = line.decode("utf-8").split(" ")
        true_labels = []
        sent = ""
        
        for s in instance:
            if "__label__" in s:
                true_labels.append(s)
            else:
                sent += "{} ".format(s)
        
        true_text.append(sent[:-1].strip())
        true = list(map(rep_fn, true_labels))
        true_list.append(true)
    

    for i, line in enumerate(tqdm(reader)):
        if "LHS" in line:
            pred_text.append(reader[i+1].strip())
            
        if "Predictions" in line:
            pred = list(map(strip_fn, reader[i+1:i+6]))
            pred_list.append(pred)
        

    print(pred_list[0])
    print(true_list[0])

    for i, (t1w, t2w) in enumerate(tqdm(zip(true_text, pred_text))):
        if fuzz.partial_ratio(t1w, t2w) < 50:
            print(i, "is not there in predictions, removing...")
            store_i = i
            true_list.pop(store_i)
            true_text.pop(store_i)
            break


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
    # predictions_df["true"] = predictions_df["true"].apply(lambda x: list(map(int, x))) 


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


    if "swiki" not in suffix:
        true_lol = true_int_list
        pred_lol = pred_int_list
    else:
        true_lol = true_list
        pred_lol = pred_list

    mlb = MultiLabelBinarizer(sparse_output=True)
    true_mat = mlb.fit_transform(true_list)
    pred_mat = mlb.transform(pred_list)

    truecat = "{}/true-{}.txt".format(pred_dir, suffix)
    predcat = "{}/pred-{}.txt".format(pred_dir, suffix)
    evalcat = "{}/eval-{}.txt".format(pred_dir, suffix)
    
    tfile = open(truecat, "w+")
    pfile = open(predcat, "w+")
    efile = open(evalcat, "w+")

    clf_report = classification_report(true_mat, pred_mat, target_names=mlb.classes_)
    uF1 = f1_score(true_mat, pred_mat, average="micro")
    mF1 = f1_score(true_mat, pred_mat, average="macro")
    p = precision_score(true_mat, pred_mat, average="samples")
    r = recall_score(true_mat, pred_mat, average="samples")

    print("Micro F1 score: {}\nMacro F1 score: {}\nPrecision score: {}\nRecall score: {}".
        format(uF1, mF1, p, r))

    efile.write("Precision score: {}\nRecall score: {}\nMicro F1 score: {}\nMacro F1 score: {}\n--------------\nClassification report: \n{}".
        format(p, r, uF1, mF1, clf_report))

    for t in true_lol:
        string = ""
        for p in t:
            string += "{} ".format(p)
        tfile.write("{}\n".format(string))

    for preds in pred_lol:
        string = ""
        for p in preds:
            string += "{} ".format(p)
        pfile.write("{}\n".format(string))

    efile.close()
    tfile.close()
    pfile.close()
    logging.info("---Fininshed!---")


if __name__ == '__main__':

    true_f = "C:/Users/harshasivajit/Documents/Starspace/data/oms/text/oms-test.txt"
    gml = "C:/Users/harshasivajit/Documents/Starspace/data/oms/cat_hier_TREE.graphml"
    mapper = get_mapper(gml)

    dict_of_items = {
    # "oms-d300-hless-pred" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d300-hless-pred.txt",
    # "oms-d400-hless-pred" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d400-hless-pred.txt",
    # "oms-d5-hless-pred" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d5-hless-pred.txt",
    # "oms-d10-hless-pred" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d10-hless-pred.txt",
    # "oms-d15-hless-pred" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d15-hless-pred.txt",
    # "oms-d20-hless-pred" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d20-hless-pred.txt",
    # "oms-d30-hless-pred" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d30-hless-pred.txt",
    # "oms-d40-hless-pred" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d40-hless-pred.txt",
    # "oms-d50-hless-pred" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d50-hless-pred.txt",
    # "oms-d60-hless-pred" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d60-hless-pred.txt",
    # "oms-d80-hless-pred" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d80-hless-pred.txt",
    # "oms-d100-hless-pred" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d100-hless-pred.txt",
    # "oms-d500-2-hless-pred" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d500-2-hless-pred.txt",
    # "oms-d5-neg-40-h-5-pred" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d5-neg-40-h-5-pred.txt",
    # "oms-d10-neg-40-h-5-pred" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d10-neg-40-h-5-pred.txt",
    # "oms-d15-neg-40-h-5-pred" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d15-neg-40-h-5-pred.txt",
    # "oms-d20-neg-40-h-5-pred" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d20-neg-40-h-5-pred.txt",
    # "oms-d30-neg-40-h-5-pred" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d30-neg-40-h-5-pred.txt",
    # "oms-d40-neg-40-h-5-pred" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d40-neg-40-h-5-pred.txt",
    # "oms-d50-neg-40-h-5-pred" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d50-neg-40-h-5-pred.txt",
    # "oms-d60-neg-40-h-5-pred" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d60-neg-40-h-5-pred.txt",
    # "oms-d80-neg-40-h-5-pred" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d80-neg-40-h-5-pred.txt",
    # "oms-d100-neg-40-h-5-pred" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d100-neg-40-h-5-pred.txt",
    "oms-d500-neg-40-h-5-pred" : "C:/Users/harshasivajit/Documents/Starspace/data/oms/pred/oms-d500-neg-40-h-5-pred.txt",
    }

    for suffix, file in dict_of_items.items():
        csv_file = get_true_pred(true_f, file, suffix)
        create_true_pred_output(mapper, csv_file, suffix)


    