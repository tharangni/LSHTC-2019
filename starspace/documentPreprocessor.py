import os
import ast
import random
import logging
import numpy as np

from sklearn.model_selection import train_test_split

from gensim.parsing.porter import PorterStemmer
from gensim.parsing.preprocessing import preprocess_string

from collections import OrderedDict

from OmniscienceReader import *

import sys
sys.path.append('..')
logging.basicConfig(level=logging.INFO)

"""
docstring for documentPreprocessor

Preprocess the document files to the required fastText format for classification
Format: __label__2 , doc bla bla 
[] run the function only if file doesn't exist
[] uncomment to convert swiki test
[X] swiki   [X] oms
[x] removing xml tags (swiki)
[x] what to do for oms?
[x] normalize text (lowercase, notifying presence of spl char)
[x] the propcessed text file should be of the format: __label__xxx doc
[x] add extra space before and after every punctuation
[x] shuffle the sentences in the final stage before passing to CLI
[?] swiki: convert _ to - in the document text
---
[x] oms labels
[x] oms abstract + text =  doc
[x] replace _ to - in labels
[x] normalize similar to swiki
[x] split traint/val/test in the end
"""
def shuffle_lines(filename, mode):
    '''
    mode: default or binary
    '''
    if mode == "default":
        rm = 'r'
        wm = 'w'
    elif mode == "binary":
        rm = 'rb'
        wm = 'wb'
    else:
        logging.info("Incorrect mode")
         

    lines = open(filename, rm).readlines()
    random.shuffle(lines)
    open(filename, wm).writelines(lines)

def adding_hierarchy(filename, cat_hier_path, mode):

    fe, ex = os.path.splitext(filename)
    f_name = fe + "joint" + ex

    if mode == "default":
        rm = 'r'
        wm = 'w'
        am = 'a'
    elif mode == "binary":
        rm = 'rb'
        wm = 'wb'
        am = 'ab'
    else:
        logging.info("Incorrect mode")

    fin = open(cat_hier_path, rm)
    hier_data = fin.read()
    fin.close()

    fin2 = open(filename, rm)
    main_data = fin2.read()
    fin2.close()

    fout0 = open(f_name, wm)
    fout0.write(main_data)
    fout0.close()

    fout = open(f_name, am)
    fout.write(hier_data)
    fout.close


def create_hier_dict(filename):
    hdict = {}

    with open(filename, "r") as fmain:
        reader = fmain.readlines()

    for i, line in enumerate(reader):
        # line = line.decode('utf-8')     
        split_ = line.split("\t")
        child = split_[0]
        parent = split_[1].replace('\n', '')
        if child not in hdict:
            hdict[child] = parent
        else:
            print("oops node {} already exists".format(child))

    return hdict


##### SWIKI #####

def swiki_replacer(lines):

    lines = lines.replace("<DOC>", "")
    lines = lines.replace("</DOC>", "")
    lines = lines.replace("<DOCNO>", "THISISADOC")
    lines = lines.replace("</DOCNO>", "")
    lines = lines.replace("<LABELS>", "OMGTHISSHOULDBEARAREWORD")
    lines = lines.replace("</LABELS>", "")
    
    lines = lines.replace(".", " . ")
    lines = lines.replace("?", " ? ")
    lines = lines.replace("!", " ! ")
    lines = lines.replace(")", " ) ")
    lines = lines.replace("(", " ( ")
    
    return lines

def document_preprocess(text):
    p = PorterStemmer()
    first = text.encode('ascii', 'ignore').decode('utf-8').lower()
    second = preprocessing.remove_stopwords(first)
    third = preprocessing.strip_punctuation(second)
    fourth = preprocessing.strip_short(preprocessing.strip_numeric(third))
    fifth = p.stem(fourth)
    return fifth

def clean_up_swiki_test(filename):


    fe, ex = os.path.splitext(filename)
    new_f = "{}_fasttext{}".format(fe, ex)
    
    with open(filename, "r") as fmain:
        reader = fmain.readlines()

    wmain = open(new_f, "w+")

    for i, lines in enumerate(tqdm(reader)):
        one = lines.replace("  ", " ").replace("\n", "")
        if len(lines) > 2 :
            umm_split = one.split("$")
            labs = umm_split[0]
            labs = ''.join(labs)
            labs = labs.replace("[", "").replace("'", "").replace("]", "").replace(",", "")
            the_rest = umm_split[1:]
            str_wut = ''.join(the_rest)
            wmain.write("{} {}\n".format(labs, str_wut))
    wmain.close()
    
    os.remove(filename)
    return new_f


def tagging_adder(label_list):

    if isinstance(label_list, list):
        temp = []
        for item in label_list:
            temp_str = "__label__{}".format(item)
            temp.append(temp_str)
    else:
        temp = "__label__{}".format(label_list)
    
    return temp

def swiki_converter(filename):
    
    cat_hier_path = "../../../Starspace/data/swiki/cat_hier_rev_fasttext.txt"
    cat_path = "../../../Starspace/data/swiki/cat_hier_inplace.txt"
    hdict = create_hier_dict(cat_hier_path)

    logging.info("--Beginning conversion for swiki--")
    fe, ex = os.path.splitext(filename)
    new_f = "{}_level1{}".format(fe, ex)
    new_h = "{}_singleL{}".format(fe, ex)
    csv_f = "{}_docs.csv".format(fe)
    true_preds = "../../../Starspace/data/swiki/text/smallGS"


    long_labels = []

    with open(true_preds, "r") as ytrue:
        true_p_reader = ytrue.readlines()

    for line in true_p_reader:
        lsplit = line.strip().split(' ')
        long_labels.append(lsplit)


    with open(filename, "r") as fmain:
        reader = fmain.readlines()
    
    labels = []
    doc_no = []
    raw = []
    content = []
    long_docs = []
    long_labelz = []

    for i, lines in enumerate(tqdm(reader)):

        fmt = swiki_replacer(lines)

        if "OMGTHISSHOULDBEARAREWORD" in fmt:
            fmt = fmt.strip().replace("OMGTHISSHOULDBEARAREWORD", "")
            fmt = fmt.split(' ')
            tmpt = list(map(int, fmt))
            labels.append(tmpt)
            pass

        if "THISISADOC" in fmt:
            fmt = fmt.strip().replace("THISISADOC", "")
            fmt = fmt.split(' ')
            doc_no.append(fmt)
            pass

        if (i+1)%5 == 0:
            fmt = document_preprocess(fmt)
            # fmt_str = " ".join(fmt)
            content.append(fmt)                 
            
            if len(labels[-1]) > 0:
              for k in labels[-1]:
                  long_labelz.append(k)
                  long_docs.append(content[-1])



    print(len(labels))
    print(len(content))
    # print(len(long_labelz))
    # print(len(long_docs))



    g = pd.DataFrame(columns = ["label", "doc"])
    h = pd.DataFrame(columns = ["label", "doc"])

    g["label"] =  labels #long_labels
    g["doc"] = content #long_docs    
    g["label"] = g["label"].apply(lambda x: tagging_adder(x))    
    
    h["label"] = long_labelz
    h["doc"] = long_docs
    h["label"] = h["label"].apply(lambda x: int(x))
    h["label"] = h["label"].apply(lambda x: tagging_adder(x))
    
    tag_labels = list(g["label"])
    w_ = open(cat_path, "w+")

    checker = []
    for item in tqdm(tag_labels):
        for j in item:
            if j not in checker:
                checker.append(j)
                string = "{} {}".format(j, hdict[j])
                w_.write(string)

    w_.close()
    print(len(checker))
    del checker

    # g.to_csv(csv_f, header=True, index=False)
    g.to_csv(new_f, header=False, index=False, sep='$', quotechar=' ')

    # h.to_csv(csv_f, header=True, index=False)
    h.to_csv(new_h, header=False, index=False, sep='$', quotechar=' ')
    # g.to_csv(new_f, header=False, index=False, sep=',', quotechar=' ')

    # temp_df = g
    # temp_df.to_csv(csv_f, index=False)
    
    new_file = clean_up_swiki_test(new_f)
    new_file2 = clean_up_swiki_test(new_h)

    logging.info("--Finished converting swiki to fasttext format--")
    
    logging.info("--Shuffling lines in the file--")
    shuffle_lines(new_file, "default")
    shuffle_lines(new_file2, "default")
    logging.info("--Finished shuffling lines--")
    if "test" not in new_file:
        adding_hierarchy(new_file, cat_path, "default")
        adding_hierarchy(new_file2, cat_hier_path, "default")
        logging.info("--Added hierarchy data--")


##### OMS #####

def oms_replacer(data):
    return data.lower().replace(".", " . ").replace("(", " ( ").replace(",", " ,").replace(")", " ) ")

def oms_tagger(label_list):
    if isinstance(label_list, list):
        temp = []
        for item in label_list:
            item = item.replace("_","-")
            temp_str = "__label__{}".format(item)
            temp.append(temp_str)
        return temp
    else:
        return "__label__{}".format(label_list)


def oms_converter(df, main_path, split_name):
    
    fe, ex = os.path.splitext(main_path)
    
    save_file_as = "{}-oms-{}_fasttext.txt".format(fe, split_name)
    save_rel_as = "{}-oms-{}_rel.txt".format(fe, split_name)
    cat_hier_path = "../../../Starspace/data/oms/cat_hier_dag2tree_rev_fasttext.txt"
    cat_path = "../../../Starspace/data/oms/cat_hier_inplace.txt"
    hdict = create_hier_dict(cat_hier_path)

    new_df = pd.DataFrame(columns=["label", "document"])
    new_df["document"] = df["doc"]
    # new_df["document"] = new_df["document"].apply(lambda x: oms_replacer(x))
    new_df["label"] = df["labels"].apply(lambda x: oms_tagger(x))
        
    temp_dict = new_df.to_dict(orient='list')
    temp = list(temp_dict.values())
    
    all_labels = temp[0]
    all_content = temp[1]
    label = []
    content = []

    print(split_name)
    print(len(all_labels))
    print(len(all_content))
    print(all_labels[1])
    logging.info("--Beginning to convert {} mode to fasttext format--".format(split_name))

    # regular saving
    wmain = open(save_file_as, "w+")
    rmain = open(save_rel_as, "w+")
    checker = []

    for lset in all_labels:
        if len(lset) == 1:
            checker.append(lset[0])
        else:
            for l in lset:
                checker.append(l)

    checker = list(set(checker))
    try:
        checker.remove("__label__science")
    except:
        print("already removed science")

    for i, item in tqdm(enumerate(all_labels)):
        string = ""
        for j in item:
            string += j + " "
            line_one = "{} {}\n".format(j, all_content[i])
            rmain.write(line_one)
        line = "{} {}\n".format(string[:-1], all_content[i])

        wmain.write(line)
    rmain.close()
    wmain.close()
    
    print(len(checker))
    
    #1. if pi not in  all labels, then just add it as an empty label to 
    # hless data. # labels w/ h == # labels w/o h

    #2. selecting only those parent nodes which have samples/instances
    # for h-data => # labels w/o h == # labels w/ h
    cmain = open(cat_path, "w+")
    temp_ = {}
    for item in checker:
        temp_[item] = hdict[item]

    for k, v in tqdm(temp_.items()):
        if v in checker:
            c_string = "{} {}\n".format(k, v)
            cmain.write(c_string)
        else:
            pass

    cmain.close()
    

    # shuffling
    shuffle_lines(save_file_as, "default")
    if split_name == "train":
        adding_hierarchy(save_file_as, cat_path, "default")
        adding_hierarchy(save_rel_as, cat_path, "default")
        logging.info("--Added hierarchy data--")
    
    logging.info("--Finished converting {} mode to fasttext--".format(split_name))


def oms_main(main_path):

    logging.info("--Reading CSV--")
    O = OmniscienceReader(main_path)
    groups = O.om_df.groupby("used_as")

    training = groups.get_group("training")
    test = groups.get_group("validation")
    print(training.shape)
    print(test.shape)

    train, validation = train_test_split(training, test_size=0.2)

    stages = {'train': train, 'valid': validation, 'test': test}

    for split, stage_df in stages.items():
        oms_converter(stage_df, main_path, split)


if __name__ == '__main__':
    swiki_converter("../../../Starspace/data/swiki/text/swiki-train.txt")
    # swiki_converter("../../../Starspace/data/swiki/text/swiki-test.txt")
    # oms_main("../../../Starspace/data/oms/text/oms-prep.tsv")
