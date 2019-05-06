import os
import random
import logging
import numpy as np

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

##### SWIKI #####

def swiki_replacer(lines):

	lines = lines.replace("<DOC>", "")
	lines = lines.replace("</DOC>", "")
	lines = lines.replace("<DOCNO>", "")
	lines = lines.replace("</DOCNO>", "")
	lines = lines.replace("<LABELS>", "OMGTHISSHOULDBEARAREWORD")
	lines = lines.replace("</LABELS>", "")
	
	lines = lines.replace(".", " . ")
	lines = lines.replace("?", " ? ")
	lines = lines.replace("!", " ! ")
	lines = lines.replace(")", " ) ")
	lines = lines.replace("(", " ( ")
	
	return lines


def clean_up(filename):
	
	fe, ex = os.path.splitext(filename)
	new_f = "{}_fasttext{}".format(fe, ex)
	
	with open(filename, "r") as fmain:
		reader = fmain.readlines()
		
	wmain = open(new_f, "w+")
	for i, lines in enumerate(tqdm(reader)):
		one = lines.replace("  ", " ").replace("\n", "")
		if len(lines) > 2:
			umm_split = one.split(",")
			labs = umm_split[0]
			therest = umm_split[1:]
			str_wut = ''.join(therest)
			wmain.write("{} ,{} \n".format(labs, str_wut))
	wmain.close()
	
	os.remove(filename)
	
	return new_f


def swiki_converter(filename):
	
	logging.info("--Beginning conversion for swiki--")
	
	fe, ex = os.path.splitext(filename)
	new_f = "{}_level1{}".format(fe, ex)
	
	with open(filename, "r") as fmain:
		reader = fmain.readlines()
	
	labels = []
	content = []
	long_labels = []

	for i, lines in enumerate(tqdm(reader)):

		fmt = swiki_replacer(lines)

		if "OMGTHISSHOULDBEARAREWORD" in fmt:
			fmt = fmt.strip().replace("OMGTHISSHOULDBEARAREWORD", "")
			fmt = fmt.split(' ')
			labels.append(fmt)

		if len(fmt) > 50:
			fmt = fmt.lower()
			if len(labels[-1]) > 0:
				for k in labels[-1]:
					long_labels.append(k)
					content.append(fmt)                    
	del labels
	
	g = pd.DataFrame(columns = ["label", "doc"])
	g["label"] = long_labels    
	g["doc"] = content    
	g["label"] = g["label"].apply(lambda x: "__label__{}".format(x))    
	g.to_csv(new_f, header=False, index=False, sep=',', quotechar=' ')
	
	new_file = clean_up(new_f)
	logging.info("--Finished converting swiki to fasttext format--")
	
	logging.info("--Shuffling lines in the file--")
	shuffle_lines(new_file, "default")
	logging.info("--Finished shuffling lines--")


##### OMS #####

def oms_replacer(data):
	return data.lower().replace(".", " . ").replace("(", " ( ").replace(",", " ,").replace(")", " ) ")


def oms_converter(df, main_path, split_name):
	
	fe, ex = os.path.splitext(main_path)
	
	save_file_as = "{}-oms-{}_fasttext.txt".format(fe, split_name)
	
	new_df = pd.DataFrame(columns=["label", "document"])
	new_df["document"] = df["title"] + " : " + df["abstract"]
	new_df["document"] = new_df["document"].apply(lambda x: oms_replacer(x))
	new_df["label"] = df["omniscience_labels"]
		
	temp_dict = new_df.to_dict(orient='list')
	temp = list(temp_dict.values())
	
	all_labels = temp[0]
	all_content = temp[1]
	label = []
	content = []

	logging.info("--Beginning to convert {} mode to fasttext format--".format(split_name))
	for i in range(len(all_labels)):
		if len(all_labels[i]) > 1:
			for j in all_labels[i]:
				label.append(j.replace("_", "-"))
				content.append(all_content[i])
		else:
			label.append(all_labels[i][0].replace("_", "-"))
			content.append(all_content[i])
			
	# regular saving
	wmain = open(save_file_as, "wb+")
	for i in range(len(label)):
		line = "__label__{} , {}\n".format(label[i], content[i])
		wmain.write(line.encode("utf-8"))
	wmain.close()
	
	# shuffling
	shuffle_lines(save_file_as, "binary")
	
	logging.info("--Finished converting {} mode to fasttext--".format(split_name))


def oms_main(main_path):

	logging.info("--Reading CSV--")

	O = OmniscienceReader(main_path)

	groups = O.om_df.groupby("used_as")

	training = groups.get_group("training")
	validation = groups.get_group("validation")
	testing = groups.get_group("unused")

	stages = {'train': training, 'valid': validation, 'test': testing}

	for split, stage_df in stages.items():
		oms_converter(stage_df, main_path, split)


if __name__ == '__main__':
	swiki_converter("../../../Starspace/data/raw-swiki/text/swiki-train.txt")
	swiki_converter("../../../Starspace/data/raw-swiki/text/swiki-test.txt")
	oms_main("../../../Starspace/data/oms/text/oms.tsv")