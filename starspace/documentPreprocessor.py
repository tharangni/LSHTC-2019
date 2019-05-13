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
			wmain.write("{}, {} \n".format(labs, str_wut))
	wmain.close()
	
	os.remove(filename)
	return new_f


def tagging_adder(label_list):
	
	temp = []
	for item in label_list:
		temp_str = "__label__Q{}R".format(item)
		temp.append(temp_str)
	
	return temp

def swiki_converter(filename):
	
	logging.info("--Beginning conversion for swiki--")
	cat_hier_path = "../../../Starspace/data/swiki/cat_hier_fasttext.txt"
	fe, ex = os.path.splitext(filename)
	new_f = "{}_level1{}".format(fe, ex)
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
	content = []
	long_docs = []

	for i, lines in enumerate(tqdm(reader)):

		fmt = swiki_replacer(lines)

		if "OMGTHISSHOULDBEARAREWORD" in fmt:
			fmt = fmt.strip().replace("OMGTHISSHOULDBEARAREWORD", "")
			fmt = fmt.split(' ')
			labels.append(fmt)
			pass

		if "THISISADOC" in fmt:
			fmt = fmt.strip().replace("THISISADOC", "")
			fmt = fmt.split(' ')
			doc_no.append(fmt)
			pass

		if (i+1)%5 == 0:
			fmt = fmt.lower()
			content.append(fmt)                 
			
			# if len(labels[-1]) > 0:
			# 	for k in labels[-1]:
			# 		long_labels.append(k)
			# 		long_docs.append(doc_no[-1])

		# if len(fmt) > 5:
		# 	fmt = fmt.lower()
		# 	content.append(fmt)                 

	# del labels
	# del doc_no

	print(len(labels))
	print(len(content))

	g = pd.DataFrame(columns = ["label", "doc"])
	g["label"] = labels #long_labels    
	g["doc"] = content    
	g["label"] = g["label"].apply(lambda x: tagging_adder(x))    
	g.to_csv(new_f, header=False, index=False, sep='$', quotechar=' ')
	# g.to_csv(new_f, header=False, index=False, sep=',', quotechar=' ')

	# temp_df = g
	# temp_df.to_csv(csv_f, index=False)
	
	new_file = clean_up_swiki_test(new_f)
	# new_file = clean_up(new_f)
	logging.info("--Finished converting swiki to fasttext format--")
	
	logging.info("--Shuffling lines in the file--")
	shuffle_lines(new_file, "default")
	logging.info("--Finished shuffling lines--")
	if "test" not in new_file:
		adding_hierarchy(new_file, cat_hier_path, "default")
		logging.info("--Added hierarchy data--")


##### OMS #####

def oms_replacer(data):
	return data.lower().replace(".", " . ").replace("(", " ( ").replace(",", " ,").replace(")", " ) ")

def oms_tagger(label_list):
	temp = []
	for item in label_list:
		item = item.replace("_","-")
		temp_str = "__label__{}".format(item)
		temp.append(temp_str)
	return temp

def oms_converter(df, main_path, split_name):
	
	fe, ex = os.path.splitext(main_path)
	
	save_file_as = "{}-oms-{}_fasttext.txt".format(fe, split_name)
	cat_hier_path = "../../../Starspace/data/oms/cat_hier_outbound_fasttext.txt"
	
	new_df = pd.DataFrame(columns=["label", "document"])
	new_df["document"] = df["title"] + " : " + df["abstract"]
	new_df["document"] = new_df["document"].apply(lambda x: oms_replacer(x))
	new_df["label"] = df["omniscience_labels"]
	new_df["label"] = new_df["label"].apply(lambda x: oms_tagger(x))
		
	temp_dict = new_df.to_dict(orient='list')
	temp = list(temp_dict.values())
	
	all_labels = temp[0]
	all_content = temp[1]
	label = []
	content = []

	print(len(all_labels))
	print(len(all_content))
	print(all_labels[1])
	logging.info("--Beginning to convert {} mode to fasttext format--".format(split_name))

	# regular saving
	wmain = open(save_file_as, "wb+")
	for i, item in enumerate(all_labels):
		string = ""
		for j in item:
			string += "{} ".format(j)

		line = "{}, {}\n".format(string, all_content[i])
		wmain.write(line.encode("utf-8"))
	wmain.close()
	
	# shuffling
	shuffle_lines(save_file_as, "binary")
	if split_name != "test":
		adding_hierarchy(save_file_as, cat_hier_path, "binary")
		logging.info("--Added hierarchy data--")
	
	logging.info("--Finished converting {} mode to fasttext--".format(split_name))


def oms_main(main_path):

	logging.info("--Reading CSV--")
	O = OmniscienceReader(main_path)
	groups = O.om_df.groupby("used_as")

	training = groups.get_group("training")
	validation = groups.get_group("validation")
	testing = groups.get_group("unused")
	print(training.shape)
	print(validation.shape)
	print(testing.shape)

	stages = {'train': training, 'valid': validation, 'test': testing}

	for split, stage_df in stages.items():
		oms_converter(stage_df, main_path, split)


if __name__ == '__main__':
	# swiki_converter("../../../Starspace/data/swiki/text/swiki-train.txt")
	# swiki_converter("../../../Starspace/data/swiki/text/swiki-test.txt")
	oms_main("../../../Starspace/data/oms/text/oms.tsv")
