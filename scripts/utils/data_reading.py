import os
import time
import torch
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm 
from pathlib import Path
from joblib import Memory
from random import sample

from collections import Counter, OrderedDict
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import load_svmlight_file


mem = Memory("./../../mycache_getdata")
@mem.cache
def lower_dim(file_path, reduce, n_components):

	data = get_data(file_path)
	new_data = data[0]

	if reduce:
		svd = TruncatedSVD(n_components=n_components, n_iter=10, random_state=None)
		current_time = time.time()
		new_data = svd.fit_transform(data[0])
		elapsed_time = time.time() - current_time
		min_, sec_ = divmod(elapsed_time, 60)
		logging.info("Elapsed time: {}min {:.2f}sec".format(min_, sec_))

	new_doc, new_labels = [], []
	
	for i, label_tuple in enumerate(data[1]):
		for each_label in label_tuple:
			new_doc.append(new_data[i])
			new_labels.append(int(each_label))

	new_doc = np.stack(new_doc, axis=0)
	
	return new_doc, new_labels


mem = Memory("./../../mycache_getdata")
@mem.cache
def get_data(filename):
	
	fname = str(Path(filename))
	fe, ex = os.path.splitext(fname) 

	try:
		data = load_svmlight_file(fname, multilabel=True)
	except:
		# Required: if the input data isn't in the correct libsvm format
		outfile = str(Path("{}_small{}".format(fe, ex)))
#         outfile = str(Path("{}_remapped{}".format(fe, ex)))
		if not os.path.isfile(outfile):
			logging.info("Remapping data to LibSVM format...")
			f = preprocess_libsvm(fname, outfile)
		else:
			logging.info("Using already remapped data...")
			f = outfile
		data = load_svmlight_file(f, multilabel=True)

	return data[0], data[1]


mem = Memory("./mycache")
@mem.cache
def rr_reader(filename):
	'''
	create a dataframe from the data-label pair
	'''

	num_entries = 200000
	df = pd.DataFrame()
	feat_dict = {}
	freq = 5

	with open(filename, "r") as f:
#         head = [next(f) for x in range(num_entries)] # retrieve only `n` docs
		for i, line in enumerate(tqdm(f)): # change to f/head depending on your needs
			instance = line.strip().split()
			labels = instance[0]
			doc_dict = OrderedDict()
			temp_dict = {}

			for pair in instance[1:]:
				feat = pair.split(":")
				if int(feat[0]) not in temp_dict:
					temp_dict[int(feat[0])] = int(feat[1])

			for key in sorted(temp_dict.keys()):
				doc_dict[key] = temp_dict[key]
			
			for k, v in doc_dict.items():
				if k in feat_dict:
					feat_dict[k] += v
				else:
					feat_dict[k] = v

			temp_df = pd.DataFrame(data = [ labels, doc_dict ]).T
			df = df.append(temp_df, ignore_index=True)
	
	df.columns = ["labels", "feat_tf"]
	df["labels"] = df["labels"].apply( lambda x: list(map(int, x.split(",")))  )

	reduced_corpus = {}
	for k, v in feat_dict.items():
		if v > freq:
			reduced_corpus[k] = v

	return df, feat_dict, reduced_corpus

# !run this only once - this is just to create a smaller train-valid set
# x, y = train_valid_split("swiki/data/train_remapped.txt")
def train_valid_split(input_file):
	
	fname = str(Path(input_file))
	fe, ex = os.path.splitext(fname) 
	fe = 'swiki/data/valid'
	
	outfile = str(Path("{}_remapped{}".format(fe, ex)))
	
	output_valid = outfile
	output_train_v = 'swiki/data/train_split_remapped.txt'
	
	ratio = 0.7
	with open(input_file, "r") as f:
		line = f.readlines()
	
	train_size = int(len(line)*ratio)
	valid_size = int(len(line) - train_size)
	
	print(train_size, valid_size)
	
	valid_samples = sample(range(len(line)), valid_size)
	
	all_samples = list(range(len(line)))
	train_samples = list(set(all_samples).difference(set(valid_samples)))
			
	print(len(valid_samples), len(train_samples))
	file = open(output_valid, "w+")
	for i in valid_samples:
		instance = line[i].strip().split()
		labels = instance[0]
		doc_dict = OrderedDict()
		temp_dict = {}
		temp_string = ''

		for pair in instance[1:]:
			feat = pair.split(":")
			if int(feat[0]) not in temp_dict:
				temp_dict[int(feat[0])] = int(feat[1])

		for key in sorted(temp_dict.keys()):
			doc_dict[key] = temp_dict[key]

		for feat, tf in doc_dict.items():
			temp_string = temp_string + "{}:{} ".format(feat, tf)        
		file.write("{} {}\n".format(labels, temp_string))
	file.close()
	
	file = open(output_train_v, "w+")
	for i in train_samples:
		instance = line[i].strip().split()
		labels = instance[0]
		doc_dict = OrderedDict()
		temp_dict = {}
		temp_string = ''

		for pair in instance[1:]:
			feat = pair.split(":")
			if int(feat[0]) not in temp_dict:
				temp_dict[int(feat[0])] = int(feat[1])

		for key in sorted(temp_dict.keys()):
			doc_dict[key] = temp_dict[key]

		for feat, tf in doc_dict.items():
			temp_string = temp_string + "{}:{} ".format(feat, tf)        
		file.write("{} {}\n".format(labels, temp_string))
	file.close()

	return train_samples, valid_samples