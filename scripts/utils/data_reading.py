import os
import ast # required for Omniscience
import time
import torch
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm 
from pathlib import Path
from joblib import Memory

from collections import Counter, OrderedDict
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import load_svmlight_file



mem = Memory("./../../mycache_getdata")
@mem.cache
def lower_dim(file_path, reduce, n_components):

	data = get_data(file_path)
	new_data = data[0]

	if reduce:
		new_data = call_svd(data[0], n_components)

	new_doc, new_labels = new_data, data[1]
	
	return new_doc, new_labels


mem = Memory("./../../mycache_getdata")
@mem.cache
def call_svd(data, n_components):
	svd = TruncatedSVD(n_components=n_components, n_iter=10, random_state=None)
	current_time = time.time()
	new_data = svd.fit_transform(data)
	elapsed_time = time.time() - current_time
	min_, sec_ = divmod(elapsed_time, 60)
	logging.info("Elapsed time: {}min {:.2f}sec".format(min_, sec_))

	return new_data


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
		if not os.path.isfile(outfile):
			logging.info("Remapping data to LibSVM format...")
			f = preprocess_libsvm(fname, outfile)
		else:
			logging.info("Using already remapped data...")
			f = outfile
		data = load_svmlight_file(f, multilabel=True)

	return data[0], data[1]



def preprocess_libsvm(input_file, output_file):
	# converts file to the required libsvm format.
	# this is very brute force but can be made faster [IMPROVE]

	file = open(output_file, "w+")
	with open(input_file, "r") as f:
		head = [next(f) for x in range(500)] # retrieve only `n` docs
		for i, line in enumerate(tqdm(head)): # change to f/head depending on your needs
			instance = line.strip().split()
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

	return output_file



class LIBSVM_Reader(object):
	"""
	docstring for LIBSVM_Reader
	[x] dim reduction
	[x] caching
	[x] return df with <doc id, doc labels, doc vec>
	[] reverse mapping df with {label_id -> doc vec}
	[] create similar for raw text also
	[] distribution of classes per document

	"""
	def __init__(self, file_path, reduce, n_components):
		super(LIBSVM_Reader, self).__init__()
		self.file_path = file_path
		self.reduce = reduce
		self.n_components = n_components
		self.view_df()

	def view_df(self):
		all_data = lower_dim(self.file_path, self.reduce, self.n_components)
		self.data_df = pd.DataFrame(columns = ["doc_id", "doc_labels", "doc_vector"])
		self.data_df["doc_labels"] = all_data[1]    
		self.data_df["doc_labels"] = self.data_df["doc_labels"].apply(lambda x: list(map(int, x)))  
		for i in tqdm(self.data_df):
			self.data_df.at[i, "doc_vector"] = torch.as_tensor(all_data[0][i], dtype=torch.float32)
			self.data_df.at[i, "doc_id"] = i
		return self.data_df




class OmniscienceReader(object):
	"""docstring for OmniscienceReader"""
	def __init__(self, file_path):
		super(OmniscienceReader, self).__init__()
		self.file_path = file_path
		self.raw_df = pd.read_csv(self.file_path, sep='\t', encoding='utf-8')
		self.preprocess()

	def preprocess(self):
		self.om_df = self.raw_df
		self.om_df["omniscience_label_ids"] = self.om_df["omniscience_label_ids"].apply(lambda x: ast.literal_eval(x) )
		self.om_df["omniscience_labels"] = self.om_df["omniscience_labels"].apply(lambda x: ast.literal_eval(x) )
		self.om_df["doc_id"] = 0
		
		for i in tqdm(self.om_df.index):
			self.om_df.at[i, "doc_id"] = i
		return	self.om_df


# remove this if it isn't required [after checking]
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