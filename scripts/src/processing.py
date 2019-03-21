import os
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm 
from pathlib import Path
from random import sample
from collections import OrderedDict, Counter

from joblib import Memory

# [x] pre-process raw text as well
# [x] min freq count
# [x] doc contents
# [x] what to use for featurizing - fasttext, sparse features


def list2tensor(inp_list):
	'''
	converts list to tensor
	'''
	list_tensors = list(map(torch.Tensor, inp_list))
	out_tensor = torch.stack(list_tensors, dim=0)

	return out_tensor



def generate_binary_yin(N_all_nodes, device):
	'''
	Alternate method to generate y_in values. If a node n belongs to an 
	instance i, it accesses the respective 16-bit binary {-1, +1} representation 
	of that node-number (from order_mapping fn) as the respective y_in.
	'''
	all_16 = []
	for i in tqdm(range(1, len(N_all_nodes)+1)):
		bin_rep = bin(i)[2:]
		rep_16 = '{:016d}'.format(int(bin_rep))
		list_16 = list(map(int, rep_16))
		all_16.append(list_16)

	t_16 = list2tensor(all_16)
	y_in_dash = torch.as_tensor(np.where(t_16.numpy() > 0, 1.0, -1.0), device = device, dtype = torch.float32)
	
	return y_in_dash


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