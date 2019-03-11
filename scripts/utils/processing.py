import os
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm 
from pathlib import Path
from random import sample
from collections import OrderedDict, Counter

from joblib import Memory
from sklearn.datasets import load_svmlight_file

# # N and labels_dict have to be present globally! (list of all the labels)
# # labels_dict because I will keep accessing it for each document
# order_label_mapping = generate_label_vector(N)
def generate_order_mapping(N_all_nodes):

	order_mapping = {}
	sorted_nodes = sorted(N_all_nodes)

	for i, each_node in enumerate(sorted_nodes):
		if each_node not in order_mapping:
			order_mapping[each_node] = i+1

	return order_mapping


def list2tensor(inp_list):
	'''
	converts list to tensor
	'''
	list_tensors = list(map(torch.Tensor, inp_list))
	out_tensor = torch.stack(list_tensors, dim=0)

	return out_tensor


mem = Memory("./../../mycache")
@mem.cache
def generate_wn(N_all_nodes, n):
	'''
	Creates unique parameters for all nodes. 
	Acts like a dictionary
	'''
	list_all = []
	for _ in tqdm(range(len(N_all_nodes))):
		c = torch.Tensor(n,).uniform_(0, 1)
		c = list(c.numpy())
		if c not in list_all:
			list_all.append(c)

	wn_tensors = list2tensor(list_all)
	
	return wn_tensors


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