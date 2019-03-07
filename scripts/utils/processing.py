import os
import torch
import numpy as np
import pandas as pd
import igraph as ig

from tqdm import tqdm 
from pathlib import Path
from random import sample
from collections import OrderedDict, Counter

from joblib import Memory
from sklearn.datasets import load_svmlight_file

def lookup_table(filename, subset):
	
	p2c_table = {}
	c2p_table = {}
	node2id = OrderedDict()
	id2node = OrderedDict()
	i = 0

	with open(filename, "r") as f:
		if not subset:
			head = f
		elif isinstance(subset, int):
			head = [next(f) for x in range(subset)] # retrieve only `n` docs
		else:
			raise ValueError("Incorrect subset type. Enter only False (boolean) or int. Encountered {} type.".format(type(subset)))
		for _, line in enumerate(tqdm(head)):
			split_line = line.strip().split()
			parent_node = int(split_line[0])
			child_node = list(map(int, split_line[1:]))
			
			# map to the respective dicts -> parent:child relationship
			# parent2child lookup table
			if parent_node not in p2c_table:
				p2c_table[parent_node] = [child_node[0]]
			else:
				p2c_table[parent_node].append(child_node[0])
				
			#child2parent lookup table
			if child_node[0] not in c2p_table:
				c2p_table[child_node[0]] = [parent_node]
			else:
				c2p_table[child_node[0]].append(parent_node)
				
			# map parent/child node to a node<->id
			if parent_node not in node2id:
				p_id = i
				node2id[parent_node] = p_id
				id2node[p_id] = parent_node
				i+=1
			else:
				p_id = node2id[parent_node]
				
			if child_node[0] not in node2id:
				c_id = i
				node2id[child_node[0]] = c_id
				id2node[c_id] = child_node[0]      
				i+=1
			else:
				c_id = node2id[child_node[0]]

	pi_parents = set(p2c_table.keys())        
	T_leaves = (c2p_table.keys() - p2c_table.keys()) 
	N_all_nodes = pi_parents.union(T_leaves)

	return p2c_table, c2p_table, node2id, id2node, list(pi_parents), list(T_leaves), list(N_all_nodes)


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


def generate_binary_yin(N_all_nodes):
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
	y_in_dash = torch.from_numpy(np.where(t_16.numpy() > 0, 1, -1)).float()
	
	return y_in_dash