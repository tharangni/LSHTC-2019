import os
import pickle
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm 
from pathlib import Path
from joblib import Memory
from random import sample
from collections import OrderedDict

warnings.simplefilter('ignore')

# p2c, c2p, n2i, i2n, pi, T, N = lookup_table("swiki/data/cat_hier.txt", subset = False)
mem = Memory("../mycache")
@mem.cache
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
def generate_label_vector(all_labels):
	order_label_mapping = {}
	sort_labels = sorted(all_labels)
	i = 1
	
	for old_label in sort_labels:
		if old_label not in order_label_mapping:
			order_label_mapping[old_label] = i
			i+=1
	return order_label_mapping

# one_hot_label_vector = {}
# one_hot_vec = {}
# one_hot_vec_fname = 'swiki/pickle/one_hot_vec.pickle'
# one_hot_label_vector_fname = 'swiki/pickle/one_hot_label_vector.pickle'
# if not os.path.isfile(one_hot_vec_fname):
# 	pickle_out = open(one_hot_vec_fname, "wb")
# 	for _, value in order_label_mapping.items():
# 		if value not in one_hot_vec:
# 			label_vec = np.zeros((len(sort_labels),))
# 			label_vec[value-1] = 1
# 			one_hot_vec[value] = label_vec
# 			del label_vec
# 	pickle.dump(one_hot_vec, pickle_out)
# 	pickle_out.close()
# else:
# 	pickle_in = open(one_hot_vec_fname, "rb")
# 	one_hot_vec = pickle.load(pickle_in)
# for key, value in order_label_mapping.items():
# 	if key not in one_hot_label_vector:
# 		one_hot_label_vector[key] = one_hot_vec[value]

	

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
	
	all_samples = list(range(len(train_data)))
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