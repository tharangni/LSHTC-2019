import os
import time
import logging
import numpy as np
import igraph as ig
import pandas as pd

from tqdm import tqdm_notebook as tqdm 
from pathlib import Path
from joblib import Memory

from collections import Counter, OrderedDict

def lookup_table(filename, subset):

	'''
	filename: <str> path to category file
	subset: <bool> or <int> False or an int representing the number of labels to sample from
	ASSUMPTION: 
	the file format should be an edgelist
	category format in the file is:
	12345 23456
	12345 34567
	the first number represents parent and the number following it represents child
	'''
	
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
			for x in child_node:
				int_x = int(x)
				if parent_node not in p2c_table:
					p2c_table[parent_node] = [int_x]
				else:
					p2c_table[parent_node].append(int_x)
				
			#child2parent lookup table
			for x in child_node:
				int_x = int(x)
				if int_x not in c2p_table:
					c2p_table[int_x] = [parent_node]
				else:
					c2p_table[int_x].append(parent_node)
				
			# map parent/child node to a node<->id
			if parent_node not in node2id:
					p_id = i
					node2id[parent_node] = p_id
					id2node[p_id] = parent_node
					i+=1
			else:
				p_id = node2id[parent_node]

			for x in child_node:
				int_x = int(x)
				if int_x not in node2id:
					c_id = i
					node2id[int_x] = c_id
					id2node[c_id] = int_x      
					i+=1
				else:
					c_id = node2id[int_x]

	pi_parents = set(p2c_table.keys())        
	T_leaves = (c2p_table.keys() - p2c_table.keys()) 
	N_all_nodes = pi_parents.union(T_leaves)


	obj = { 
	"parent2child" : p2c_table,
	"child2parent" : c2p_table,
	"node2id" : node2id,
	"id2node" : id2node,
	"pi_parents" : list(pi_parents),
	"T_leaves" : list(T_leaves),
	"N_all_nodes" : list(N_all_nodes) }

	return obj



def hierarchy2graph(p2c_table, node2id, directed):

	edges = []
	for parent, children in p2c_table.items():
		p_id = node2id[parent]
		for child in children:
			c_id = node2id[child]
			edges.append((p_id, c_id))
	vertices = [k for k, v in node2id.items()]
	g = ig.Graph(n=len(node2id), edges=edges, directed=directed, vertex_attrs={"name": vertices})
	return g


def hierarchy_type(c2p_table):
	#  is this logic correct? (comparing against binary trees)
	temp = []
	
	for vals in list(c2p_table.values()):
		for x in vals:
			temp.append(x)

	if(len(temp) == len(list(c2p_table.keys()))):
		type_ = 'tree'
	else:
		type_ = 'graph'

	del temp
	return type_


def read_omniscience(csv_file):
	'''
	APPLICABLE ONLY FOR CSV FILES WHICH HAVE A PATH COLUMN THAT ENTAILS HIERARCHICAL PATH
	The hierarchical path should be of the format:
	<label-id1/label-id2/label-id3...> each line. 
	the first label id of each line indicates the parent node and the ids following it 
	represent children
	# C:/Users/harshasivajit/Documents/master-ai/rr13/OmniScience/original/os_tree.csv
	# read the csv only if the category file doesn't exist
	'''
	
	fe, ex = os.path.splitext(csv_file)
	fe = fe + "_cat_hier"
	fe_l = fe + "_labels"
	ex = ".txt"
	full_file = fe + ex
	full_labels = fe_l + ex
	
	if not os.path.isfile(full_file):
		oms = pd.read_csv(csv_file, sep=',', encoding='utf-8')
		parent_id = oms["parentconceptid"]
		concept_id = oms["conceptid"]
		file = open(full_file, "w+")
		for i in tqdm(range(len(oms))):
			parent = parent_id[i]
			child = concept_id[i]
			if not np.isnan(parent):
				str_each_line = "{} {}".format(int(parent), child)
				file.write(str_each_line + '\n')
		file.close()

	if not os.path.isfile(full_labels):
		oms = pd.read_csv(csv_file, sep=',', encoding='utf-8')
		parent_id = oms["parentlabel"]
		concept_id = oms["label"]
		file = open(full_labels, mode = "wb+")
		for i in tqdm(range(len(oms))):
			parent = parent_id[i]
			child = concept_id[i]
			if isinstance(parent, str):
				str_each_line = "{}#{}\n".format(parent, child)
				str_each_line = str_each_line.encode('utf-8')
				file.write(str_each_line)
		file.close()
	return full_file


def lookup_text(filename, subset):

	'''
	APPLICABLE ONLY FOR TEXT LABEL DATA
	filename: <str> path to category file
	subset: <bool> or <int> False or an int representing the number of labels to sample from
	ASSUMPTION:
	category format in the file is encoded in bytes separated by #:
	b'science#physics
	b'physics#mechanics
	the first number represents parent and the number following it represents child
	'''
	
	p2c_table = {}
	c2p_table = {}
	node2id = OrderedDict()
	id2node = OrderedDict()
	i = 0
	
	with open(filename, "rb") as f:
		if not subset:
			head = f
		elif isinstance(subset, int):
			head = [next(f) for x in range(subset)] # retrieve only `n` docs
		else:
			raise ValueError("Incorrect subset type. Enter only False (boolean) or int. Encountered {} type.".format(type(subset)))
		for _, line in enumerate(tqdm(head)):
			split_line = line.strip().split(b'#')
			parent_node = split_line[0]
			parent_node = parent_node.decode("utf-8") 
			child_node = list(map(bytes, split_line[1:]))

			# map to the respective dicts -> parent:child relationship
			# parent2child lookup table
			for x in child_node:
				str_x = x.decode("utf-8") 
				if parent_node not in p2c_table:
					p2c_table[parent_node] = [str_x]
				else:
					if str_x not in p2c_table[parent_node]:
						p2c_table[parent_node].append(str_x)
				
			#child2parent lookup table
			for x in child_node:
				int_x = x.decode("utf-8") 
				if int_x not in c2p_table:
					c2p_table[int_x] = [parent_node]
				else:
					if parent_node not in c2p_table[int_x]:
						c2p_table[int_x].append(parent_node)
				
			# map parent/child node to a node<->id
			if parent_node not in node2id:
				p_id = i
				node2id[parent_node] = p_id
				id2node[p_id] = parent_node
				i+=1
			else:
				p_id = node2id[parent_node]
			
			for x in child_node:
				int_x = x.decode("utf-8") 
				if int_x not in node2id:
					c_id = i
					node2id[int_x] = c_id
					id2node[c_id] = int_x      
					i+=1
				else:
					c_id = node2id[int_x]

	pi_parents = set(p2c_table.keys())        
	T_leaves = (c2p_table.keys() - p2c_table.keys()) 
	N_all_nodes = pi_parents.union(T_leaves)
	
	obj = { 
	"parent2child" : p2c_table,
	"child2parent" : c2p_table,
	"node2id" : node2id,
	"id2node" : id2node,
	"pi_parents" : list(pi_parents),
	"T_leaves" : list(T_leaves),
	"N_all_nodes" : list(N_all_nodes) }

	return obj
