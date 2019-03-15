import os
import time
import torch
import random
import logging
import numpy as np
import igraph as ig

from tqdm import tqdm 
from pathlib import Path
from joblib import Memory

from collections import Counter, OrderedDict

def lookup_table(filename, subset):

	'''
	filename: <str> path to category file
	subset: <bool> or <int> False or an int representing the number of labels to sample from
	ASSUMPTION:
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
				if parent_node not in p2c_table:
					p2c_table[parent_node] = [int(x)]
				else:
					p2c_table[parent_node].append(int(x))
				
			#child2parent lookup table
			for x in child_node:
				int_x = int(x)
				if x not in c2p_table:
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
	
	return p2c_table, c2p_table, node2id, id2node, list(pi_parents), list(T_leaves), list(N_all_nodes)



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
