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



def hierarchy2graph(p2c_table, node2id):

	edges = []
	for parent, children in p2c_table.items():
		p_id = node2id[parent]
		for child in children:
			c_id = node2id[child]
			edges.append((p_id, c_id))
	vertices = [k for k, v in node2id.items()]
	g = ig.Graph(n=len(node2id), edges=edges, directed=True, vertex_attrs={"name": vertices})
	return g


def hierarchy_vectors(graph_obj, ix2node, p2c, n, device):
	
	node2vec = {}
	
	# 1. find the root node. in degree = 0
	in_degree = graph_obj.degree(type = "in")
	root_node = ix2node[np.where(np.array(in_degree)==0)[0][0]]
	
	# 2. generate random vector for root
	root_vector = np.random.normal(loc = 1, scale = 0.1, size = n)
	
	for parent, children in p2c.items():
		if parent == root_node:
			node2vec[parent] = torch.as_tensor(root_vector, device = device, dtype = torch.float32)

		# 3. children: find immediate neighbours of root (1 level down)
		# 4. generate random vectors for each neighbour at uniform randomness
		for child in children:
			rand = random.uniform(0.0001, 0.0005)
			if child not in node2vec:
				curr_vector = node2vec[parent] + rand
				node2vec[child] = torch.as_tensor(curr_vector, device = device, dtype = torch.float32)
				
	return node2vec