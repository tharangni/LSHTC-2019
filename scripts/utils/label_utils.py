import os
import torch
import random
import logging

from scripts.utils.hierarchy import *
logging.basicConfig(level=logging.INFO)

class HierarchyUtils(object):
	"""
	docstring for TreeUtils
	[x] ideas: w_n, w_pi, 
	leaf check, 
	#parents checker, 
	[x] depth of tree, 
	[x] depth + path of tree till that point,
	[x] subsample from subtree
	[] REPRESENTATIVE LABEL EMBEDDINGS
	display all info in table
	[x] island checker
	[x] un/directed graph vector generation
	[x] path frequency distribution
	"""
	
	def __init__(self, category_file, directed, is_text):
		
		super(HierarchyUtils, self).__init__()
		
		self.category_file = category_file
		self.directed = directed
		self.is_text = is_text
		
		if is_text:
			lookup = lookup_text(self.category_file, False)
		else:
			lookup = lookup_table(self.category_file, False)

		self.parent2child_table = lookup["parent2child"]
		self.child2parent_table = lookup["child2parent"]
		self.node2id = lookup["node2id"]
		self.id2node = lookup["id2node"]
		self.pi_parents = lookup["pi_parents"]
		self.T_leaves = lookup["T_leaves"]
		self.N_all_nodes = lookup["N_all_nodes"]
		
		self.hier_type = hierarchy_type(self.child2parent_table)
		
		self.hier_obj = hierarchy2graph(self.parent2child_table, self.node2id, self.directed)


	def get_depth(self, hist = False):
			
		assert bool(self.directed) == True, "Depth cannot be calculated for undirected graphs"

		if hist:
			return print(self.hier_obj.path_length_hist())
		return self.hier_obj.diameter()


	def draw_graph(self, num_samples=25):
		
		assert num_samples < 100, "Sample size of {} is too large to display output properly".format(num_samples)
		
		p2c, _, n2i, _, _, _, _ = lookup_table(self.category_file, num_samples)
		g = hierarchy2graph(p2c, n2i, self.directed)
		layout = g.layout("kk")
		g.vs["label"] = g.vs["name"]
		return ig.plot(g, layout = layout)


	def get_shortest_path(self, src, dest):
		'''
		pass only id of node as source and destination vertices
		'''
		return self.hier_obj.get_shortest_paths(src, dest)

	def BFS(self, s, n = 32, device = 'cpu', only_nodes = False): 
		'''
		- s : starting node (id)
		- BFS traversal of a graph 
		- passing the id of node as an arg
		'''
		visited = [False] * (len(self.N_all_nodes)) 
		node2vec = {}
		subtree = []

		queue = [] 
		
		queue.append(s) 
		visited[s] = True
		
		if self.id2node[s] not in node2vec:
			root_vector = torch.distributions.normal.Normal(torch.tensor([0.5]), torch.tensor([0.1]))
			node2vec[self.id2node[s]] = root_vector.sample((n,))

		while queue: 

			s = queue.pop(0) 
			subtree.append(s)
			# print (self.id2node[s]) 

			for i in self.hier_obj.neighbors(s): 
				if visited[i] == False: 
					queue.append(i) 
					visited[i] = True
					if self.id2node[i] not in node2vec:
						rand = random.uniform(0.0001, 0.0005)
						node2vec[self.id2node[i]] = node2vec[self.id2node[s]] + rand

		if only_nodes:
			res = subtree
		else:
			res = node2vec

		return res


	def find_components(self):
	# returns the number of connected components of a graph
	# also returns the list of `starting vertices` for these components

		fe, ex = os.path.splitext(self.category_file)
		fe = fe + "_components"
		full_file = fe + ex

		if not os.path.isfile(full_file):
			file = open(full_file, "w+")
			file.write(str(self.hier_obj.clusters()))
			file.close()
		
		with open(full_file, "r") as f:
			lines = f.readlines()

		num_compenents = int(lines[0].split()[-2])

		component_vertex = []
		for each_line in lines[1:]:
			try:
				start = int(each_line.strip().split(",")[0].strip().split("[")[1].strip().split("]")[1])
				if start not in component_vertex:
					component_vertex.append(start)
			except:
				pass
		
		return component_vertex, num_compenents


	def island_checker(self):
		# returns <bool> if islands exist in the graph

		_, n = self.find_components()
		if n > 1:
			logging.info("{} number of islands exist".format(n))
			res = True
		else:
			logging.info("No islands exist. There is only one fully connected component")
			res = False
		return res

	
	def subtree(self, starting_node):
		'''
		returns node set of a subtree. essentially all the children under that node
		'''

		subtree_sample = self.BFS(starting_node, 0, 'cpu', True)
		return subtree_sample


	def generate_vectors(self, device = 'cpu', neighbours = True):
		
		n = len(self.N_all_nodes)

		if self.directed:
			# this method works for trees + dag
			node2vec = {}

			# 1. find the root node(s). in degree = 0
			in_degree = self.hier_obj.degree(mode = "in")
			in_degree_nodes = np.where(np.array(in_degree)==0)[0]

			# 2. use BFS for generating level order label vectors
			for x in in_degree_nodes:
				temp = self.BFS(	x, n, device, False)
				node2vec =  {**node2vec, **temp}
			res = node2vec

			if neighbours and self.hier_type == 'tree':
				# this is for trees
				w_pi = {}

				for node, vector in node2vec.items():
					if node not in w_pi and node in self.child2parent_table.keys():
						node_parent = self.child2parent_table[node][0]
						w_pi[node] = node2vec[node_parent]

				res = node2vec, w_pi

			# compute neighbours (their vectors) in case of graphs
			elif neighbours and self.hier_type == 'graph':
				w_neighs = {}
				for node, vec in node2vec.items():
					neighbours = self.hier_obj.neighbors(self.node2id[node], mode="in")
					if node not in w_neighs:
						w_neighs[node] = [node2vec[self.id2node[neigh_vecs]] for neigh_vecs in neighbours]
				res = node2vec, w_neighs
		else:
			# undirected graphs don't have 0 degree
			# so starting nodes can be vertices from components (subgraphs/island graphs)
			starting_nodes, _ = self.find_components() 

			node2vec = {}
			
			for x in tqdm(starting_nodes):
				temp = self.BFS(self.node2id[x], n, device, False)
				for node, vec in temp.items():
					if node not in node2vec:
						node2vec[node] = vec
			
			res = node2vec
			
			if neighbours:
				w_neighs = {}
				for node, vec in node2vec.items():
					neighbours = self.hier_obj.neighbors(self.node2id[node])
					if node not in w_neighs:
						w_neighs[node] = [node2vec[self.id2node[neigh_vecs]] for neigh_vecs in neighbours]
				res = node2vec, w_neighs			

		return res



if __name__ == '__main__':
	path = os.path.relpath(path="OmniScience/original/os_tree_cat_hier.txt")
	# T = HierarchyUtils(path, False, True)
	# print(T.draw_graph(50))
	# print(T.get_depth(True))

	# vecs = T.generate_vectors()	
	# print(len(vecs[0]))
	# print("-"*50)
	# print(len(vecs[1]))