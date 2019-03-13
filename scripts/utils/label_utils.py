import os

from hierarchy import *

class TreeUtils(object):
	"""
	docstring for TreeUtils
	[x] ideas: w_n, w_pi, 
	leaf check, 
	#parents checker, 
	[x] depth of tree, 
	depth + path of tree till that point,
	subsample from subtree
	display all info in table
	undirected graph vector generation
	[x] path frequency distribution
	"""
	
	def __init__(self, category_file, subset, directed):
		super(TreeUtils, self).__init__()
		self.category_file = category_file
		self.subset = subset
		self.directed = directed
		self.parent2child_table, self.child2parent_table, self.node2id, \
		self.id2node, self.pi_parents, self.T_leaves, self.N_all_nodes = lookup_table(self.category_file, self.subset)
		self.graph_obj = hierarchy2graph(self.parent2child_table, self.node2id, self.directed)
		
		
	def get_depth(self, hist = False):
		
		assert bool(self.directed) == True, "Depth can be calculated only for DAG"
		
		if hist:
			return print(self.graph_obj.path_length_hist())
		return self.graph_obj.diameter()

	
	def draw_graph(self, num_samples=25):
		
		assert num_samples < 100, "Sample size of {} is too large to display output properly".format(num_samples)
		
		p2c, _, n2i, _, _, _, _ = lookup_table(self.category_file, num_samples)
		g = hierarchy2graph(p2c, n2i, self.directed)
		layout = g.layout("kk")
		g.vs["label"] = g.vs["name"]
		return ig.plot(g, layout = layout)

		
	def generate_vectors(self, n = 16, device = 'cpu', parent = True):
		
		if self.directed:
			node2vec = {}

			# 1. find the root node(s). in degree = 0
			in_degree = self.graph_obj.degree(mode = "in")
			in_degree_nodes = np.where(np.array(in_degree)==0)[0]

			for each_root_node in (in_degree_nodes):
				r_node = self.id2node[each_root_node]
				depth = 1
				if r_node not in node2vec:
					# 2. generate random vector for root
					root_vector = np.random.normal(loc = 1, scale = 0.1, size = n)
					node2vec[r_node] = torch.as_tensor(root_vector, device = device, dtype = torch.float32)

				# 3. children: find immediate neighbours of root (1 level down)
				# 4. generate random vectors for each neighbour at uniform randomness
				for child in self.graph_obj.neighbors(self.node2id[r_node]):
					if self.id2node[child] not in node2vec:
						rand = random.uniform(0.0001, 0.0005)
						curr_vector = node2vec[self.child2parent_table[self.id2node[child]][0]] + rand
						node2vec[self.id2node[child]] = torch.as_tensor(curr_vector, device = device, dtype = torch.float32)   

			# 5. repeat the above process at each level of the tree (after R0 -> 1st level -> 2nd level ...)
			while(len(node2vec)<len(self.N_all_nodes)):
				depth += 1
				level_nodes = list(node2vec.keys())
				for each_node in (level_nodes):
					for child in self.graph_obj.neighbors(self.node2id[each_node]):
						if self.id2node[child] not in node2vec:
							all_children = self.child2parent_table[self.id2node[child]]
							rand = random.uniform(0.0001, 0.0005)
							try:
								curr_vector = node2vec[all_children[0]] + rand
							except:
								continue
							node2vec[self.id2node[child]] = torch.as_tensor(curr_vector, device = device, dtype = torch.float32)

			res = node2vec

			if parent:
				w_pi = {}

				for node, vector in node2vec.items():
					if node not in w_pi and node in self.child2parent_table.keys():
						node_parent = self.child2parent_table[node][0]
						w_pi[node] = node2vec[node_parent]

				res = node2vec, w_pi
		
		else:
			print("it's for an undirected graph")
		
		return res

		

if __name__ == '__main__':
	path = os.path.relpath(path="swiki/data/cat_hier.txt")
	T = TreeUtils(path, False, True)
	print(T.draw_graph())
	print(T.get_depth(True))

	vecs = T.generate_vectors()
	print(vecs[0][262159])
	print("**"*50)
	print(vecs[1][262159])