import os

from hierarchy import *

class HierarchyUtils(object):
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
		super(HierarchyUtils, self).__init__()
		self.category_file = category_file
		self.directed = directed
		self.parent2child_table, self.child2parent_table, self.node2id, \
		self.id2node, self.pi_parents, self.T_leaves, self.N_all_nodes = lookup_table(self.category_file, subset)
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


	# BFS traversal of a graph 
	# passing the id of node as an arg
	def BFS(self, s, n = 16, device = 'cpu'): 

	    # Mark all the vertices as not visited 
	    visited = [False] * (len(self.N_all_nodes)) 
	    node2vec = {}
	    
	    # Create a queue for BFS 
	    queue = [] 
	    
	    # Mark the source node as  
	    # visited and enqueue it 
	    queue.append(s) 
	    visited[s] = True
	    
	    if self.id2node[s] not in node2vec:
	        root_vector = np.random.normal(loc = 0.8, scale = 0.1, size = n)
	        node2vec[self.id2node[s]] = torch.as_tensor(root_vector, device = device, dtype = torch.float32)

	    while queue: 

	        # Dequeue a vertex from  
	        # queue and print it 
	        s = queue.pop(0) 
	        # print (self.id2node[s]) 

	        # Get all adjacent vertices of the 
	        # dequeued vertex s. If a adjacent 
	        # has not been visited, then mark it 
	        # visited and enqueue it 
	        for i in self.hier_obj.neighbors(s): 
	            if visited[i] == False: 
	                queue.append(i) 
	                visited[i] = True
	                if self.id2node[i] not in node2vec:
	                    rand = random.uniform(0.0001, 0.0005)
	                    node2vec[self.id2node[i]] = node2vec[self.id2node[s]] + rand
	    return node2vec

		
	def generate_vectors(self, n = 16, device = 'cpu', neighbours = True):
		
		if self.directed:
			# this method works for trees + dag
			node2vec = {}

			# 1. find the root node(s). in degree = 0
			in_degree = self.hier_obj.degree(mode = "in")
			in_degree_nodes = np.where(np.array(in_degree)==0)[0]
			in_degree_nodes = [self.id2node[x] for x in in_degree_nodes]

			while(len(node2vec) < len(self.N_all_nodes)):
				for e_in in in_degree_nodes:
					# 2. generate random vector for root
					if e_in not in node2vec:
						root_vector = np.random.normal(loc = 0.8, scale = 0.1, size = n)
						node2vec[e_in] = torch.as_tensor(root_vector, device = device, dtype = torch.float32)

					# 3. children: find immediate neighbours of root (1 level down)
					# 4. generate random vectors for each neighbour at uniform randomness
					neighbours = self.hier_obj.neighbors(self.node2id[e_in])
					for neighbour_edges in neighbours:
						if self.id2node[neighbour_edges] not in node2vec:
							rand = random.uniform(0.0001, 0.0005)
							curr_vector = node2vec[e_in] + rand
							node2vec[self.id2node[neighbour_edges]] = torch.as_tensor(curr_vector, device = device, dtype = torch.float32)

				# 5. repeat the above process at each level of the graph (after R0 -> 1st level -> 2nd level ...)
				in_degree_nodes = list(node2vec.keys())

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
					neighbours = self.hier_obj.neighbors(self.node2id[node])
					if node not in w_neighs:
						w_neighs[node] = [node2vec[self.id2node[neigh_vecs]] for neigh_vecs in neighbours]
				res = node2vec, w_neighs


		else:
			# [TODO]: Raincheck 0.0
			starting_node = 0 # no reason why. just starting with the 0th node
			
			node2vec = self.	BFS(starting_node, n, device)
			
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
	path = os.path.relpath(path="swiki/data/cat_hier.txt")
	T = HierarchyUtils(path, False, True)
	print(T.hier_type)
	# print(T.get_depth(True))

	vecs = T.generate_vectors()	
	print(len(vecs[0]))
	print("*"*50)
	print(len(vecs[1]))