import os
import logging
import igraph as ig
import pandas as pd
import networkx as nx

logging.basicConfig(level=logging.INFO)


class EntityProcessor(object):
	"""
	docstring for EntityProcessor
	
	Preprocesses the category file to the required fastText format

	[x] Ensure that hierarchy is tree (hence we need graphml format - it is easier then)
	[x] lowercase text
	[x] use '#' as delim in case of utf-8 text/str text
	[x] replace <space> between words with another delimiter such as <->
	[x] <head> <tab> <relation> <tab> __label__<tail> <endline>
	[x] if no arborescence, then select max outbound set of nodes
	"""
	def __init__(self, edgelist_file, delim):
		super(EntityProcessor, self).__init__()
		
		self.edgelist_file = edgelist_file
		self.delim = delim
		self.convert2GraphML()
		self.treeConverter()
		# self.adding_tagger()


	def convert2GraphML(self):
	
		fe, ex = os.path.splitext(self.edgelist_file)
		csv_file = "{}_graph.csv".format(fe)
		self.gml_file = "{}_graph.graphml".format(fe)

		file_exist = os.path.isfile(self.gml_file)

		if not file_exist:
			# 1. convert to pandas df 
			if self.delim == '#':
				with open(self.edgelist_file, "rb") as fmain:
					reader = fmain.readlines()

				row_appender = []
				for i, lines in enumerate(reader):
					new_line = lines.decode('utf-8')
					new_line = new_line.lower()
					new_line = new_line.replace(' ', '-')
					new_line = new_line.strip().split('#')
					row_appender.append(new_line)
			
			else:
				with open(self.edgelist_file, "r") as fmain:
					reader = fmain.readlines()

				row_appender = []
				for i, lines in enumerate(reader):
					new_line = lines.strip().split(' ')
					new_ = [new_line[0], new_line[1]]
					row_appender.append(new_)

			cat = pd.DataFrame(row_appender, columns=["parent", "child"])
			cat.to_csv(csv_file, index=False)

			# 2. use df object as edgelist to create graphml object
			load_df = cat

			self.Graph = nx.from_pandas_edgelist(load_df, 'parent', 'child', create_using=nx.DiGraph)
			nx.write_graphml_xml(self.Graph, self.gml_file)

		else:

			self.Graph = nx.read_graphml(self.gml_file)

		logging.info("Converted to graphml!")


	def treeConverter(self):
		
		is_treee = nx.is_tree(self.Graph)
		is_arb = nx.is_arborescence(self.Graph)

		if not is_arb:
			logging.info("Arborescence is not possible due to in-bound and out-bound edges. Therefore keeping only out-bound edges.")
			self.reduceToOutBound()
		elif not is_treee:
			logging.info("Converting dag to tree")
			arb_graph = nx.minimum_spanning_arborescence(self.Graph)
			if nx.is_tree(arb_graph):
				logging.info("Converted to tree! But some information is lost...")
				nx.write_edgelist(arb_graph, self.edgelist_file)
				self.removeParantheses(self.edgelist_file)
				nx.write_graphml(arb_graph, self.gml_file)
				self.Graph = arb_graph


	def reduceToOutBound(self):
		
		fe, ex = os.path.splitext(self.edgelist_file)
		new_f = "{}_outbound{}".format(fe, ex)
		
		inDegree = self.Graph.in_degree()
		
		neighbours_n = {}
		remove_n = []
		
		for n, degree in inDegree:
			if degree == 0:
				neighbours_n[n] = list(self.Graph.neighbors(n))
				
		neighbours_n = sorted(neighbours_n.items(), key = lambda x: len(x[1]), reverse=True)   
		
		for i, j in neighbours_n[1:]:
			remove_n.append(i)
			
		self.Graph.remove_nodes_from(remove_n)
		nx.write_edgelist(self.Graph, new_f)
		self.removeParantheses(new_f)
		self.edgelist_file = new_f


	def fasttextConverter(self):
		logging.info("--Beginning conversion--")
		fe, ex = os.path.splitext(self.edgelist_file)
		new_f = "{}_fasttext{}".format(fe, ex)


		if self.delim != '#':
			with open(self.edgelist_file, "r") as fmain:
				reader = fmain.readlines()
				
			fin = open(new_f, "w+")

			for i, line in enumerate(reader):
				split_line = line.strip().split(' ')
				try:
					parent = int(split_line[0])
					child = int(split_line[1])
				except:
					parent = str(split_line[0])
					child = str(split_line[1])

				parent_of = "parent-of \t __label__A{} \t __label__A{}\n".format(parent, child)
				fin.write(parent_of)
				child_of = "child-of \t __label__A{} \t __label__A{}\n".format(child, parent)
				fin.write(child_of)

			fin.close()

		else:
			with open(self.edgelist_file, "rb") as fmain:
				reader = fmain.readlines()
				
			fin = open(new_f, "wb+")

			for i, line in enumerate(reader):
				line = line.decode('utf-8')
				split_line = line.strip().split(' ')
				try:
					parent = int(split_line[0])
					child = int(split_line[1])
				except:
					parent = str(split_line[0])
					child = str(split_line[1])

				parent_of = "parent-of \t __label__{} \t __label__{}\n".format(parent, child)
				fin.write(parent_of.encode('utf-8'))
				child_of = "child-of \t __label__{} \t __label__{}\n".format(child, parent)
				fin.write(child_of.encode('utf-8'))

			fin.close()



		logging.info("--Converted to fastText format--")


	def removeParantheses(self, file):
		
		if self.delim == '#':
			with open(file, "rb") as fmain:
				reader = fmain.readlines()

			file_str = ""

			for i, lines in enumerate(reader):
				lines = lines.decode('utf-8')	
				line = lines.strip().replace("{'weight': 1}", "")
				line = lines.strip().replace("{}", "")

				file_str += "{}\n".format(line)

			with open(file, "wb") as fmain:
				fmain.write(file_str.encode('utf-8'))

		else:
			with open(file, "r") as fmain:
				reader = fmain.readlines()

			file_str = ""

			for i, lines in enumerate(reader):
				line = lines.strip().replace("{'weight': 1}", "")
				line = lines.strip().replace("{}", "")

				file_str += "{}\n".format(line)

			with open(file, "w") as fmain:
				fmain.write(file_str)



if __name__=="__main__":
	path = os.path.relpath(path="../../../Starspace/data/oms/cat_hier.txt")
	T = EntityProcessor(path, '#')
	T.fasttextConverter()

	path = os.path.relpath(path="../../../Starspace/data/raw-swiki/cat_hier.txt")
	T = EntityProcessor(path, ' ')
	T.fasttextConverter()
	# print(T.H.draw_graph())

# notes to see if it actually works on datasets:
# 1. raw swiki: yes (parent-of, child-of) relations
# 2. omniscience (full category): 