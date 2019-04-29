import os
import ast # required for Omniscience
import time
import torch
import pickle
import logging
import smart_open

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm 
from pathlib import Path
from joblib import Memory
from random import sample
from scipy.sparse import csr_matrix

from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from collections import Counter, OrderedDict, defaultdict

from skmultilearn import problem_transform
from imblearn.over_sampling import RandomOverSampler, SMOTE

from gensim.models.fasttext import FastText
from gensim.parsing.preprocessing import preprocess_string

# debugging ML code/NNs
# !!!unit testing !!!
num_gpus = torch.cuda.device_count()
device = torch.device("cuda" if (torch.cuda.is_available() and num_gpus > 0) else "cpu")

if torch.cuda.is_available():
	torch.Tensor = torch.cuda.FloatTensor
else:
	torch.Tensor = torch.FloatTensor


mem = Memory("./mycache_getdata")
@mem.cache
def lower_dim(file_path, reduce, n_components):

	data = get_data(file_path)
	new_data = data[0]

	if reduce:
		new_data = call_svd(data[0], n_components)
	else:
		new_data = data[0].todense()

	lbin = MultiLabelBinarizer(sparse_output=True)
	label_matrix = lbin.fit_transform(data[1])
	
	new_doc, new_labels = new_data, data[1]
	
	return new_doc, new_labels, label_matrix, lbin


mem = Memory("./mycache_getdata")
@mem.cache
def call_svd(data, n_components):
	svd = TruncatedSVD(n_components=n_components, n_iter=10, random_state=None)
	current_time = time.time()
	new_data = svd.fit_transform(data)
	elapsed_time = time.time() - current_time
	min_, sec_ = divmod(elapsed_time, 60)
	logging.info("Elapsed time: {}min {:.2f}sec".format(min_, sec_))

	return new_data


mem = Memory("./mycache_getdata")
@mem.cache
def get_data(filename):
	
	fname = str(Path(filename))
	fe, ex = os.path.splitext(fname) 

	try:
# 		data = load_svmlight_file(fname,  multilabel=True)
		data = read_svmlight_file(fname, None)
	except:
		# Required: if the input data isn't in the correct libsvm format
		outfile = str(Path("{}_small{}".format(fe, ex)))
		if not os.path.isfile(outfile):
			logging.info("Remapping data to LibSVM format...")
			f = preprocess_libsvm(fname, outfile)
		else:
			logging.info("Using already remapped data...")
			f = outfile
		data = load_svmlight_file(f, multilabel=True)

	return data[0], data[1]


mem = Memory("./mycache_getdata")
@mem.cache
def read_svmlight_file(file_path, n_features):

	with open(file_path) as fin:
		line_index = 0
		data_indices = list()
		data = list()
		labels = []
		for line in (fin):
			lbl_feat_str, sep, comment = line.strip().partition("#")
			tokens1 = lbl_feat_str.split(',')
			tokens2 = tokens1[-1].split()

			line_labels = [int(i) for i in tokens1[:-1] + tokens2[:1]]
			labels.append(line_labels)

			features = tokens2[1:]
			for f in features:
				fid, fval = f.split(':')
				data_indices.append([line_index, int(fid)]) #-1])
				data.append(float(fval))
			line_index += 1

	n_feat = max(np.array(data)).astype(int) + 1

# 	print("data {}".format(n_feat))

	assert np.all(np.array(data) >= 0)
	assert np.all(np.array(data_indices) >= 0)
	
	if n_features == None:
		X = csr_matrix((np.array(data), np.array(data_indices).T))
	else:
		X = csr_matrix((np.array(data), np.array(data_indices).T), shape = (line_index, n_features))

	return X, labels



def preprocess_libsvm(input_file, output_file):
	# converts file to the required libsvm format.
	# this is very brute force but can be made faster [IMPROVE]

	file = open(output_file, "w+")
	with open(input_file, "r") as f:
		head = [next(f) for x in range(500)] # retrieve only `n` docs
		for i, line in enumerate(tqdm(f)): # change to f/head depending on your needs
			instance = line.strip().split()
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

	return output_file


def class_statistics(df, name, show):
	'''
	statistics display
	<df> : pandas df
	<name> : df name, str
	<show> : plot-show, bool
	'''
	if name == 'omniscience':
		col = "omniscience_labels"
	else:
		# TODO: check column name again for other datasets
		col = "doc_labels"

	# 1. Label ids vs. number of labels
	class_distb = Counter()
	
	for index in tqdm(df.index):
		for labels in df.at[index, col]:
			class_distb[labels] += 1
	
	class_distb = sorted(class_distb.items(), key=lambda x: x[1], reverse=True)

	kk = [k[0] for k in class_distb]
	vv = [k[1] for k in class_distb]

	plt.figure(figsize=(12,8))
	if show:
		plt.figure(1)
		plt.barh(y = kk[:20], width = vv[:20])
		plt.xlabel("Number of labels")
		plt.ylabel("Labels")
		plt.title("Top 20")
		plt.tight_layout();


	# 2. Number of labels per instance (label counts vs. number of instances)
	label_dist = Counter()

	for index in tqdm(df.index):
		label_len = len(df.at[index, col])
		label_dist[label_len] += 1

	label_dist = sorted(label_dist.items(), key = lambda _: _[0], reverse = False )

	kkk = [k[0] for k in label_dist]
	vvv = [k[1] for k in label_dist]

	if show:
		plt.figure(2)
		plt.bar(x = kkk[:20], height = vvv[:20])
		plt.xlabel("Count of labels")
		plt.ylabel("Number of instances")
		plt.title("Decreasing order (count 1 excluded)")
		plt.tight_layout();

	return class_distb, label_dist



class LIBSVM_Reader(object):
	"""
	docstring for LIBSVM_Reader
	[] unit tests (folder)
	'''src/lib/fies
	test/files'''
	[x] dim reduction
	[x] caching
	[x] return df with <doc id, doc labels, doc vec>
	[x] create similar for raw text also (2/2) (OMS[x], RCV1[x])
	[x] distribution of classes per document

	"""
	def __init__(self, file_path, reduce, n_components, subsample, split):
		super(LIBSVM_Reader, self).__init__()
		self.split = split
		self.file_path = file_path
		self.reduce = reduce
		self.n_components = n_components
		self.subsample = subsample

		all_data = lower_dim(self.file_path, self.reduce, self.n_components)
		self.all_x = all_data[0]
		self.all_y = all_data[1]
		self.label_matrix = all_data[2]
		self.binarizer = all_data[3]
		self.view_df()
		self.label_to_doc_mapping()

		# map to corresponding class ids
		class_labels = self.binarizer.classes_
		temp = {}
		for j, i in enumerate(list(class_labels)):
			if i not in temp:
				temp[i] = j

		self.small_mapper = temp
		
		if self.split == 'train':
			self.oversample()
			self.view_df(subsample)
		

	def view_df(self):

		self.data_df = pd.DataFrame(columns = ["doc_id", "doc_labels", "doc_vector", "label_vec"])
        
		if not self.subsample:
			self.data_df["doc_labels"] = self.all_y  
			self.data_df["doc_labels"] = self.data_df["doc_labels"].apply(lambda x: list(map(int, x)))  
			for i in tqdm(self.data_df.index):
# 				temp = self.label_matrix[i].toarray().squeeze()
# 				self.data_df.at[i, "label_vec"] = temp                
				self.data_df.at[i, "doc_vector"] = torch.as_tensor(self.all_x[i], device=device, dtype=torch.float32)
				self.data_df.at[i, "doc_id"] = i
		else:
			# 10% of the original data
			orig_data = round(len(self.all_y) * self.subsample)
			sample_ids = sample(range(len(self.all_y)), orig_data)
			sample_ids = range(orig_data)
			temp_labels = [self.all_y[i] for i in sample_ids]
			self.data_df["doc_labels"] = temp_labels
			self.data_df["doc_labels"] = self.data_df["doc_labels"].apply(lambda x: list(map(int, x)))  
			for i, j in tqdm(enumerate(sample_ids)):
				self.data_df.at[i, "doc_vector"] = self.all_x[j]
				self.data_df.at[i, "label_vec"] = self.label_matrix[j].toarray()
				self.data_df.at[i, "doc_id"] = i

		return self.data_df

	def label_to_doc_mapping(self):
		
		mapper = defaultdict(set)

		for ix in tqdm(self.data_df.index):
			for labelid in self.data_df.at[ix, "doc_labels"]:
				mapper[labelid].add(self.data_df.at[ix, "doc_id"])
		
		self.rev_df = pd.DataFrame(columns=["label_id", "doc_id_list"])
		self.rev_df["label_id"] = list(mapper.keys())
		self.rev_df["doc_id_list"] = list(mapper.values())

		return self.rev_df

	def oversample(self):

		ally = self.all_y
		allx = self.all_x
				
		new_doc, new_labels = [], []

		for i, label_tuple in enumerate(ally):
			for each_label in label_tuple:
				new_doc.append(allx[i])
				new_labels.append(int(each_label))

		new_doc = np.stack(new_doc, axis=0)

		ymat_list = new_labels

		cou = Counter()
		for i in ymat_list:
			cou[i] += 1
		
		d = np.array(list(cou.values()))

		cutoff = round(d.mean()).astype(int)

		cids = {}
		for k, v in cou.items():
			if v < cutoff:
				cids[k] = np.random.randint(cutoff-20, cutoff+1)
			else:
				cids[k] = v

		lp = problem_transform.LabelPowerset()
		transform_ymat = lp.transform(self.label_matrix) 
		ros = RandomOverSampler(random_state=42, sampling_strategy="minority")
		# ros = RandomOverSampler(random_state=42, sampling_strategy=cids)
		self.all_x, y_sm = ros.fit_sample(self.all_x, transform_ymat)
		

		self.label_matrix = lp.inverse_transform(y_sm)
		self.all_y = self.binarizer.inverse_transform(self.label_matrix)



class CSV_Reader(object):
	"""
	docstring for CSV_Reader
	df format: doc_vec, doc_id, doc_labels
	"""
	def __init__(self, filename, subsample):
		super(CSV_Reader, self).__init__()
		self.filename = filename	
		self.subsample = subsample

		if "omniscience" in self.filename.lower():
			self.dataset = "omniscience"
		elif "rcv1" in self.filename.lower():
			self.dataset = "rcv1"
		self.view_csv()
		self.label_to_doc_mapping()
	
	def view_csv(self):
		'''
		maybe put this in a separate class called csv reader?
		FUNCTION TO READ CSV FILES & DOC2VEC
		***the csv files should be tab separated***
		'''
		read_df = pd.read_csv(self.filename, sep="\t")
		self.data_df = pd.DataFrame(columns = ["doc_id", "doc_labels", "doc_vector"])

		if self.dataset == "omniscience":
			self.data_df["doc_id"] = read_df["doc_id"]
			self.data_df["doc_labels"] = read_df["omniscience_label_ids"]
			self.data_df["doc_vector"] = read_df["vec"]
		
		elif self.dataset == "rcv1":
			self.data_df["doc_id"] = read_df["doc_id"]
			self.data_df["doc_labels"] = read_df["topic_ids"]
			self.data_df["doc_vector"] = read_df["vec"]

		self.data_df = self.data_df.dropna()
		
		if self.subsample:
			orig_data = round(len(self.data_df) * 0.1)
			sample_ids = np.random.choice(self.data_df.index, size = orig_data, replace=False)
			self.data_df = self.data_df.iloc[sample_ids, ]

		self.data_df["doc_labels"] = self.data_df["doc_labels"].apply(lambda x: ast.literal_eval(x))
		self.data_df["doc_vector"] = self.data_df["doc_vector"].apply(lambda x: np.fromstring(x.strip().replace("\n", " ").strip().replace("[", "").strip().replace("]", ""), sep=' '))
		self.data_df["doc_vector"] = self.data_df["doc_vector"].apply(lambda x: torch.as_tensor(x, device=device, dtype=torch.float32))
		return self.data_df

	def label_to_doc_mapping(self):
		
		mapper = defaultdict(set)

		for ix in tqdm(self.data_df.index):
			for labelid in self.data_df.at[ix, "doc_labels"]:
				mapper[labelid].add(self.data_df.at[ix, "doc_id"])
		
		self.rev_df = pd.DataFrame(columns=["label_id", "doc_id_list"])
		self.rev_df["label_id"] = list(mapper.keys())
		self.rev_df["doc_id_list"] = list(mapper.values())

		return self.rev_df


class FTextIter(object):
	def __init__(self, file_path):
		super(FTextIter, self).__init__()
		self.file_path = file_path

	def __iter__(self):
		with smart_open.smart_open(self.file_path, 'r', encoding='utf-8') as fin:
			for line in fin:
				line = preprocess_string(line)
				yield list(line)


class OmniscienceReader(object):
	"""
	docstring for OmniscienceReader
	[x] preprocess raw text: gensim preprocesser : stop words + stemming + lemma + tokenize + -num
	[x] raw text -> word2vec using fasttext
	[x] avg word2vec across docs to create doc2vec
	[x] distirbution of classes per document
	"""
	def __init__(self, file_path):
		super(OmniscienceReader, self).__init__()
		self.file_path = file_path
		self.preprocess()

	def preprocess(self):
		self.om_df = pd.read_csv(self.file_path, sep='\t', encoding='utf-8')
		self.om_df = self.om_df.dropna()

		self.om_df["omniscience_label_ids"] = self.om_df["omniscience_label_ids"].apply(lambda x: ast.literal_eval(x) )
		self.om_df["omniscience_labels"] = self.om_df["omniscience_labels"].apply(lambda x: ast.literal_eval(x) )
		self.om_df["category"] = self.om_df["file_id"].apply(lambda x: x.split(":")[0])
		self.om_df["doc_id"] = 0

		for i in tqdm(self.om_df.index):
			self.om_df.at[i, "doc_id"] = i
			if self.om_df.at[i, "category"] == "EVISE.PII":
				self.om_df.at[i, "omniscience_label_ids"] = list(map(int, self.om_df.at[i, "omniscience_label_ids"][0]))

		return self.om_df


	def get_text(self, df):
	
		name = df.iloc[0]["category"]	
		fname = "{}_raw.txt".format(name)
		
		file = open(fname, "wb+")

		for i in tqdm(df.index):
			str_each_line = df.at[i, "abstract"] + '\n'
			file.write(str_each_line.encode('utf-8'))

		file.close()
		
		return name, fname


	def fasttext_generator(self, prefix, name, fname):

		filename = "{}/{}.model".format(prefix, name)

		if not os.path.isfile(filename):
			moo = FastText(size=300, window=3, min_count=1) # hs=0, negative=0, size=300
			moo.build_vocab(sentences=FTextIter(fname))
			total_examples = moo.corpus_count
			moo.train(sentences=FTextIter(fname), total_examples=total_examples, epochs=5)
			moo.save(filename)
		else:
			moo = FastText.load(filename)

		return moo
	

	def gen_doc2vec(self, prefix):
		'''
		prefix: os file path to pickle file/saved models
		'''
		frames = "{}/frames.pkl".format(prefix)
		
		if not os.path.isfile(frames):
			logging.info("Generating document vectors...")
			grouped = self.om_df.groupby('category')
			gkeys = list(grouped.groups.keys())
			group_dfs = [grouped.get_group(tag) for tag in gkeys]
			for df in group_dfs:
				name, fname = self.get_text(df)
				model = self.fasttext_generator(prefix, name, fname)
				
				temp_col = []
				for sent in tqdm(FTextIter(fname)):
					vec = 0.0
					for word in sent:
						vec += model.wv[word] 
					temp_col.append(np.array(vec)/len(sent))
				df["vec"] = temp_col
			with open(frames, "wb") as f:
				pickle.dump(group_dfs, f)
		else:
			logging.info("Loading exisiting model... ")
			with open(frames, "rb") as f:
				group_dfs = pickle.load(f)

		res = pd.concat(group_dfs)
		result = res.sort_index(0)
		
		new_df = result
		
		del result, res

		return new_df

