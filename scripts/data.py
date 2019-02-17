# https://github.com/gcdart/MulticlassClassifier/blob/master/src/ml/LogisticRegression.java
# https://www.kaggle.com/c/lshtc/discussion/6911#38233 - preprocessing: multilabels comma should not have spaces
# https://www.kaggle.com/c/lshtc/discussion/14048 - dataset statistics
## currently reading LWIKI dataset 

import os
import logging
import argparse

from tqdm import tqdm
from pathlib import Path
from joblib import Memory
from collections import OrderedDict
from sklearn.datasets import load_svmlight_file

logging.basicConfig(level=logging.INFO)
mem = Memory("../mycache")


@mem.cache
def get_data(filename):

	fname = str(Path(filename))
	fe, ex = os.path.splitext(fname) 

	try:
		data = load_svmlight_file(fname, multilabel=True)
	except:
		# Required: if the input data isn't in the correct libsvm format
		outfile = str(Path("{}_remapped{}".format(fe, ex)))
		if not os.path.isfile(outfile):
			logging.warning("Remapping data to LibSVM format...")
			f = preprocess_libsvm(fname, outfile)
		else:
			logging.warning("Using already remapped data...")
			f = outfile
		data = load_svmlight_file(f, multilabel=True)
		
	return data[0], data[1]



def preprocess_libsvm(input_file, output_file):
	# converts file to the required libsvm format.
	# this is very brute force but can be made faster [IMPROVE]

	file = open(output_file, "w+")
	with open(input_file, "r") as f:
		for _, line in enumerate(tqdm(f)):
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


def label_extractor(labels):
	
	leaf_labels = set()
	labels_per_doc = []

	for i in labels:
		labels_per_doc.append(len(i))
		for j in i:
			leaf_labels.add(j)
		
	return leaf_labels, labels_per_doc


def read_hier(filename):
	
	# returns class labels
	
	unique_h = set()
	
	with open(filename, "r") as f:
		for i, line in enumerate(f):
			words = line.strip().split()
			for w in words:
				unique_h.add(int(w))

	return unique_h


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()

	# parser arguments: input file
	parser.add_argument("-tr", "--training", type = str,  help = "enter path of training data")
	parser.add_argument("-te", "--testing", type = str,  help = "enter path of testing data")
	parser.add_argument("-cat", "--category", type = str,  help = "enter path of cateogory data")

	args = parser.parse_args()
	print(args)

	# x, a - contains all the feat:value data
	# y, b - contains all the labels (per line)

	# Training data stuff
	x, y = get_data(args.training)
	leaf_labels_y, labels_per_doc_y = label_extractor(y)
	print("Training data: {} - num documents X num features".format(x.shape))
	print("Training data: {} - avg num_labels per instance".format(sum(labels_per_doc_y)/len(y)))
	print("Training data: {} - num leaf labels".format(len(leaf_labels_y)))
	
	# Testing data stuff
	a, b = get_data(args.testing)
	leaf_labels_b, labels_per_doc_b = label_extractor(b)
	print("Testing data: {} - num t_documents X num t_features".format(a.shape))
	print("Testing data: {} - avg num_labels per instance".format(sum(labels_per_doc_b)/len(b)))
	print("Testing data: {} - num leaf labels".format(len(leaf_labels_b)))
	
	# Hierarchical categories
	hier = read_hier(args.category)
	print("Hierarchical categories: {} - num class labels".format(len(hier)))


'''
OUTPUT: LWIKI
-------------
Training data: (2365436, 1617899) - num documents X num features
Training data: 3.2620557055866235 - avg num_labels per instance
Training data: 325056 - num leaf labels
Testing data: (452167, 1617864) - num t_documents X num t_features
Testing data: 2.0 - avg num_labels per instance
Testing data: 452168 - num leaf labels (t_documents + 1 because of 0 label associated with every document)
Hierarchical categories: 478020 - num class labels

OUTPUT: SWIKI
-------------
Namespace(category='../swiki/data/cat_hier.txt', testing='../swiki/data/test.txt', training='../swiki/data/train.txt')
Training data: (456886, 2085164) - num documents X num features
Training data: 1.8446439593246455 - avg num_labels per instance
Training data: 36504 - num leaf labels
Testing data: (81262, 2085161) - num t_documents X num t_features
Testing data: 1.0 - avg num_labels per instance
Testing data: 1 - num leaf labels
Hierarchical categories: 50312 - num class labels
'''
