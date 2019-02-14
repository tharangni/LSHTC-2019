# https://github.com/gcdart/MulticlassClassifier/blob/master/src/ml/LogisticRegression.java
# https://www.kaggle.com/c/lshtc/discussion/6911#38233 - preprocessing: multilabels comma should not have spaces
# https://www.kaggle.com/c/lshtc/discussion/14048 - dataset statistics
## currently reading LWIKI dataset 

import os
from joblib import Memory
from sklearn.datasets import load_svmlight_file


mem = Memory("../mycache")

@mem.cache
def get_data(filename):

    data = load_svmlight_file(filename, multilabel=True)
    
    return data[0], data[1]


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
	
	# x, a - contains all the feat:value data
	# y, b - contains all the labels (per line)
	x, y = get_data("../lwiki/train-remapped/train-remapped.csv")
	a, b = get_data("../lwiki/test-remapped/test-remapped.csv")
	hier = read_hier("../lwiki/hierarchy/hierarchy.txt")

	leaf_labels_y, labels_per_doc_y = label_extractor(y)
	leaf_labels_b, labels_per_doc_b = label_extractor(b)

	# Training data stuff
	print("Training data: {} - num documents X num features".format(x.shape))
	print("Training data: {} - avg num_labels per instance".format(sum(labels_per_doc_y)/len(y)))
	print("Training data: {} - num leaf labels".format(len(leaf_labels_y)))
	

	# Testing data stuff
	print("Testing data: {} - num t_documents X num t_features".format(a.shape))
	print("Testing data: {} - avg num_labels per instance".format(sum(labels_per_doc_b)/len(b)))
	print("Testing data: {} - num leaf labels".format(len(leaf_labels_b)))
	

	# Hierarchical categories
	print("Hierarchical categories: {} - num class labels".format(len(hier)))


'''
OUTPUT
------
Training data: (2365436, 1617899) - num documents X num features
Training data: 3.2620557055866235 - avg num_labels per instance
Training data: 325056 - num leaf labels
Testing data: (452167, 1617864) - num t_documents X num t_features
Testing data: 2.0 - avg num_labels per instance
Testing data: 452168 - num leaf labels
Hierarchical categories: 478020 - num class labels
'''