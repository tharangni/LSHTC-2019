import numpy as np
import pandas as pd

from collections import OrderedDict

def rr_reader(filename):
	'''
	- create a dataframe from the input data
	- returns a dataframe that has ["labels", "feat_tf"]
	'''
	num_entries = 1000
	df = pd.DataFrame()
	
	with open(filename, "r") as f:
		head = [next(f) for x in range(num_entries)] # retrieve only `n` docs
		for i, line in enumerate(head): # change to f/head depending on your needs
			instance = line.strip().split()
			labels = instance[0]
			doc_dict = OrderedDict()
			temp_dict = {}

			for pair in instance[1:]:
				feat = pair.split(":")
				if int(feat[0]) not in temp_dict:
					temp_dict[int(feat[0])] = int(feat[1])

			for key in sorted(temp_dict.keys()):
				doc_dict[key] = temp_dict[key]
				
			temp_df = pd.DataFrame(data = [ labels, doc_dict ]).T
			df = df.append(temp_df, ignore_index=True)
	
	df.columns = ["labels", "feat_tf"]
	df["labels"] = df["labels"].apply( lambda x: list(map(int, x.split(",")))  )
	return df

if __name__ == '__main__':
	df = rr_reader("../swiki/data/train.txt")
	print(df.head(5))