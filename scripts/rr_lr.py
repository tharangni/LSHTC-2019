import numpy as np
import pandas as pd

def rr_reader(data, labels):
	'''
	create a dataframe from the data-label pair
	'''

	num_entries = data.shape[0]
	df = pd.DataFrame(columns=['doc_id', 'labels', 'feat_tf'])

	return df