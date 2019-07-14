'''TODO: Implement Logistic Multi-label classifier'''

import time
import torch
import scipy
import random
import numpy as np
import scipy.sparse

from tqdm import tqdm
from logbase import LogisticBase
from logcost import LogisticCost
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, precision_score, recall_score, precision_score, accuracy_score, roc_auc_score

class LogisticScut(LogisticBase):

    def __init__(self, *args,**kwargs):
        super(LogisticScut, self).__init__(*args,**kwargs)
        self.threshold = 0
    
    def scut_threshold(self, scores, y_true):

        sorted_scores = np.sort(scores)[::-1]
        here = list(set(sorted_scores))

        midpoints = (sorted_scores[1:] + sorted_scores[:-1])/2
        mid = sorted(list(set(midpoints)), reverse=True)
        best_thresh, best_f1, store_i, best_auc = sorted_scores[0], 0, 0, 0
        lim = round(0.01*len(mid))
        
        for i, threshold in enumerate(mid):
            y_pred = np.array(scores > threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_auc = roc_auc_score(y_true, y_pred)
                best_thresh = threshold
                best_f1 = f1
                store_i = i
                
            if i == store_i+100:
                break

        print("th: {} f1: {} best_auc: {}".format(best_thresh, best_f1, best_auc))
        return best_thresh, best_f1


    def fit(self, X, y, cost_vector = []):
        '''Train the model'''

        # if cost vector not provided the default cost is 1 for all examples
        num_examples  = X.shape[0]
        if len(cost_vector) == 0:
            cost_vector = np.ones((num_examples,1))
        else:
            cost_vector = np.array(cost_vector).reshape((num_examples,1))

        # split dataset first, then tune the threshold
        if np.sum(y==1) < 2:
            self.threshold = 0
        else:
            niter = 5
            dev_thresholds = []
            sss = StratifiedShuffleSplit(n_splits =niter, test_size=0.3)
            for train_index, dev_index in sss.split(X, y):
                X_train, X_dev = X[train_index,:], X[dev_index,:]
                y_train, y_dev = y[train_index], y[dev_index]
                
                dev_base_model = LogisticCost(rho=self.rho, intercept_scaling=self.intercept_scaling, 
                    w_n = self.W, w_pi=self.W_prev, children=self.children, mod_cn=self.len_cn)
                dev_base_model.fit(X_train, y_train, cost_vector[train_index])
                dev_scores = dev_base_model.decision_function(X_dev) # oh o
                
                prelim, best_f1 = self.scut_threshold(dev_scores, y_dev)
                if best_f1 == 0:
                    dev_thresholds.append(prelim)
                    break
                else:
                    dev_thresholds.append(prelim)
                
            self.threshold = np.mean(dev_thresholds)

        base_model = LogisticCost(rho=self.rho, intercept_scaling=self.intercept_scaling, 
            w_n = self.W, w_pi=self.W_prev, children=self.children, mod_cn=self.len_cn)
        base_model.fit(X, y, cost_vector)
        self.base_model = base_model

    def predict(self, X):
        '''predict the labels of each instance'''
        scores = self.base_model.decision_function(X)
        y_pred = np.array(scores > self.threshold).astype(int)
        return y_pred.flatten()

