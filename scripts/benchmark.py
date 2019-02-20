import numpy as np

from time import time

from sklearn import metrics
from sklearn.utils.extmath import density
from sklearn.preprocessing import MultiLabelBinarizer

def benchmark_clf(clf, train_data, train_labels, test_data, test_labels, cats):
	print('+' * 50)
	print("Training: ")
	print(clf)

	# print(y_train)
	# y_train_t = np.argmax(MultiLabelBinarizer().fit_transform(y_train), axis = 1)
	# x_train_t = np.argmax(MultiLabelBinarizer().fit_transform(x_train), axis = 1)

	# x_train_t = x_train
	# y_train_t = y_train

	# x_test_t = x_test
	# y_test_t = y_test
	
	t0 = time()
	clf.fit(train_data, train_labels)
	train_time = time() - t0
	print("train time: {:0.3f}s".format(train_time))

	t0 = time()
	pred = clf.predict(test_data)
	test_time = time() - t0
	print("test time: {:0.3f}s".format(test_time))

	print(pred)

	score = metrics.accuracy_score(test_labels, pred)
	print("accuracy: {:0.3f}".format(score))

	if hasattr(clf, 'coef_'):
		print("dimensionality: {:d}".format(clf.coef_.shape[1]))
		print("density: {:f}".format(density(clf.coef_)))

	# print("classification report:")
	# print(metrics.classification_report(y_test_t, pred, target_names=y_train))

	# print("confusion matrix:")
	# print(metrics.confusion_matrix(y_test_t, pred))

	clf_descr = str(clf).split('(')[0]
	return clf_descr, score, train_time, test_time

