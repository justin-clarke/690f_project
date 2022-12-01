# SAIL-ON object classifier training 

import sklearn
# from sklearn import utils
# from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn import cluster
from scipy.special import expit
import numpy as np
import pickle
import time
import random
import json

print("sklearn version: " + sklearn.__version__)


# function to hotpatch
def _initialize_parameters_hotpatch(self, X, random_state):
	"""Initialize the model parameters.
	Parameters
	----------
	X : array-like, shape  (n_samples, n_features)
	random_state : RandomState
		A random number generator instance that controls the random seed
		used for the method chosen to initialize the parameters.
	"""
	n_samples, _ = X.shape

	if self.init_params == 'kdl':
		# print("Initialinzing with KDL parameters")
		resp = np.zeros((n_samples, self.n_components))
		label = [label_to_int[i] for i in train_labels]
		resp[np.arange(n_samples), label] = 1
	elif self.init_params == 'kmeans':
		resp = np.zeros((n_samples, self.n_components))
		label = cluster.KMeans(n_clusters=self.n_components, n_init=1,
							   random_state=random_state).fit(X).labels_
		resp[np.arange(n_samples), label] = 1
	elif self.init_params == 'random':
		resp = random_state.rand(n_samples, self.n_components)
		resp /= resp.sum(axis=1)[:, np.newaxis]
	else:
		raise ValueError("Unimplemented initialization method '%s'"
						 % self.init_params)

	self._initialize(X, resp)


# hotpatch to initialize class responsibilities to known classes for training data
sklearn.mixture._base.BaseMixture._initialize_parameters = _initialize_parameters_hotpatch

# must be defined for score_samples_hotpatch to work
def _check_X(X, n_components=None, n_features=None, ensure_min_samples=1):
    """Check the input data X.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    n_components : int
    Returns
    -------
    X : array, shape (n_samples, n_features)
    """
    X = check_array(X, dtype=[np.float64, np.float32],
                    ensure_min_samples=ensure_min_samples)
    if n_components is not None and X.shape[0] < n_components:
        raise ValueError('Expected n_samples >= n_components '
                         'but got n_components = %d, n_samples = %d'
                         % (n_components, X.shape[0]))
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError("Expected the input data X have %d features, "
                         "but got %d features"
                         % (n_features, X.shape[1]))
    return X



def score_samples_hotpatch(self, X):
        """Compute the weighted log probabilities for each sample.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log probabilities of each data point in X.
        """
        print()
        print("***** Using score_samples_hotpatch  *****")
        print()

        check_is_fitted(self)
        X = _check_X(X, None, self.means_.shape[1])

        return self._estimate_weighted_log_prob(X)

sklearn.mixture._base.BaseMixture.score_samples = score_samples_hotpatch


label_to_int = {'Platform': 0,
				'TNT': 1,
				'bird_black': 2,
				'bird_blue': 3,
				'bird_red': 4,
				'bird_white': 5,
				'bird_yellow': 6,
				'butterfly': 7,
				'ice_circle': 8,
				'ice_circle_small': 9,
				'ice_rect_big': 10,
				'ice_rect_fat': 11,
				'ice_rect_medium': 12,
				'ice_rect_small': 13,
				'ice_rect_tiny': 14,
				'ice_square_hole': 15,
				'ice_square_small': 16,
				'ice_square_tiny': 17,
				'ice_triang': 18,
				'ice_triang_hole': 19,
				'magician': 20,
				'pig_basic_small': 21,
				'stone_circle': 22,
				'stone_circle_small': 23,
				'stone_rect_big': 24,
				'stone_rect_fat': 25,
				'stone_rect_medium': 26,
				'stone_rect_small': 27,
				'stone_rect_tiny': 28,
				'stone_square_hole': 29,
				'stone_square_small': 30,
				'stone_square_tiny': 31,
				'stone_triang': 32,
				'stone_triang_hole': 33,
				'wizard': 34,
				'wood_circle': 35,
				'wood_circle_small': 36,
				'wood_rect_big': 37,
				'wood_rect_fat': 38,
				'wood_rect_medium': 39,
				'wood_rect_small': 40,
				'wood_rect_tiny': 41,
				'wood_square_hole': 42,
				'wood_square_small': 43,
				'wood_square_tiny': 44,
				'wood_triang': 45,
				'wood_triang_hole': 46,
				'worm': 47}



model_save_name = "model_gmm_7200levels_5samples_v058_docker.sav"
print_errors = 0

print("Reading Data")
data = [] 
labels = []
f = open("non-novel_7200levels_5samples_v058_docker.txt")
line_count = 0
for iline in f:
	line_count += 1
	if line_count % 100000 == 0:
		print("Reading line: " + str(line_count))
	l = iline.strip()
	l = eval(l)
	data.append(l[:-1])
	labels.append(l[-1])
f.close()

print("Total data size: " + str(len(data)))


# num_test_points = int(len(data) * 0.20)
# print("Selecting " + str(num_test_points) + " test data points")
# test_indices = random.sample([i for i in range(len(data))], num_test_points)
# test_indices.sort()


# print("Dividing data and labels into test and train sets")
# data = np.array(data)
# labels = np.array(labels)
# print("- Slicing test data arrays")
# test_data = data[test_indices]
# test_labels = labels[test_indices]
# print("- Selecting train data")

# train_data = []
# train_labels = []
# data_idx = 0
# test_idxs_idx = 0
# test_index_val = test_indices[test_idxs_idx]
# while data_idx < len(data):
# 	if data_idx % 250000 == 0:
# 		print("--  processing data row: " + str(data_idx))
# 	if  data_idx != test_index_val:
# 		train_data.append(data[data_idx])
# 		train_labels.append(labels[data_idx])
# 		data_idx += 1
# 	else:
# 		data_idx += 1
# 		if test_idxs_idx == len(test_indices) - 1:
# 			train_data.extend(data[data_idx:])
# 			train_labels.extend(labels[data_idx:])
# 			data_idx = len(data)
# 		else:
# 			test_idxs_idx += 1
# 			test_index_val = test_indices[test_idxs_idx]


# print("Train data size: " + str(len(train_data)))
# print("Train label size: " + str(len(train_labels)))
# print("Test data size: " + str(len(test_data)))
# print("Test label size: " + str(len(test_labels)))


train_data = np.array(data)
train_labels = np.array(labels)

start = time.perf_counter()
print("Training")
model = GaussianMixture(n_components=len(label_to_int), init_params='kdl', max_iter=1).fit(train_data) 
end = time.perf_counter()
total = end - start
print("Done")
print("Training time: " + str(total))

# print("Predicting")
# preds = model.predict(test_data)

# error_count = 0
# for i in range(len(preds)):
# 	if preds[i] != label_to_int[test_labels[i]]:
# 		error_count += 1

# print("Accuracy: " + str((len(preds) - error_count) / len(preds)))

# save model
print("saving model")
pickle.dump(model, open(model_save_name, 'wb'))
print("Done")

# probs=model.predict_proba(data)



