# classifies the objects in a single ground truth list
# runs a secondary novelty classifier to identify novelty
# in shape or material.  

from scipy.special import expit
import os.path as osp
from sklearn.mixture import GaussianMixture
import sklearn
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn import cluster
import numpy as np
import pickle
import time
import random
import json


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
        # print()
        # print("***** Using score_samples_hotpatch  *****")
        # print()

        check_is_fitted(self)
        X = _check_X(X, None, self.means_.shape[1])

        return self._estimate_weighted_log_prob(X)




class Classifier(object):
	def __init__(self):

		# hotpatch to stop normalization of class probabilities
		sklearn.mixture._base.BaseMixture.score_samples = score_samples_hotpatch

		classifer_path = osp.join('sciencebirds', 'classifier')
		self.model = pickle.load(open(osp.join(classifer_path, 'model_gmm_7200levels_5samples_v058_docker.sav'), 'rb'))
		self.shape_model = pickle.load(open(osp.join(classifer_path, 'model_gmm_7200levels_5samples_v058_shape_docker.sav'), 'rb'))
		self.material_model = pickle.load(open(osp.join(classifer_path, 'model_gmm_7200levels_5samples_v058_material.sav'), 'rb'))
		# self.model = pickle.load(open('model_gmm_7200levels_5samples_v058_docker.sav', 'rb'))
		# self.shape_model = pickle.load(open('model_gmm_7200levels_5samples_v058_shape_docker.sav', 'rb'))
		# self.material_model = pickle.load(open('model_gmm_7200levels_5samples_v058_material.sav', 'rb'))

		self.thresholds = {
			'Platform' : -24361.38642730479,
			'TNT' : 1490.7589607038003,
			'bird_blue' : 1605.1154668693296,
			'bird_red' : 1609.0145784905553,
			'bird_white' : 1601.6520709227163,
			'bird_yellow' : 1603.7285525867437,
			'butterfly' : 1609.2952417655608,
			'ice_circle' : 1609.5166029692684,
			'ice_circle_small' : 1575.6217551016898,
			'ice_rect_big' : 1549.704104449636,
			'ice_rect_fat' : 1409.9469153845007,
			'ice_rect_medium' : 1056.5133072239648,
			'ice_rect_small' : 1370.6443185124826,
			'ice_rect_tiny' : 936.2855674306505,
			'ice_square_hole' : 1426.4435748391272,
			'ice_square_small' : 1478.797152851145,
			'ice_square_tiny' : 1380.1195999053305,
			'ice_triang' : 1565.3221684869113,
			'ice_triang_hole' : 1450.7714131312866,
			'magician' : 1606.3906531265118,
			'pig_basic_small' : 1545.3035977900042,
			'stone_circle' : 1590.4925668439178,
			'stone_circle_small' : 1410.525384052307,
			'stone_rect_big' : 1582.50761233491,
			'stone_rect_fat' : 1431.224907098644,
			'stone_rect_medium' : 1582.50761233491,
			'stone_rect_small' : 1388.5588233424796,
			'stone_rect_tiny' : 1359.5377991638004,
			'stone_square_hole' : 1491.6387145497001,
			'stone_square_small' : 1378.5955367458641,
			'stone_square_tiny' : 990.0674422200443,
			'stone_triang' : 1569.5363370187624,
			'stone_triang_hole' : 1368.146544006328,
			'wizard' : 1603.0546317957844,
			'wood_circle' : 1598.074672623617,
			'wood_circle_small' : 1590.9222139340889,
			'wood_rect_big' : 559.8843630980731,
			'wood_rect_fat' : 1431.4932476301374,
			'wood_rect_medium' : 1493.0137833796102,
			'wood_rect_small' : 1369.7977692952693,
			'wood_rect_tiny' : 841.6890769495143,
			'wood_square_hole' : 1431.2797437273127,
			'wood_square_small' : 1562.0605194325406,
			'wood_square_tiny' : 1519.981156845018,
			'wood_triang' : 1605.8507023777042,
			'wood_triang_hole' : 1563.779199681592,
			'worm' : 1527.7470676838436
		}
		self.shape_thresholds = {
			'Platform' : -138.8817578898678,
			'TNT' : 91.21483700478056,
			'bird_blue' : 74.60620835819607,
			'bird_red' : 82.0520611164455,
			'bird_white' : 79.11614420411347,
			'bird_yellow' : 79.61633374856997,
			'butterfly' : 77.78041924287267,
			'circle' : 57.70589991646306,
			'circle_small' : 36.91218233040006,
			'magician' : 74.99044618684042,
			'pig_basic_small' : 17.471136357130337,
			'rect_big' : 68.6505335755852,
			'rect_fat' : -98.69074137428835,
			'rect_medium' : 55.508101199356965,
			'rect_small' : -103.0603602596863,
			'rect_tiny' : -180.90662059272586,
			'square_hole' : -133.54914728868064,
			'square_small' : -71.31586762582278,
			'square_tiny' : -104.8750963520078,
			'triang' : 20.878857829345925,
			'triang_hole' : -267.259043956384,
			'wizard' : 71.1273326637823,
			'worm' : -3.8101755186592636
		}
		self.material_thresholds = {
			'Platform' : 1517.2400978893284,
			'TNT' : 1527.6687335439428,
			'bird' : 1309.9082609615907,
			'butterfly' : 1527.2593867117973,
			'ice' : 1498.2719924819926,
			'magician' : 1527.1302619542835,
			'pig' : 1527.2798751055532,
			'stone' : 1305.3468194624634,
			'wizard' : 1528.1258663482724,
			'wood' : 1492.765506670465,
			'worm' : 1527.2013410755158
		}
		self.simba_name_dict = {
		'Ground' : 'ground',
		'Trajectory' : 'trajectory',
		'Slingshot' : 'slingshot',
		'Platform' : 'platform',
		'TNT' : 'tnt',
		'bird_black' : 'black_bird',
		'bird_blue' : 'blue_bird',
		'bird_red' : 'red_bird',
		'bird_white' : 'white_bird',
		'bird_yellow' : 'yellow_bird', 
		'ice_circle' : 'ice',
		'ice_circle_small' : 'ice',
		'ice_rect_big' : 'ice',
		'ice_rect_fat' : 'ice',
		'ice_rect_medium' : 'ice',
		'ice_rect_small' : 'ice',
		'ice_rect_tiny' : 'ice',
		'ice_square_hole' : 'ice',
		'ice_square_small' : 'ice',
		'ice_square_tiny' : 'ice',
		'ice_triang' : 'ice',
		'ice_triang_hole' : 'ice',
		'pig_basic_medium' : 'pig',
		'pig_basic_small' : 'pig',
		'stone_circle' : 'stone',
		'stone_circle_small' : 'stone',
		'stone_rect_big' : 'stone',
		'stone_rect_fat' : 'stone',
		'stone_rect_medium' : 'stone',
		'stone_rect_small' : 'stone',
		'stone_rect_tiny' : 'stone',
		'stone_square_hole' : 'stone',
		'stone_square_small' : 'stone',
		'stone_square_tiny' : 'stone',
		'stone_triang' : 'stone',
		'stone_triang_hole' : 'stone',
		'wood_circle' : 'wood',
		'wood_circle_small' : 'wood',
		'wood_rect_big' : 'wood',
		'wood_rect_fat' : 'wood',
		'wood_rect_medium' : 'wood',
		'wood_rect_small' : 'wood',
		'wood_rect_tiny' : 'wood',
		'wood_square_hole' : 'wood',
		'wood_square_small' : 'wood',
		'wood_square_tiny' : 'wood',
		'wood_triang' : 'wood',
		'wood_triang_hole' : 'wood',
		'butterfly' : 'butterfly',
		'wizard' : 'wizard',
		'magician' : 'magician',
		'worm' : 'worm',
		'novel' : 'novel'
		}
		
		# int to label mapping for gmm classifier
		self.int_to_label =  {
		0: 'Platform', 
		1: 'TNT', 
		2: 'bird_black', 
		3: 'bird_blue', 
		4: 'bird_red', 
		5: 'bird_white', 
		6: 'bird_yellow', 
		7: 'butterfly', 
		8: 'ice_circle', 
		9: 'ice_circle_small', 
		10: 'ice_rect_big', 
		11: 'ice_rect_fat', 
		12: 'ice_rect_medium', 
		13: 'ice_rect_small', 
		14: 'ice_rect_tiny', 
		15: 'ice_square_hole', 
		16: 'ice_square_small', 
		17: 'ice_square_tiny', 
		18: 'ice_triang', 
		19: 'ice_triang_hole', 
		20: 'magician', 
		21: 'pig_basic_small', 
		22: 'stone_circle', 
		23: 'stone_circle_small', 
		24: 'stone_rect_big', 
		25: 'stone_rect_fat', 
		26: 'stone_rect_medium', 
		27: 'stone_rect_small', 
		28: 'stone_rect_tiny', 
		29: 'stone_square_hole', 
		30: 'stone_square_small', 
		31: 'stone_square_tiny', 
		32: 'stone_triang', 
		33: 'stone_triang_hole', 
		34: 'wizard', 
		35: 'wood_circle', 
		36: 'wood_circle_small', 
		37: 'wood_rect_big', 
		38: 'wood_rect_fat', 
		39: 'wood_rect_medium', 
		40: 'wood_rect_small', 
		41: 'wood_rect_tiny', 
		42: 'wood_square_hole', 
		43: 'wood_square_small', 
		44: 'wood_square_tiny', 
		45: 'wood_triang', 
		46: 'wood_triang_hole', 
		47: 'worm'}

		self.shape_int_to_label = {
		0:'Platform',
		1:'TNT',
		2:'bird_black',
		3:'bird_blue',
		4:'bird_red',
		5:'bird_white',
		6:'bird_yellow',
		7:'circle',
		8:'circle_small',
		9:'pig_basic_small',
		10:'rect_big',
		11:'rect_fat',
		12:'rect_medium',
		13:'rect_small',
		14:'rect_tiny',
		15:'square_hole',
		16:'square_small',
		17:'square_tiny',
		18:'triang',
		19:'triang_hole',
		20:'butterfly',
		21:'magician',
		22:'wizard',
		23:'worm'}

		self.material_int_to_label = {
		0:'Platform',
		1:'TNT',
		2:'bird',
		3:'ice',
		4:'pig',
		5:'stone',
		6:'wood',
		7:'butterfly',
		8:'magician',
		9:'wizard',
		10:'worm'}

	
	def simba_name_converter(self, name):
		return self.simba_name_dict[name]


	# area of a polygon with an arbitrary number of vertices
	def polygon_area(self, x, y):
		correction = x[-1] * y[0] - y[-1]* x[0]
		main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
		return 0.5*np.abs(main_area + correction)


	# takes a ground truth list and updates it to include additional features
	def _gt_update(self, gt):
		i = 0
		while i < len(gt['features']):
			feature = gt['features'][i]
			if feature['properties']['label'] not in ['Ground', 'Trajectory', 'Slingshot']:
				
				# add area, number of vertices, and number of contours of each object
				feature['properties']['contour_count'] = len(feature['geometry']['coordinates'])
				ob_verts = feature['geometry']['coordinates'][0] # first list of vertices is the object outline
				xs = np.array([j[0] for j in ob_verts])
				ys = np.array([j[1] for j in ob_verts])
				a = self.polygon_area(xs, ys)
				# remainder of the contours are obj cutouts
				cutout_areas = [self.polygon_area(np.array([j[0] for j in c]), np.array([j[1] for j in c]))
															for c in feature['geometry']['coordinates'][1:]]
				# we subtract the areas of the cutout contours to get the object area
				# feature['properties']['area'] = a - sum(cutout_areas). # Model was not trained this way. Will retrain and fix later.
				feature['properties']['area'] = a
				feature['properties']['vertex_count'] = len(xs)

				# add ratio of height / width
				width = abs(max(xs) - min(xs))
				height = abs(max(ys) - min(ys))
				shape_ratio = height / width
				feature['properties']['shape_ratio'] = shape_ratio

				edges = []
				for j in range(-1, len(xs) - 1):
					e = np.sqrt((xs[j + 1] - xs[j])**2 + (ys[j + 1] - ys[j])**2)
					edges.append(e)
				shortest = min(edges)
				longest = max(edges)
				mean_edge = np.mean(edges)
				median_edge = np.median(edges)
				edge_sum = sum(edges)
				feature['properties']['shortest_edge'] = shortest
				feature['properties']['longest_edge'] = longest
				feature['properties']['mean_edge'] = mean_edge
				feature['properties']['median_edge'] = median_edge
				feature['properties']['edge_sum'] = edge_sum

			else:
				# assign Ground, Trajectory, and Slingshot 
				# 0 for all numeric fields
				# ensures keys exist for every object
				feature['properties']['area'] = 0
				feature['properties']['vertex_count'] = 0
				feature['properties']['contour_count'] = 0
				feature['properties']['shortest_edge'] = 0
				feature['properties']['longest_edge'] = 0
				feature['properties']['mean_edge'] = 0
				feature['properties']['median_edge'] = 0
				feature['properties']['edge_sum'] = 0
				feature['properties']['shape_ratio'] = 0

			i += 1
		return gt


	# converts ground truth data into feature vectors for the classifier 
	def _get_data(self, gt):
		self._gt_update(gt)

		# convert each object to a feature vector
		data = []
		for i in range(len(gt['features'])):
			feature = gt['features'][i]
			if feature['properties']['label'] not in ['Ground', 'Trajectory', 'Slingshot']:
				# create record and append to training_data
				unit = []
				# add numeric variables
				unit.append(feature['properties']['area'])
				unit.append(feature['properties']['shape_ratio'])
				unit.append(feature['properties']['shortest_edge'])
				unit.append(feature['properties']['longest_edge'])
				unit.append(feature['properties']['mean_edge'])
				unit.append(feature['properties']['median_edge'])
				unit.append(feature['properties']['edge_sum'])
				unit.append(feature['properties']['contour_count'])
				
				# add categorical variables
				vc = feature['properties']['vertex_count']
				for c in range(4, 20):
					if c == vc:
						unit.append(1)
					else:
						unit.append(0)

				# add colormap as 256 element vector
				colormap = feature['properties']['colormap']
				colors = [int(j['color']) for j in colormap]
				vals = [float(j['percent']) for j in colormap]
				full_map = [0 for j in range(0, 256)]
				index = 0
				while index < len(colors):
					c = colors[index]
					full_map[c] = vals[index]
					index += 1
				# feature['properties']['full_colormap'] = full_map
				unit.extend(full_map)

				data.append(unit)
		return data


	# Level 3 Novelty Detector
	def _l3(self, gt):
		cushion = 17 # number of pixels of overlap allowed when comparing object locations
		novelty_detection_flag = 0

		slingshot = [i for i in gt['features'] if i['properties']['label'] == 'Slingshot']
		
		# ----------------------------------------------------------
		# check for multiple slingshots
		if len(slingshot) > 1:
			novelty_detection_flag = 1

		slingshot = slingshot[0]

		# ----------------------------------------------------------
		# check slingshot location (on left half of screen)
		slingshot_coords = slingshot['geometry']['coordinates'][0]
		slingshot_xs = np.array([j[0] for j in slingshot_coords])
		slingshot_ys = np.array([j[1] for j in slingshot_coords])
		if len([i for i in slingshot_xs if i > 320]) > 0:
			novelty_detection_flag = 1

		# ----------------------------------------------------------
		# check that at least one bird is classified
		birds = [i for i in gt['features'] if 'bird' in i['properties']['kdl_label']]
		if len(birds) == 0:
			novelty_detection_flag = 1

		# ----------------------------------------------------------
		# sum all colormaps, then count number of values > 0. If it is less than ~10, probably novel?
		objects = [ob for ob in gt['features'] if ob['properties']['label'] not in ['Ground', 'Trajectory', 'Slingshot']]
		full_map = [0 for j in range(0, 256)]
		for ob in objects:
			colormap = ob['properties']['colormap']
			for i in colormap:
				full_map[i['color']] += i['percent']
		if len([i for i in full_map if i > 0]) < 10:
			novelty_detection_flag = 1

		return novelty_detection_flag


	# classifies all objects in a single ground truth list
	def classify(self, gt, switchoff=False):
		if switchoff:
			return gt
		if len(gt) < 1: # if gt is empty 
			return gt

		data = self._get_data(gt)
		if len(data) < 1:  # if there are only basic objects data will be empty
			return gt

		data = np.array(data) # format for sklearn
		predictions = self.model.predict(data)  # list of integers corresponding to integer components
		labels = [self.int_to_label[i] for i in predictions]
		
		# uncomment to classify medium pigs as pig_basic_samll
		# for i in range(len(labels)):
		# 	if labels[i] == 'pig_basic_medium':
		# 		labels[i] == 'pig_basic_small'

		probs = self.model.score_samples(data)
		p = [np.max(i) for i in probs]  # max class probability across all classes for each unit 
		data2 = [] # data for secondary classifier
		labels2 = [] # labels for secondary classifier
		label_index = 0
		for i in range(len(gt['features'])):
			feature = gt['features'][i]
			if feature['properties']['label'] not in ['Ground', 'Trajectory', 'Slingshot']:
				feature['properties']['kdl_label'] = labels[label_index]
				if feature['properties']['kdl_label'] in ['TNT', 'Platform', 'butterfly', 'magician', 'wizard', 'worm']:
					mat = feature['properties']['kdl_label']
				else:
					mat = ''
					ch = 0
					while feature['properties']['kdl_label'][ch] != '_':
						mat = mat + feature['properties']['kdl_label'][ch]
						ch += 1
				feature['properties']['material'] = mat
				# set shape
				if mat in ['TNT', 'Platform', 'butterfly', 'magician', 'wizard', 'worm']:
					feature['properties']['shape'] = mat
				elif mat in ['pig', 'bird']:
					feature['properties']['shape'] = feature['properties']['kdl_label']
				else:
					feature['properties']['shape'] = feature['properties']['kdl_label'][ch+1:]
				
				# detect novelty
				if p[label_index] < self.thresholds[feature['properties']['kdl_label']]:
					feature['properties']['novel'] = 1
					feature['properties']['max_class_prob'] = p[label_index]
					data2.append(data[label_index])
					labels2.append(labels[label_index])
				else:
					feature['properties']['novel'] = 0

				label_index += 1

			else:
				feature['properties']['kdl_label'] = feature['properties']['label']
				feature['properties']['material'] = feature['properties']['label']
				feature['properties']['shape'] = feature['properties']['label']
				feature['properties']['novel'] = 0

			# set simba_type 
			if feature['properties']['novel'] == 1:
				feature['properties']['simba_type'] = 'novel'
			else:
				feature['properties']['simba_type'] = self.simba_name_converter(feature['properties']['kdl_label'])
				

		# if any novel objects are detected run secondary classifiers
		if len(data2) > 0:
			shape_data = []
			material_data = []
			for i in range(len(data2)):
				shape_data.append(data2[i][:24])
				material_data.append(data2[i][24:])
			# format for sklearn
			shape_data = np.array(shape_data) 
			material_data = np.array(material_data)

			# run secondary classification of novel objects
			material_preds = self.material_model.predict(material_data)
			material_predictions = [self.material_int_to_label[i] for i in material_preds]
			shape_preds = self.shape_model.predict(shape_data)
			shape_predictions = [self.shape_int_to_label[i] for i in shape_preds]

			shape_probs = self.shape_model.score_samples(shape_data)
			shape_probs = [np.max(i) for i in shape_probs]
			material_probs = self.material_model.score_samples(material_data)
			material_probs = [np.max(i) for i in material_probs]

			label_index = 0 
			for i in range(len(gt['features'])):
				feature = gt['features'][i]
				if feature['properties']['label'] not in ['Ground', 'Trajectory', 'Slingshot'] and feature['properties']['novel'] == 1:
					# detect shape novelty
					if shape_probs[label_index] < self.shape_thresholds[feature['properties']['shape']]:
						feature['properties']['shape_novelty'] = 1
						feature['properties']['novel_shape'] = 'novel'

					else:
						feature['properties']['shape_novelty'] = 0
						feature['properties']['novel_shape'] = shape_predictions[label_index]

					# detect material novelty
					if material_probs[label_index] < self.material_thresholds[feature['properties']['material']]:
						feature['properties']['material_novelty'] = 1
						feature['properties']['novel_material'] = 'novel'
					else:
						feature['properties']['material_novelty'] = 0
						feature['properties']['novel_material'] = material_predictions[label_index]

					label_index += 1

		# detect Level 3 Novelty
		gt['l3_detection'] = self._l3(gt)

		return gt







