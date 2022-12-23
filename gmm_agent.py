import numpy as np
import sklearn
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, check_random_state
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import mahalanobis

# define agent class
# outlier_check: whether to check the CDF for the expected number of outliers at that distance
class Agent():
    def __init__(self, train_data, num_components=1, recluster_limit=5, outlier_check=False):
        self.num_components = num_components
        self.train_data = np.array([np.array([i[0], i[1]]) for i in train_data])
        self.original_train_data = train_data
#         print("train_data.size: " + str(self.train_data.size))
#         print(self.train_data[0])
        self.model = GaussianMixture(n_components=num_components, 
                                     covariance_type='full', 
                                     n_init=10).fit(train_data)
#         self.lp_threshold = min(self.model.score_samples(self.train_data))  # log probability threshold
        self.m_distances = np.array([self.m_dist(i) for i in self.train_data])
        self.m_threshold = max(np.array([self.m_dist(i) for i in self.train_data]))  # squared mahalanobis threshold
        self.classified_samples = []  # list of dicts with sample, prediction, m_dist, log_prob, metric, novel_flag
              
        self.novel_count = 0
        self.recluster_limit = recluster_limit
        self.outlier_check = outlier_check
        
    # return the log probability of the single sample for the most likely component
    # if return_all=True returns the log prob for all components
    def log_prob(self, sample, return_all=False):
        if len(sample.shape) == 1 or sample.shape[0] == 1:
            if return_all:
                return self.model.score_samples(sample.reshape(1,-1))[0]
#             return max(self.model.score_samples(sample.reshape(1,-1))[0])
#             print("score:")
#             print(self.model.score_samples(sample.reshape(1,-1))[0])
            return max(self.model.score_samples(sample.reshape(1,-1)))
        else:
            if return_all:
                return self.model.score_samples(sample)
            return max(self.model.score_samples(sample))

    # calculate minimum squared mahalanobis distance for a single sample
    # parameters: trained gmm model, data sample to classify, flag to return list of all distances
    # NOTE: can get the same thing for cluster probabilities using model.predict_proba
    def m_dist(self, sample, return_all=False):
        cluster_means = self.model.means_
        covariances = self.model.covariances_
        m_distances = []
        for i in range(len(cluster_means)):
            d = mahalanobis(cluster_means[i], sample, np.linalg.inv(covariances[i]))
            d2 = d**2
            m_distances.append(d2)
        if return_all:
            return m_distances
        else:
            return min(m_distances)
    
    # metric is either "log_prob" or "m_dist"
    def classify(self, sample, metric="m_dist"):
        sample = np.array(sample)
        pred = self.model.predict(sample.reshape(1,-1))[0]
        novel = 0
        md = self.m_dist(sample)
        lp = self.log_prob(sample)
        sample_dict = {"sample":sample, "pred":pred, "m_dist":md, "log_prob":lp}

        if metric == 'log_prob':
            if lp < self.lp_threshold:
                novel = 1
            sample_dict["metric"] = "log_prob"
            sample_dict["novel"] = novel
        elif metric == 'm_dist':
            if md > self.m_threshold:
                novel = 1
            sample_dict["metric"] = "m_dist"
            sample_dict["novel"] = novel
        else:
            print('***  ' + str(metric) + ' IS NOT A VALID METRIC  ***')
 
        self.novel_count += novel
        
        self.classified_samples.append(sample_dict)
        
        # reclustering
        if self.novel_count == self.recluster_limit:
            # check for outliers vs true novelty here
            # find minimum distance of all n potentially novel points
            # calulate probability of finding n points past that distance
            novel_proportion = len([i for i in self.classified_samples if i["novel"] == 1]) / len(self.train_data)
            min_candidate_m_dist = min([i["m_dist"] for i in self.classified_samples])
           
            if self.outlier_check == True:
                if chi2.ppf(1 - novel_proportion, self.train_data.size[1]) < min_candidate_m_dist:
                    # candidate points are truly novel
#                     print(str(self.novel_count) + " novel samples detected. RECLUSTERING!")
                    new_data = np.array([np.array(i["sample"]) for i in self.classified_samples])
                    new_data = np.concatenate((self.train_data, new_data))  # add new samples to training data
                    new_model = GaussianMixture(n_components=self.num_components + 1, 
                                             covariance_type='full', 
                                             n_init=10).fit(new_data)
                    # compare new model with current model and choose the best
                    aic = self.model.aic(new_data)
                    new_aic = new_model.aic(new_data)
                    if new_aic < aic:
                        print("Using new model with " + str(new_model.n_components) + " components")
                        self.model = new_model
                        self.num_components += 1
                        self.novel_count = 0
#                     else:
#                         print("Keeping current model")
                else:
                    print("Candidates are likely outliers and not truly novel")
            else:
#                 print(str(self.novel_count) + " novel samples detected. RECLUSTERING!")
                new_data = np.array([np.array(i["sample"]) for i in self.classified_samples])
                new_data = np.concatenate((self.train_data, new_data))  # add new samples to training data
                new_model = GaussianMixture(n_components=self.num_components + 1, 
                                         covariance_type='full', 
                                         n_init=10).fit(new_data)
                # compare new model with current model and choose the best
                aic = self.model.aic(new_data)
                new_aic = new_model.aic(new_data)
                if new_aic < aic:
#                     print("Using new model with " + str(new_model.n_components) + " components")
                    self.model = new_model
                    self.num_components += 1
                    self.novel_count = 0
#                 else:
#                     print("Keeping current model")
                
            
            self.novel_count = 0 # reset novel count after reclustering