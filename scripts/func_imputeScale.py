import numpy as np
import pandas as pd

from sklearn import svm, datasets
from sklearn.metrics import auc, roc_curve, average_precision_score
from sklearn.metrics import RocCurveDisplay  
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score,cross_val_predict, LeaveOneOut

from sklearn.metrics import confusion_matrix  
from sklearn.inspection import permutation_importance
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, MinMaxScaler, Binarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.utils import resample



class SupervisedSelector():
	'''
	Part of prediction pipeline - only parses the variables it receives and returns pruned dataset
	''' 
	def __init__(self, preprocessor, features, argument=None):
		self.preprocessor = preprocessor
		self.features = features
		self.argument = argument

	def fit(self, X, y=None):
		return self

	def transform(self, X, y=None):
		columns = [ele[5:] for ele in self.preprocessor.get_feature_names_out()]
		df_X = pd.DataFrame(X, columns=columns)
		self.X_ = df_X.loc[:,self.features]
		return self.X_

	def get_feature_names(self):
		return self.X_.columns.tolist()



def imputation_scaling(num_columns, bin_columns, cat_columns, X, ohe=False):

	'''
	### Imputation ###
	- Numerical features (float64):
		- MICE
		- MinMaxScaler
    - Categorical features (int64):
        - KNN
    - Categorical features (objects):
        - SimpleImputer("most_frequent")
	'''

	if ohe:
		enc_cat = OneHotEncoder(dtype=np.int64, sparse_output=False)
	else:
		enc_cat =  OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=9999)

	num_transformer = Pipeline([
		("imputer", IterativeImputer(random_state=11,       ### Set random state so models can be compared with same split!
						   max_iter=10,
						   verbose=0,
						   tol=0.001,
						   sample_posterior=True,
                           min_value=[0, 0, 0, 0, 0],   # manually fix the minimum value to be imputed for each variable...
						   n_nearest_features=5)),
        ("scaler", MinMaxScaler())])

	bin_transformer = Pipeline([
		("imputer", KNNImputer(n_neighbors=5)),
		("scaler", MinMaxScaler()),
		("binarizer", Binarizer())])
	

	cat_transformer = Pipeline([
		("imputer", SimpleImputer(strategy='most_frequent')),
		("encoding", enc_cat),
		("scaler", MinMaxScaler())          ### to get everything on the same scale in case of no OHE
		])

	preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_columns),
												   ("bin", bin_transformer, bin_columns),
												   ("cat", cat_transformer, cat_columns)])

	return preprocessor.fit(X)



''' Same function but made for pipes; not best way to have two different versions but works '''

class pipe_supervisedSelector():
    def __init__(self, features, argument=None):
        self.features = features
        self.argument = argument

    def fit(self, X, y=None):
        return self

    def transform(self, X,y=None):
        df_X = pd.DataFrame(X)
        X_ = df_X.loc[:,self.features]
        return X_

def pipe_imputation_scaling(num_columns, bin_columns, cat_columns):

    '''
    ### Imputation ###
    - Numerical features (float64):
        - MICE
        - MinMaxScaler
    - Categorical features (int64):
        - KNN
    - Categorical features (objects):
        - SimpleImputer("most_frequent")
    '''

    num_transformer = Pipeline([
        #("scaler", MinMaxScaler()),
        ("imputer", IterativeImputer(random_state=11,  
                           max_iter=10,
                           verbose=0,
                           tol=0.001,
                           sample_posterior=True,
                           min_value=[0]*len(num_columns),   # manually fix the minimum value to be imputed for each variable...
                           n_nearest_features=5)),
        ("scaler", MinMaxScaler())])

    bin_transformer = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5))])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("encoding", OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=9999))
        ])

    preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_columns),
                                                   ("bin", bin_transformer, bin_columns),
                                                   ("cat", cat_transformer, cat_columns)])

    return preprocessor    


def sample_w_replacement(data, n_size: int, stratify=None, random_state=None):
    
    rng = np.random.RandomState(seed=random_state)
    idx = np.arange(data.shape[0])

    if stratify is None:
        train_idx = rng.choice(idx, size=n_size, replace=True)
        test_idx = np.setdiff1d(idx, train_idx, assume_unique=False)
        #print("No stratification",train_idx, test_idx)
        return train_idx, test_idx
    
    else: 
        idx = np.arange(data.shape[0])

        ## Get freq of stratify label
        unique, counts = np.unique(stratify, return_counts=True)
        counts = counts / stratify.size
        # print(unique, counts, "\n")

        ## Determine number of samples needed in each group to keep freq
        sample_counts = np.array([])
        for val in counts:
            sample_counts = np.append(sample_counts, np.ceil(n_size * val)).astype(int)
        # print(unique, sample_counts, "\n")

        ## Resample each class individually, then merge
        all_train_idx = np.array([])
        all_test_idx = np.array([])
        for label, ct in zip(unique, sample_counts):
            samples = idx[stratify == label]
            train_idx_resampled, test_idx_resambled = sample_w_replacement(samples, n_size=ct)
            train_idx_resampled = samples[train_idx_resampled]
            test_idx_resambled = samples[test_idx_resambled]
            ### append to global list
            all_train_idx = np.append(all_train_idx, train_idx_resampled).astype(int)
            all_test_idx = np.append(all_test_idx, test_idx_resambled).astype(int)
        return all_train_idx, all_test_idx


