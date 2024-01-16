import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import re
import sys

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
	- Cate!pip install numpy==1.20 SimpleImputer("most_frequent")
	'''

	if ohe:
		enc_cat = OneHotEncoder(dtype=np.int64, sparse_output=False)
	else:
		enc_cat =  OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=9999)

	num_transformer = Pipeline([
		("scaler", MinMaxScaler()),
		("imputer", IterativeImputer(random_state=11,       ### Set random state so models can be compared with same split!
						   max_iter=10,
						   verbose=0,
						   tol=0.001,
						   sample_posterior=True,
						   n_nearest_features=5))])

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



def classify_CV(clf, X, y, sel_variables, outer_cv=5, model="", save_to="", title = "",saveIndivdualPred=True,**kwargs):
	
	# #####################################    
	# Classification  
	# Run classifier with CV (from gridsearch input, so essentially DCV)

	kf = StratifiedKFold(n_splits=outer_cv)
	df_results = pd.DataFrame()  	
	
	all_y = []
	all_probs = []
	all_predicts = []
	all_y_pid = []
	 
	for i, (train, test) in enumerate(kf.split(X, y)):
		  
		
		X_train = X.iloc[train].copy()
		y_train = y.iloc[train].copy()
		
		X_test = X.iloc[test].copy() 
		y_test_pid = y.iloc[test].index
		y_test = y.iloc[test].values.astype("int").copy() 
		
		''' Impute & Scale '''
		num_columns = X_train.select_dtypes(include=["float64"]).columns
		bin_columns = X_train.select_dtypes(include=["int64"]).columns
		cat_columns = X_train.select_dtypes(include=["object"]).columns
		### Train set
		preprocessor = imputation_scaling(num_columns, bin_columns, cat_columns, X_train)     
		X_train_imputed = preprocessor.fit_transform(X_train)
		X_train_imputed = SupervisedSelector(preprocessor, sel_variables).transform(X_train_imputed)
		print(X_train_imputed.shape)
		### Test set
		preprocessor_test = imputation_scaling(num_columns, bin_columns, cat_columns, X_test)     
		X_test_imputed = preprocessor_test.fit_transform(X_test)
		X_test_imputed = SupervisedSelector(preprocessor_test, sel_variables).transform(X_test_imputed)
		print(X_test_imputed.shape)
		
		''' Train Model '''
		clf.fit(X_train_imputed, y_train) #hyperparameter tuning
		
	
		y_pred = clf.predict(X_test_imputed)  
		y_predProba = clf.predict_proba(X_test_imputed)[:,1].tolist()

		all_y = all_y + y_test.tolist()
		all_probs = all_probs + y_predProba
		all_predicts = all_predicts + y_pred.tolist()
		all_y_pid = all_y_pid + y_test_pid.tolist()
	

	all_y = np.array(all_y)
	all_probs = np.array(all_probs)
	all_predicts = np.array(all_predicts)
	tn, fp, fn, tp = confusion_matrix(all_y, all_predicts).ravel()
	prec = (tp/(tp+fp))
	rec = (tp/(tp+fn))
	f1 = (tp/(tp+0.5*(fp+fn)))
	acc = ((tp+tn)/(tp+tn+fp+fn))
	# roc and auc
	fpr, tpr, _ = roc_curve(all_y,all_probs)
	auc_val = auc(fpr, tpr)
	precision_recall = average_precision_score(all_y,all_probs)
	results = {"precision": prec,
		"recall": rec,
		"f1": f1,
		"accuracy": acc,
		"model": model,
		"auc": auc_val,
		"average_prec":precision_recall
		}  
	df_results = pd.concat([df_results, pd.Series(results)])  

	final_results = {}
	final_results['df_results'] = df_results  
	if saveIndivdualPred:
		df_indPred = pd.DataFrame(np.column_stack((all_y_pid, all_y, all_probs, all_predicts)), columns=["pid","y_true", "y_predProb", "y_pred"])
		final_results["df_indPred"] = df_indPred
	return final_results
