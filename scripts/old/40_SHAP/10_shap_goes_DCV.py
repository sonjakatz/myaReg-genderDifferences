import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
import shap
import pickle

from func_preprocess import read_data, subset_wo_missigness, remove_NA, parseVariables, clean_data, impute_scale 
from func_prediction import imputation_scaling, SupervisedSelector

np.random.seed(11) 

########  SETTINGS  ########
percentBoruta = 80
CV_repeats = 30
n_splits_outer = 10      
n_splits_inner = 3     

### Read dataset and split
PATH_base = "/home/WUR/katz001/PROJECTS/myaReg-genderDifferences"
target = "gender"

''' 
Select features
'''
varFolder = "CV_v3_mgfaRecoded"
vars = f"{target}_bootstrapped_iterativeBoruta_{percentBoruta}perc"    
varPath = f"{PATH_base}/results/20_featureSelection/{varFolder}/{vars}.txt"

# varFolder = "manual_selection"
# vars = f"allVars_mgfaRecoded"    # clinical+bestSterols
# varPath = f"{PATH_base}/results/20_featureSelection/{varFolder}/{vars}.txt"

''' 
Define paths
'''
resultsPath = f"{PATH_base}/results/30_predictions/{varFolder}/{vars}/SHAP"
os.makedirs(resultsPath, exist_ok=True)


''' 1. read data '''
data = read_data(PATH_base, FILENAME="all_data_edited_v3_mgfaRecoded_inverse")

''' 
2. OPTIONAL: make subanalysis for variables with high missigness 
specify which variable should be kept in the dataset; remove rest of vars with too much missigness
'''
#######
var_subset_analysis = False   ### False 
#######
if var_subset_analysis: 
    data = subset_wo_missigness(data, var_subset_analysis)


''' 
4. clean variables; e.g. MGFA classification
'''
data_clean_parsed = clean_data(data)

''' 
5. prepare X and y
'''
X = data_clean_parsed.drop(target, axis=1)
y = data_clean_parsed[target]

pid = X.index.copy()
X = X.reset_index(drop=True) 

''' 
Read in variables
'''
sel_variables = pd.read_csv(varPath, header=None)[0].tolist()
num_columns = X.select_dtypes(include=["float64"]).columns
bin_columns = X.select_dtypes(include=["int64"]).columns
cat_columns = X.select_dtypes(include=["object"]).columns
preprocessor = imputation_scaling(num_columns, bin_columns, cat_columns, X)     
columnOrderAfterPreprocessing = [ele[5:] for ele in preprocessor.get_feature_names_out()]

##############################################################  Repeated DCV  ##############################################################

param_grid_rfc = {'n_estimators': [100, 300, 500],      ### changed
               'max_depth': [2,4],         ### changed
               'max_features': [2,4],  ### changed
               }


random_states = np.random.randint(10000, size=CV_repeats) 

######## Use a dict to track the SHAP values and predicted probas of label 1 each observation per CV repitition 
shap_values_per_cv = dict()
predProb_per_cv = dict()
pid_per_cv = dict()
for sample in X.index:
    ## Create keys for each sample
    shap_values_per_cv[sample] = {} 
    predProb_per_cv[sample] = {} 
    pid_per_cv[sample] = {}
    ## Then, keys for each CV fold within each sample
    for CV_repeat in range(CV_repeats):
        shap_values_per_cv[sample][CV_repeat] = {}
        predProb_per_cv[sample][CV_repeat] = {}
        pid_per_cv[sample][CV_repeat] = {}
        

for i, CV_repeat in enumerate(range(CV_repeats)): #-#-#
    #Verbose 
    print('\n------------ CV Repeat number:', CV_repeat)
    all_pid = []

    #Establish CV scheme
    outer_CV = StratifiedKFold(n_splits=n_splits_outer, random_state=random_states[i], shuffle=True)         ### n_split=5 should work well
    ix_training, ix_test = [], []
    # Loop through each fold and append the training & test indices to the empty lists above
    for fold in outer_CV.split(X, y):
        ix_training.append(fold[0]), ix_test.append(fold[1])

    for i, (train_outer_ix, test_outer_ix) in enumerate(zip(ix_training, ix_test)): #-#-#
        print('\n------ Fold Number:',i)
        X_train, X_test = X.iloc[train_outer_ix, :], X.iloc[test_outer_ix, :]
        y_train, y_test = y.iloc[train_outer_ix].values, y.iloc[test_outer_ix].values
        X_train = preprocessor.fit_transform(X_train)
        X_train_imputed = SupervisedSelector(preprocessor, sel_variables).transform(X_train)    
        X_test = preprocessor.fit_transform(X_test)
        X_test_imputed = SupervisedSelector(preprocessor, sel_variables).transform(X_test)   

        #### Inner CV
        inner_CV = StratifiedKFold(n_splits=n_splits_inner, random_state=11, shuffle=True)       ### n_split=3 should work well
        model = RandomForestClassifier()
        gs = GridSearchCV(model, param_grid_rfc, scoring="roc_auc", cv=inner_CV, refit=True).fit(X_train_imputed, y_train)
        clf = gs.best_estimator_.fit(X_train_imputed, y_train)

        y_predProba = clf.predict_proba(X_test_imputed)
        print(f"AUC:\t{round(roc_auc_score(y_test, y_predProba[:,1]), 4)}")
        
        
        # Use SHAP to explain predictions
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test_imputed)[1]
        # Extract SHAP information per fold per sample 
        for i, test_index in enumerate(test_outer_ix):
            shap_values_per_cv[test_index][CV_repeat] = shap_values[i] #-#-#
            predProb_per_cv[test_index][CV_repeat] = y_predProba[i,1]
            pid_per_cv[test_index][CV_repeat] = pid[i]

        ## Also save labels?
        pid_y_test = pid[test_outer_ix].tolist()
        all_pid += pid_y_test

with open(f"{resultsPath}/repeatedDCV_{CV_repeats}iter_shap.p", "wb") as f:
    pickle.dump(shap_values_per_cv, f)

with open(f"{resultsPath}/repeatedDCV_{CV_repeats}iter_predProb.p", "wb") as f:
    pickle.dump(predProb_per_cv, f)

pd.DataFrame(all_pid).to_csv(f"{resultsPath}/pid.csv", header=None)