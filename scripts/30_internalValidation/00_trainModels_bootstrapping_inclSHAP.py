
'''
Internal validation using bootstrapping (for CI calculation)

- boostrapping: resample with replacement (n_samples = X.shape[0])
- include hyperparameter tuning 
- n_iter ~ 200 [acc to this](https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html#method-22-bootstrap-confidence-intervals-using-the-percentile-method)
- include SHAP analysis

(c) Sonja Katz, 2024
'''


PATH_base = "/home/WUR/katz001/PROJECTS/myaReg-genderDifferences"

import os
import numpy as np
import json
import pandas as pd
import sys 
import errno  
sys.path.append(f"{PATH_base}/scripts")
import seaborn as sns 
import shap

from func_preprocess import read_data
from func_imputeScale import pipe_imputation_scaling, pipe_supervisedSelector
from func_clf import classify_boostrap_inclSHAP


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
import pickle


def get_input():
    try:
        n_bootstrap = sys.argv[1]
        dataset = sys.argv[2]
        percentBoruta = sys.argv[3]
    except IndexError:
        print("ERROR\tPlease enter a valid dataset name (ENTRY, PRESURGERY,POSTSURGERY, BL)")
        sys.exit()
    return int(n_bootstrap), dataset, int(percentBoruta)


''' 
Prepare input
'''
n_bootstrap, dataset, percentBoruta = get_input()
target = "gender"
saveFig_quickCheck = False

################### Variable selection ###################

''' 
IF: automated feature selection
'''
varFolder = "boruta"
vars = f"{target}_bootstrapped_iterativeBoruta_{percentBoruta}perc"

# ''' 
# ELSE: Manual variable list
# '''
# varFolder = "manual"
# vars = "partiallyValidated" #"allVariables"

#########################################################


''' 
Select features
'''
varPath = f"{PATH_base}/results/20_featureSelection/{dataset}/{varFolder}/{vars}.txt"

''' 
Define paths
'''
folderFigures = f"{PATH_base}/figures/30_internalValidation/{dataset}/{vars}"
os.makedirs(folderFigures, exist_ok=True)
resultsPath = f"{PATH_base}/results/30_internalValidation/{dataset}/{vars}"
os.makedirs(resultsPath, exist_ok=True)


models = {
          'rfc': RandomForestClassifier(), 
         }               


grids = {'rfc':{
               'classifier__n_estimators': [100, 300, 700],     
               'classifier__max_depth': [2,4,6],         
               'classifier__max_features': [2,4,6],  
               }
         }   

''' 
Read data
'''
data = read_data(PATH_base,FILENAME=f"{dataset}")
X_orig = data.drop(target, axis=1)
y_orig = data[target]

''' 
Split
'''
X = data.drop(target, axis=1)
y = data[target]

## FOR DEVELOPMENT PURPOSES: smaller dataset
# X = X.iloc[:100,:]
# y = y[:100]
print(X.shape)
print(y.value_counts())


''' 
Read in variables
'''
sel_variables = pd.read_csv(varPath, header=None)[0].tolist()
print(len(sel_variables), sel_variables)

''' 
Prepare imputation and scaling
'''
num_columns = X.loc[:,sel_variables].select_dtypes(include=["float64"]).columns
bin_columns = X.loc[:,sel_variables].select_dtypes(include=["int64"]).columns
cat_columns = X.loc[:,sel_variables].select_dtypes(include=["object"]).columns
preprocessor = pipe_imputation_scaling(num_columns, bin_columns, cat_columns)  

''' 
Run Pipeline
'''
model = 'rfc'
dic_summary = dict()
dic_summary_shap = dict()
dic_summary_predProba = dict()

for i in range(n_bootstrap):

    ''' 
    Assemble pipeline
    '''
    pipe = Pipeline([("selector", pipe_supervisedSelector(sel_variables)),
                        ("imputation", preprocessor),
                        ("classifier", models[model])])


    dic_bootstrap_results, dic_shap_values, dic_proba = classify_boostrap_inclSHAP(X, 
                                                                        y, 
                                                                        pipe, 
                                                                        hp_grid=grids[model],
                                                                        perc_samples_per_boostrap=1)
    ### Save AUC and ROC for quick check
    dic_summary[i] = dic_bootstrap_results

    ### Save SHAP values
    for k,v in dic_shap_values.items():
        if k in dic_summary_shap.keys():
            dic_summary_shap[k].append(v)
        else:
            dic_summary_shap[k] = [v]

    ### Save individual predictions for further analyses
    dic_summary_predProba[i] = dic_proba

    
with open(f'{resultsPath}/bootstrap_{model}_n{n_bootstrap}_qc.pickle', 'wb') as f:
    pickle.dump(dic_summary, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'{resultsPath}/bootstrap_{model}_n{n_bootstrap}_shap.pickle', 'wb') as f:
    pickle.dump(dic_summary_shap, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'{resultsPath}/bootstrap_{model}_n{n_bootstrap}_predProba.pickle', 'wb') as f:
    pickle.dump(dic_summary_predProba, f, protocol=pickle.HIGHEST_PROTOCOL)


