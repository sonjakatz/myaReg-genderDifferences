
'''
Externa validation using bootstrapping (for CI calculation)

- load model trained on development cohort
- boostrapping test dataset: resample with replacement (n_samples = X.shape[0]) [inspiration](https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html#method-22-bootstrap-confidence-intervals-using-the-percentile-method)
- include SHAP analysis
- n_iter ~ 1000 

(c) Sonja Katz, 2024
'''


PATH = "/home/WUR/katz001/PROJECTS/myaReg-genderDifferences"

import os
import numpy as np
import json
import pandas as pd
import sys 
import errno  
sys.path.append(f"{PATH}/scripts")
import seaborn as sns 
import matplotlib.pyplot as plt
import shap

from func_imputeScale import pipe_imputation_scaling, pipe_supervisedSelector
from func_clf import externalValidate_boostrap_inclSHAP


import joblib

from sklearn.pipeline import Pipeline
import pickle



def get_input():
    try:
        n_bootstrap = sys.argv[1]
    except IndexError:
        print("give valid number of bootstrapping iterations")
        sys.exit()
    return int(n_bootstrap)



''' 
Prepare data --> change here for different setups!
'''
n_bootstrap = get_input()
target = "gender"


dataset = "natural"
percentBoruta = 100
varFolder = "boruta"
vars = f"{target}_bootstrapped_iterativeBoruta_{percentBoruta}perc"



''' 
Define paths
'''
folderFigures = f"{PATH}/figures/40_externalValidation/{dataset}/{vars}"
os.makedirs(folderFigures, exist_ok=True)
resultsPath = f"{PATH}/results/40_externalValidation/{dataset}"
os.makedirs(resultsPath, exist_ok=True)


dataPath = f"{PATH}/data/validation/"
modelPath = f"{PATH}/results/30_internalValidation/{dataset}/{vars}"

''' 
Variables
'''
with open(f"{PATH}/data/validation/discovery_validation_variables_translation.json", "r") as f: varTranslation = json.load(f)
dutch_varTranslation = {v: k for k, v in varTranslation.items()}

''' 
Read validation data
'''
with open(f"{PATH}/data/validation/validation_dtypes.json", "r") as f:
    dtypes = json.load(f)

data = pd.read_csv(f"{dataPath}/dutch_MG_patients_V2_recoded.csv", index_col=0, dtype=dtypes)
tmp = data.select_dtypes(include=["float32"]).columns 
data[tmp] = data[tmp].astype(pd.Int64Dtype())


''' Only parse variables needed for model '''
variables = pd.read_csv(f"{PATH}/results/20_featureSelection/{dataset}/{varFolder}/{vars}.txt", 
                       header=None)[0].tolist()
variables.append("gender")
variables_dutch = [varTranslation[ele] for ele in variables]
data = data.loc[:,variables_dutch]

''' translate varnames to German registry to fit models! '''
data.columns = [dutch_varTranslation[ele] for ele in data.columns]


''' 
Split
'''
X_val = data.drop(target, axis=1)
y_val = data[target].values
print(variables)
print(variables_dutch)

# # FOR DEVELOPMENT PURPOSES: smaller dataset
# X_val = X_val.iloc[:70,:]
# y_val = y_val[:70]


''' 
Prepare imputation and scaling
'''
num_columns = X_val.select_dtypes(include=["float64"]).columns
bin_columns = X_val.select_dtypes(include=["int64"]).columns
cat_columns = X_val.select_dtypes(include=["object"]).columns
preprocessor = pipe_imputation_scaling(num_columns, bin_columns, cat_columns)  

### Impute test data set on its own to avoid information leakage between internal and external validation!
X_val_imp = pd.DataFrame(preprocessor.fit_transform(X_val))
X_val_imp.index = X_val.index


'''
Load fitted model 
'''

pipe_fit = joblib.load(f"{modelPath}/model_fitted_wholeDataset.sav")
clf_fit = pipe_fit["classifier"]


''' 
Run Pipeline
'''
dic_summary = dict()
dic_summary_shap = dict()
dic_summary_predProba = dict()

for i in range(n_bootstrap):

    dic_bootstrap_results, dic_shap_values, dic_proba = externalValidate_boostrap_inclSHAP(X_val_imp, 
                                                                                   y_val, 
                                                                                   clf_fit,
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

    
with open(f'{resultsPath}/bootstrap_validation_n{n_bootstrap}_qc.pickle', 'wb') as f:
    pickle.dump(dic_summary, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'{resultsPath}/bootstrap_validation_n{n_bootstrap}_shap.pickle', 'wb') as f:
    pickle.dump(dic_summary_shap, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'{resultsPath}/bootstrap_validation_n{n_bootstrap}_predProba.pickle', 'wb') as f:
    pickle.dump(dic_summary_predProba, f, protocol=pickle.HIGHEST_PROTOCOL)




