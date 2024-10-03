'''
Train model on whole dataset to use later for external validation

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

from func_preprocess import read_data, subset_wo_missigness, remove_NA, parseVariables, clean_data, impute_scale 
from func_imputeScale import pipe_imputation_scaling, pipe_supervisedSelector
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
import joblib


def get_input():
    try:
        dataset = sys.argv[1]
        percentBoruta = sys.argv[2]
    except IndexError:
        print("ERROR\tPlease enter a valid dataset name (ENTRY, PRESURGERY,POSTSURGERY, BL)")
        sys.exit()
    return dataset, int(percentBoruta)


''' 
Prepare data --> change here for different setups!
'''
dataset, percentBoruta = get_input()
target = "gender"

''' 
Select features
'''
# varFolder = "manual"
# vars = "partiallyValidated" #"allVariables"

varFolder = "boruta"
vars = f"{target}_bootstrapped_iterativeBoruta_{percentBoruta}perc"

varPath = f"{PATH}/results/20_featureSelection/{dataset}/{varFolder}/{vars}.txt"


''' 
Define paths
'''
folderFigures = f"{PATH}/figures/30_internalValidation/{dataset}/{vars}"
os.makedirs(folderFigures, exist_ok=True)
resultsPath = f"{PATH}/results/30_internalValidation/{dataset}/{vars}"
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
data = read_data(PATH,FILENAME=f"{dataset}")
X = data.drop(target, axis=1)
y = data[target]

# ## FOR DEVELOPMENT PURPOSES: smaller dataset
# X = X.iloc[:150,:]
# y = y[:150]

print(X.shape)
print(y.value_counts())

''' 
Read in variables
'''
sel_variables = pd.read_csv(varPath, header=None)[0].tolist()
print(sel_variables)

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

''' 
Assemble pipeline
'''
pipe = Pipeline([("selector", pipe_supervisedSelector(sel_variables)),
                        ("imputation", preprocessor),
                        ("classifier", models[model])])

### Inner CV: random seed
gs = GridSearchCV(pipe, grids[model], scoring='balanced_accuracy', verbose=1, cv=5, n_jobs=-1) 
gs.fit(X, y)

print(gs.best_estimator_)

''' 
Save model
'''
filename = f'{resultsPath}/model_fitted_wholeDataset.sav'
joblib.dump(gs.best_estimator_, filename)


