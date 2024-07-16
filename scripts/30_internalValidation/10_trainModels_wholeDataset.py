'''
Train model on whole dataset to use later for external validation

(c) Sonja Katz, 2024
'''


PATH = "/home/WUR/katz001/PROJECTS/permit-nsti-gas"

import os
import numpy as np
import json
import pandas as pd
import sys 
import errno  
sys.path.append(f"{PATH}/scripts")
import seaborn as sns 
import matplotlib.pyplot as plt

from func_preprocess import pipe_imputation_scaling, pipe_supervisedSelector
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
import joblib


def get_input():
    try:
        datasetTimepoint = sys.argv[1]
    except IndexError:
        print("ERROR\tPlease enter a valid dataset name (ENTRY, PRESURGERY,POSTSURGERY, BL)")
        sys.exit()
    return datasetTimepoint


''' 
Prepare data --> change here for different setups!
'''
datasetTimepoint = get_input()
#datasetTimepoint = "PRESURGERY"

target = "Conclusion_micro"
percentBoruta = 100

''' 
Select features
'''
vars = f"{target}_bootstrapped_iterativeBoruta_{percentBoruta}perc"   
varPath = f"{PATH}/results/20_featureSelection/{datasetTimepoint}/CV/{vars}.txt"

''' 
Define paths
'''
folderFigures = f"{PATH}/figures/{datasetTimepoint}/40_internalValidation"
os.makedirs(folderFigures, exist_ok=True)
resultsPath = f"{PATH}/results/40_internalValidation/{datasetTimepoint}"
os.makedirs(resultsPath, exist_ok=True)
dataPath = f"{PATH}/results/10_preprocessed/"
dataset = f"{datasetTimepoint}_{target}_preprocessed.csv"


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
with open(f"{PATH}/data/data_dtypes.json", "r") as f:
    dtypes = json.load(f)
data = pd.read_csv(f"{dataPath}/{dataset}", index_col=0, dtype=dtypes)
tmp = data.select_dtypes(include=["float32"]).columns 
data[tmp] = data[tmp].astype(pd.Int64Dtype())

''' 
Split
'''
X = data.drop(target, axis=1)
y = data[target]

# ## FOR DEVELOPMENT PURPOSES: smaller dataset
# X = X.iloc[:150,:]
# y = y[:150]

''' 
Read in variables
'''
sel_variables = pd.read_csv(varPath, header=None)[0].tolist()

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
gs = GridSearchCV(pipe, grids[model], scoring='balanced_accuracy', verbose=1, cv=3, n_jobs=-1) 
gs.fit(X, y)

print(gs.best_estimator_)

''' 
Save model
'''
filename = f'{resultsPath}/model_fitted_wholeDataset.sav'
joblib.dump(gs.best_estimator_, filename)


