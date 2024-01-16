''' 
Start remotely to run in the background like this:

adapted code from Ziga Pušnik (University of Ljubljana)
'''


import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt 
import random
import sys, errno  
sys.path.append("../")


from func_preprocess import read_data, subset_wo_missigness, remove_NA, parseVariables, clean_data, impute_scale 
from func_prediction import classify_CV

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')


PATH_base = "/home/WUR/katz001/PROJECTS/myaReg-genderDifferences"

''' 
Prepare data --> change here for different setups!
'''
target = "gender"
percentBoruta = 100

''' 
Select features
'''
vars = f"{target}_bootstrapped_iterativeBoruta_{percentBoruta}perc"
varPath = f"{PATH_base}/results/20_featureSelection/CV_v3_mgfaRecoded/{vars}.txt"

''' 
Define paths
'''
resultsPath = f"{PATH_base}/results/30_predictions/CV_v3_mgfaRecoded/{vars}/modelComparison"
os.makedirs(resultsPath, exist_ok=True)


models = {
          'rfc': RandomForestClassifier()
         }               


grids = {'rfc':{
                'n_estimators': [100, 300, 1000],      ### changed
               'max_depth': [2,4,6],         ### changed
               'max_features': [2,4],  ### changed
               }}   

inner_cv = 3
outer_cv = 10

''' 
Read data
'''
data = read_data(PATH_base,FILENAME="all_data_edited_v3_mgfaRecoded_inverse")
X = data.drop(target, axis=1)
y = data[target]

# ##### FOR DEVELOPMENT PURPOSES: smaller dataset
# X = X.iloc[:100,:]
# y = y[:100]


''' 
Read in variables
'''
sel_variables = pd.read_csv(varPath, header=None)[0].tolist()
print(sel_variables)

''' 
Run Pipeline
'''

try: 
      for model in models.keys():
            df_before = pd.DataFrame()    

            saveIndivdualPred = True

            clf = GridSearchCV(models[model], grids[model], scoring='balanced_accuracy', verbose=1, cv=inner_cv, n_jobs=-1) ## cv=3  ##changed this for svc to cv=2; otherwise takes too long!
            result = classify_CV(clf, 
                                 X, 
                                 y,
                                 sel_variables,
                                 outer_cv = outer_cv, 
                                 model=model, 
                                 save_to = resultsPath + f"/{model}", 
                                 saveIndivdualPred = saveIndivdualPred)


            #print(result)
            ''' Prepare results '''
            result['model'] = model

            ## df_before = pd.concat([df_before, result["df_results"]], ignore_index=True)       

            #df_importances = pd.concat([df_importances, result['importances_df']],  ignore_index=True)
            if saveIndivdualPred:
                  df_indPred = pd.DataFrame()
                  df_indPred = pd.concat([df_indPred, result["df_indPred"]], ignore_index=True)

            ''' Save to file '''
            ## df_before.to_csv((resultsPath+f"/prediction_cv_test_{model}.csv"), index=False)          
            df_indPred.to_csv((resultsPath+f"/individualPredictions_test_{model}.csv"), index=False)   

except IOError as e: 
      if e.errno == errno.EPIPE:
          print(e)
          pass