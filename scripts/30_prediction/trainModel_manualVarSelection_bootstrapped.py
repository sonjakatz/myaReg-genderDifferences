''' 

Random Forest prediction of gender using variable selected by BorutaPy

- double cross-validation (DCV) (inner_cv=3, outer_cv=10)
- includes iterative bootstrapping (80%) to be able to subsequently calculate the confidence interval of predictions



Note: to get probabilties for individual patients run script "trainModel_featureSelection.py"

(c) Sonja Katz


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
from func_prediction import imputation_scaling, SupervisedSelector, classify_CV

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


import warnings
warnings.filterwarnings('ignore')


PATH_base = "/home/WUR/katz001/PROJECTS/myaReg-genderDifferences"

''' 
Prepare data --> change here for different setups!
'''
target = "gender"

''' 
Select features
'''
vars = f"allVars_mgfaRecoded"    
varPath = f"{PATH_base}/results/20_featureSelection/manual_selection/{vars}.txt"

''' 
Define paths
'''
resultsPath = f"{PATH_base}/results/30_predictions/manual_selection/{vars}/modelComparison"
os.makedirs(resultsPath, exist_ok=True)


models = {
          'rfc': RandomForestClassifier()
         }               


grids = {'rfc':{
                'n_estimators': [100, 300, 500],      ### changed  #1000
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

#### bootstrapped 
num_bootstrap = 5

for i in range(num_bootstrap):
      bootstrapped_pid = random.choices(X.index.tolist(), k=round(X.shape[0]*.8))
      X = X.loc[bootstrapped_pid, :]
      y = y[bootstrapped_pid]

      ''' 
      Run Pipeline
      '''

      try: 
            for model in models.keys():

                  ## Read in existing results file and add there
                  if os.path.isfile(f"{resultsPath}/prediction_cv_test_{model}.csv"):
                        df_before = pd.read_csv(f"{resultsPath}/prediction_cv_test_{model}.csv")
                  else: 
                     df_before = pd.DataFrame()    

                  saveIndivdualPred = False

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

                  df_before.index = result["df_results"].index
                  df_before = pd.concat([df_before, result["df_results"]], axis=1)       

                  # if saveIndivdualPred:
                  #       df_indPred = pd.DataFrame()
                  #       df_indPred = pd.concat([df_indPred, result["df_indPred"]], ignore_index=True)

                  ''' Save to file '''
                  df_before.to_csv((resultsPath+f"/prediction_cv_test_{model}.csv"), index=False)          
                  # df_indPred.to_csv((resultsPath+f"/individualPredictions_test_{model}.csv"), index=False)   

      except IOError as e: 
            if e.errno == errno.EPIPE:
                  print(e)
            pass