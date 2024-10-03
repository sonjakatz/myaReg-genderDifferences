
'''

Iterative bootstrapped BorutaPy 

(c) Sonja Katz, 2024

'''

## normal
from boruta import BorutaPy
import os
import sys
import numpy as np
import json
import sys
sys.path.append("../")

from func_preprocess import read_data, subset_wo_missigness, remove_NA, parseVariables, clean_data
from func_imputeScale import imputation_scaling

from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, cross_validate
import pandas as pd


def run_iterativeBoruta(X, y, cols, perc=100, n_iter=100, max_iter=100):

    dict_boruta = {}

    for i in range(n_iter):
        print(f"Round {i+1} of {n_iter}")

        ''' 
        Setup and run Boruta
        '''
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
        feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=None, perc=perc, max_iter=max_iter)
        feat_selector.fit(X, y)

        ''' 
        Get selected variables and save in dict
        '''
        selVars = np.array(cols)[feat_selector.support_]
        for var in selVars: 
            if var in dict_boruta.keys():
                dict_boruta[var] += 1
            else: 
                dict_boruta[var] = 1

    ### Normalise regarding number of iterations
    dict_boruta.update((x, y/n_iter) for x, y in dict_boruta.items())
    
    return dict_boruta


PATH = "/home/WUR/katz001/PROJECTS/myaReg-genderDifferences"

###################     EDIT HERE    ###################
target = "gender"
dataset = "natural"    #"histologie_subgroup"
PATH_out = f"{PATH}/results/20_featureSelection/{dataset}/boruta"

''' 1. read data '''
data = read_data(PATH, FILENAME=f"{dataset}")  ### {dataset}_variables

#########################################################            

os.makedirs(PATH_out, exist_ok=True)


''' 
2. OPTIONAL: make subanalysis for variables with high missigness 
specify which variable should be kept in the dataset; remove rest of vars with too much missigness

CV_v1: False
CV_v2_histo: ['histologie_sprb']

'''
#######
var_subset_analysis = False   ### False     
#######
if var_subset_analysis: 
    data = subset_wo_missigness(data, var_subset_analysis)
data_clean = remove_NA(data, cutoff_perc=35)

''' 
3. OPTIONAL. remove the ones we dont like (correlated or else)

'''
vars2remove = pd.read_csv(f"{PATH}/data/variables_to_remove_fullRegistry.txt", header=None)[0].tolist()     
data_clean_parsed = parseVariables(data_clean, vars2remove)
print(data_clean_parsed.columns)

''' 
4. clean variables; e.g. MGFA classification
'''
data_clean_parsed = clean_data(data_clean_parsed)

''' 
5. prepare X and y
'''
X = data_clean_parsed.drop(target, axis=1)
y = data_clean_parsed[target]

print(X.shape)
print(X)
print(y.value_counts())

### Prepare Preprocessing ###
### get columns to apply transformation to ###
num_columns = X.select_dtypes(include=["float64"]).columns
bin_columns = X.select_dtypes(include=["int64"]).columns
cat_columns = X.select_dtypes(include=["object"]).columns
preprocessor = imputation_scaling(num_columns, bin_columns, cat_columns, X)
columnOrderAfterPreprocessing = [ele[5:] for ele in preprocessor.get_feature_names_out()]


for perc in [100]:    ### 100,80

    for i in range(1,50):  ## 1,30

        outname_json=f"{i}__{target}_iterativeBoruta_{perc}perc.json"


        ''' Bootstrap '''
        X_train, _, y_train, _ = train_test_split(X, y, stratify=y, train_size=0.8, random_state=None)

        ''' Impute '''
        X_ = preprocessor.fit_transform(X_train)
        y_ = y_train.values.ravel().astype('int')

        print(X_.shape)
        
        ''' Iterative Boruta'''
        n_iter = 50
        dict_iterBoruta = run_iterativeBoruta(X=X_,
                                            y=y_, 
                                            cols=columnOrderAfterPreprocessing, 
                                            perc=perc,
                                            n_iter=n_iter,
                                            max_iter=100)

        with open(f"{PATH_out}/{outname_json}", "w") as f: json.dump(dict_iterBoruta, f, indent=4)
