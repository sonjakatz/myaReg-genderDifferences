import numpy as np
import pandas as pd
import json
import sys
import os
#PATH = "/home/sonja/PROJECTS/myaReg"

from func_imputeScale import imputation_scaling


def read_data(PATH, FILENAME):
    ''' Load data '''
    print("\n\nLOADING DATA")

    with open(f"{PATH}/data/data_dtypes.json", "r") as f:
        dtypes = json.load(f)
    data = pd.read_csv(f"{PATH}/data/{FILENAME}.csv", index_col=0, dtype=dtypes)
    ## Fix problem with integers accepting NAN 
    tmp = data.select_dtypes(include=["float32"]).columns 
    data[tmp] = data[tmp].astype(pd.Int64Dtype())
    return data

def subset_wo_missigness(data, selVar):
    print("\n\nSUBSETTING PATIENTS")
    data_masked = data.copy()
    for i in selVar: 
        mask = data_masked.loc[:, i].notna()
        data_masked = data_masked.loc[mask, :]
    print(data_masked.shape)
    return data_masked


def remove_NA(data, cutoff_perc = 35):
    ''' 
    Removal variables with too much missingness
    '''
    print(f"\n\nREMOVING MISSINGNESS (cutoff={cutoff_perc}%)")
    missingess_per_var = (data.isna().sum() / data.shape[0])*100
    varKeep = missingess_per_var[missingess_per_var <= cutoff_perc].index

    print(f"Keeping {len(varKeep)}/{data.shape[1]} variables")

    data_clean = data.loc[:,varKeep]

    print(f"Discarded: {missingess_per_var[missingess_per_var > cutoff_perc].index.tolist()}")
    return data_clean

def parseVariables(data, vars2remove):
    print(f"\n\nREMOVING BIASING / UNWANTED VARIABLES")
    #data_pruned = data.loc[:,vars2keep]
    vars2keep = [i for i in data.columns if i not in vars2remove]
    data_pruned = data.loc[:,vars2keep]
    print(f"Discarded: {[i for i in data.columns if i not in vars2keep]}")
    print(data_pruned.shape)
    return data_pruned


def clean_data(data):
    ''' 
    Re-index MGFA score
    '''
    print(f"\n\nCLEANING DATASET")
    mfga_reindex = {"0":"0",        
                    "1":"1",
                    "2":"2",
                    "3":"2",
                    "4":"3",
                    "5":"3",
                    "6":"4",
                    "7":"4",
                    "8":"5"
    }
    if "aktueller_mgfa_score" in data.columns:
        data.loc[:,["aktueller_mgfa_score"]] = data.loc[:,["aktueller_mgfa_score"]].replace(mfga_reindex)
    elif "mgfaklassifikation_schlimmste_historisch_rb" in data.columns:
        data.loc[:,["mgfaklassifikation_schlimmste_historisch_rb"]] = data.loc[:,["mgfaklassifikation_schlimmste_historisch_rb"]].replace(mfga_reindex)
    return data



def impute_scale(data, ohe_yn=False):
    ''' 
    Impute and scale
    '''
    print(f"\n\nIMPUTING & SCALING")

    ### Prepare Preprocessing ###
    ### get columns to apply transformation to ###
    num_columns = data.select_dtypes(include=["float64"]).columns
    bin_columns = data.select_dtypes(include=["int64"]).columns
    cat_columns= data.select_dtypes(include=["object"]).columns
    preprocessor = imputation_scaling(num_columns, bin_columns, cat_columns, data, ohe=ohe_yn)
    columnOrderAfterPreprocessing = [ele[5:] for ele in preprocessor.get_feature_names_out()]
    data_imputedScaled = preprocessor.fit_transform(data)
    df_imputedScaled = pd.DataFrame(data_imputedScaled, columns=columnOrderAfterPreprocessing)

    '''
    Optional: reverse transform for plotting: 
    '''

    num_columns_inverse = pd.Series(columnOrderAfterPreprocessing)[pd.Series(preprocessor.get_feature_names_out()).str.contains("num")].tolist()
    num_columns_inverse_vals = preprocessor.named_transformers_["num"]["scaler"].inverse_transform(df_imputedScaled[num_columns_inverse])
    num_columns_inverse = pd.DataFrame(num_columns_inverse_vals, columns=num_columns).astype("float")
    num_columns_inverse

    bin_columns_inverse = pd.Series(columnOrderAfterPreprocessing)[pd.Series(preprocessor.get_feature_names_out()).str.contains("bin")].tolist()
    bin_columns_inverse = round(df_imputedScaled[bin_columns_inverse].copy(),0)         ### Round to get binary values again
    bin_columns_inverse = bin_columns_inverse.astype("int64")

    try: 
        cat_columns_inverse = pd.Series(columnOrderAfterPreprocessing)[pd.Series(preprocessor.get_feature_names_out()).str.contains("cat")].tolist()
        if ohe_yn:
            cat_columns_inverse_vals = preprocessor.named_transformers_["cat"]["encoding"].inverse_transform(df_imputedScaled[cat_columns_inverse])
        else:
            cat_columns_inverse_vals = preprocessor.named_transformers_["cat"]["scaler"].inverse_transform(df_imputedScaled[cat_columns_inverse])
            cat_columns_inverse_vals = preprocessor.named_transformers_["cat"]["encoding"].inverse_transform(cat_columns_inverse_vals)
        cat_columns_inverse = pd.DataFrame(cat_columns_inverse_vals, columns=cat_columns).astype("int64")
    except:
        cat_columns_inverse = pd.DataFrame()
    
    df_inverseTransform = pd.concat([num_columns_inverse, bin_columns_inverse, cat_columns_inverse], axis=1) ##

    print("\n")

    return df_imputedScaled, df_inverseTransform



