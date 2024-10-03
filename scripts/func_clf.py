import pandas as pd
import numpy as np

from sklearn.model_selection import LeaveOneOut, StratifiedKFold, GridSearchCV
from sklearn.metrics import auc, roc_curve, average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
import shap

from func_imputeScale import sample_w_replacement


def classify_leave_one_out_cv(clf, 
                              X, 
                              y, 
                              model='',
                              saveIndivdualPred=True,
                              **kwargs):
    ''' 
    Run classifier with leave-one-out cross-validation 
    '''

    cv_outer = LeaveOneOut()
	
    df_results = pd.DataFrame() 
    all_y = np.array([])
    all_probs = np.array([])

    for i, (train, test) in enumerate(cv_outer.split(X, y)):
        X_train = X.iloc[train].copy()
        y_train = y.iloc[train].copy()
        X_val = X.iloc[test] # no need to copy, only a signle element
        y_test = y.iloc[test].iloc[0] # no need to copy, only a signle element

        # Hyperparameter tuning
        clf.fit(X_train, y_train) 
        y_predProba = clf.best_estimator_.predict_proba(X_val)[:,1]

        all_y = np.append(all_y, y_test)
        all_probs = np.append(all_probs, y_predProba)

	# Only calculate scores that use probabilities (not affected by a change in threshold!)
    final_results = {}
    fpr, tpr, _ = roc_curve(all_y,all_probs)
    auc_val = auc(fpr, tpr)
    precision_recall = average_precision_score(all_y,all_probs)
    results = {
                "auc": auc_val,
		        "average_prec":precision_recall,
                "model": model}  
    df_results = pd.concat([df_results, pd.Series(results)])  
    final_results['df_results'] = df_results 

    if saveIndivdualPred:
        df_indPred = pd.DataFrame(np.column_stack((all_y, all_probs)), columns=["y_true", "y_predProb"])
        final_results["df_indPred"] = df_indPred
              
    return final_results


def classify_dcv(X, y, pipe, hp_grid,
                 n_split_outer=5,
                 randomState_cvOuter=13):
    ''' 
    Run classifier with DCV
    '''
    ### Outer CV: set fixed seed
    cv_outer = StratifiedKFold(n_splits=n_split_outer, 
                               shuffle=True,
                               random_state=randomState_cvOuter)

    all_y = np.array([])
    all_probs = np.array([])
    for i, (train, test) in enumerate(cv_outer.split(X, y)):
        X_train = X.iloc[train].copy(); y_train = y.iloc[train].copy()
        X_val = X.iloc[test].copy(); y_test = y.iloc[test].copy()

        ### Inner CV: random seed
        gs = GridSearchCV(pipe, hp_grid, scoring='balanced_accuracy', verbose=1, cv=3, n_jobs=-1) 
        gs.fit(X_train, y_train)

        y_predProba = gs.best_estimator_.predict_proba(X_val)[:,1]
        all_y = np.append(all_y, y_test)
        all_probs = np.append(all_probs, y_predProba)
    
    fpr, tpr, _ = roc_curve(all_y,all_probs)
    auc_val = auc(fpr, tpr)
    precision_recall = average_precision_score(all_y,all_probs)
    results = {"auc": auc_val,
		       "average_prec":precision_recall}  
    return results




def classify_boostrap_inclSHAP(X, 
                                y, 
                                pipe, 
                                hp_grid,
                                perc_samples_per_boostrap=1):
    
    ''' 
    Run classifier with bootstrapping and SHAP analysis
    '''


    ''' Bootstrap '''
    train_idx, test_idx = sample_w_replacement(X, 
                                               n_size=np.ceil(X.shape[0]*perc_samples_per_boostrap).astype("int"),
                                               stratify=y.to_numpy(),
                                               random_state=None)       

    print(len(train_idx), len(test_idx))

    X_train = X.iloc[train_idx].copy(); y_train = y.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy(); y_test = y.iloc[test_idx].copy()

    ######### Bootstrapping sanity checks #########
    # _, a = np.unique(y_train,return_counts=True)
    # _, b = np.unique(y_test,return_counts=True)
    # print(f"Size X_train; labels y: {X_train.shape}, {a}")
    # print(f"Size X_test; labels y: {X_test.shape}, {b}")
    # print(train_idx, test_idx)
    ################################################

    ### Inner CV: random seed
    gs = GridSearchCV(pipe, hp_grid, scoring='balanced_accuracy', verbose=1, cv=3, n_jobs=-1) 
    gs.fit(X_train, y_train)

    ### Score
    y_predProba = gs.best_estimator_.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test,y_predProba)
    auc = roc_auc_score(y_test, y_predProba)
    precision_recall = average_precision_score(y_test,y_predProba)
    dic_results = {"auc": auc,
		           "average_prec":precision_recall, 
                   "fpr":fpr, 
                   "tpr":tpr}  
    
    ''' 
    SHAP 
    '''
    # Use SHAP to explain predictions
    shap_values_per_bootstrap = dict()
    explainer = shap.TreeExplainer(gs.best_estimator_["classifier"])
    X_test_imputed = gs.best_estimator_["imputation"].transform(X_test)
    shap_values = explainer.shap_values(X_test_imputed)[1] #explainer.shap_values(X_test_imputed)[:,:,1]
    # Extract SHAP information per fold per sample 
    for i, idx in enumerate(test_idx):
        shap_values_per_bootstrap[idx] = shap_values[i] #-#-#

    ''' 
    Save pred_proba()
    '''    
    proba_per_bootstrap = dict()
    for i, idx in enumerate(X.index[test_idx]):
        proba_per_bootstrap[idx] = y_predProba[i] #-#-#    



    return dic_results, shap_values_per_bootstrap, proba_per_bootstrap




def externalValidate_boostrap_inclSHAP(X, 
                                       y, 
                                       clf, 
                                       perc_samples_per_boostrap=1):
    
    ''' 
    Validate prior fitted classifier with bootstrapping of test set and SHAP analysis
    '''


    ''' Bootstrap '''
    val_idx, _ = sample_w_replacement(X, 
                                      n_size=np.ceil(X.shape[0]*perc_samples_per_boostrap).astype("int"),
                                      stratify=y.to_numpy(),
                                      random_state=None)       



    X_val = X.iloc[val_idx].copy(); y_val = y[val_idx].copy()
    ### X_val = X.iloc[test_idx].copy(); y_test = y[test_idx].copy()

    # ######## Bootstrapping sanity checks #########
    # _, a = np.unique(y_val,return_counts=True)
    # print(f"Size X_val; labels y: {X_val.shape}, {a}")
    # ###############################################

    print(X_val.shape)


    ### Score
    y_predProba = clf.predict_proba(X_val)[:,1]
    fpr, tpr, _ = roc_curve(y_val,y_predProba)
    auc_val = roc_auc_score(y_val, y_predProba)
    precision_recall = average_precision_score(y_val,y_predProba)
    dic_results = {"auc": auc_val,
		           "average_prec":precision_recall, 
                   "fpr":fpr, 
                   "tpr":tpr}  
    

    ''' 
    SHAP 
    '''
    # Use SHAP to explain predictions
    shap_values_per_bootstrap = dict()
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_val)[1]   #[:,:,1]
    # Extract SHAP information per fold per sample 
    for i, idx in enumerate(val_idx):
        shap_values_per_bootstrap[idx] = shap_values[i] #-#-#

    ''' 
    Save pred_proba()
    '''    
    proba_per_bootstrap = dict()
    for i, idx in enumerate(X.index[val_idx]):
        proba_per_bootstrap[idx] = y_predProba[i] #-#-#    


    return dic_results, shap_values_per_bootstrap, proba_per_bootstrap


