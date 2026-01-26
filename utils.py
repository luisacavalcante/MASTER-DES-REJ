import pandas as pd
import numpy as np
from tqdm import tqdm
from train_models import CONFIG, load_model_by_name
from classes import Pool, METADESR
from sklearn.metrics import (accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score)

def load_all_models(experiments:str|list, model_names:list=CONFIG['BASE_MODELS']) -> dict:
    models = {}

    if(isinstance(experiments, str)):
        experiments = [experiments]*len(model_names)
    else:
        if(len(experiments) != len(model_names)):
            raise ValueError('Variável "experiments" deve ser uma lista com tamanho igual à "models", ou ser apenas uma string')

    for i in range(len(model_names)):
        trained_model, _ = load_model_by_name(experiments[i], model_names[i])
        models[model_names[i]] = trained_model

    return models

def rejection_overhall(rejector, x, y, rej_rates, methods=['avg','median','min','max']):
    results_log = pd.DataFrame(columns=['Method','Rejection Rate','Accuracy','Precision','Recall','F1-Score'])
    rejection_history = pd.DataFrame(columns=['idx','Method','Rejection Rate'])

    rej_rates = np.asarray(rej_rates)
    if(any(rej_rates<0) or any(rej_rates>1)):
        raise ValueError("Apenas valores entre 0 e 1")
    if(rej_rates.max() != 1):
        rej_rates = np.append(rej_rates, [1])
    if(isinstance(y, pd.DataFrame) or isinstance(y, pd.Series)):
        y = y.values
    total_iterations = len(methods) * len(rej_rates)

    with tqdm(total=total_iterations, desc="Processing") as pbar:
        for method in methods:
            for reject_rate in rej_rates:
                y_pred = rejector.reject_rate_predict(x, reject_rate, method, warnings=False)
                idx = y_pred.dropna().index
                
                idx_rej = y_pred[y_pred.isna()].index
                rejected = rejection_history.loc[rejection_history['Method']==method,'idx']
                idx_rej = list(set(idx_rej) - set(rejected))
                for i in range(len(idx_rej)):
                    rejection_history.loc[rejection_history.shape[0],:] = [idx_rej[i], method, reject_rate]
                
                if(reject_rate!=1):
                    acc = accuracy_score(y[idx], y_pred[idx])
                    if(len(set(y))>2):
                        pre = precision_score(y[idx], y_pred[idx], zero_division=0, average='macro')
                        rec = recall_score(y[idx], y_pred[idx], zero_division=0, average='macro')
                        f1s = f1_score(y[idx], y_pred[idx], zero_division=0, average='macro')
                    else:
                        pre = precision_score(y[idx], y_pred[idx], zero_division=0)
                        rec = recall_score(y[idx], y_pred[idx], zero_division=0)
                        f1s = f1_score(y[idx], y_pred[idx], zero_division=0)
                    results_log.loc[results_log.shape[0]] = [method,reject_rate,acc,pre,rec,f1s]
                    
                pbar.set_postfix({'Method': method, 'Rate': f'{reject_rate:.2f}'})
                pbar.update(1)

    return results_log, rejection_history

from itertools import product 
def rejection_threshold_overhall(rejector, x, y, thresholds:list[float]=[0.5], methods=['avg','median','min','max']):
    # Vale para Pool com rejeição e METADES-R
    if(isinstance(rejector, METADESR)):
        threshold_type = 'Selection Threshold'
        def predict_func(method, threshold):
            rejector.set_predict_params(rejection_method=method, selection_threshold=threshold)
            return rejector.predict(x)
    elif(isinstance(rejector, Pool)):
        threshold_type = 'Rejection Threshold'
        def predict_func(method, threshold):
            return rejector.reject_threshold_predict(x,
                                                     reject_threshold=threshold,
                                                     reject_method=method,
                                                     warnings=False)

    results_log = pd.DataFrame(columns=['Method',threshold_type,'Rejection Rate','Accuracy','Precision','Recall','F1-Score'])
    thresholds = np.asarray(thresholds)
    if(any(thresholds<0) or any(thresholds>1)):
        raise ValueError("Apenas valores entre 0 e 1")
    if(thresholds.min() != 0):
        thresholds = np.append(thresholds, [0])
    if(isinstance(y, pd.DataFrame) or isinstance(y, pd.Series)):
        y = y.values
    
    thresholds.sort()
    total_iterations = len(methods) * len(thresholds)
    with tqdm(total=total_iterations, desc="Processing") as pbar:
        for method, threshold in product(methods, thresholds):
            
            y_pred = predict_func(method, threshold)
            idx = ~y_pred.mask
                        
            if(any(idx)):
                acc = accuracy_score(y[idx], y_pred[idx])
                if(len(set(y))>2):
                    pre = precision_score(y[idx], y_pred[idx], zero_division=0, average='macro')
                    rec = recall_score(y[idx], y_pred[idx], zero_division=0, average='macro')
                    f1s = f1_score(y[idx], y_pred[idx], zero_division=0, average='macro')
                else:
                    pre = precision_score(y[idx], y_pred[idx], zero_division=0)
                    rec = recall_score(y[idx], y_pred[idx], zero_division=0)
                    f1s = f1_score(y[idx], y_pred[idx], zero_division=0)
                rej_rate = 1-((idx).sum()/len(y))
                results_log.loc[results_log.shape[0]] = [method,threshold,rej_rate,acc,pre,rec,f1s]
                
            pbar.set_postfix({'Method': f'{method}', threshold_type: f'{threshold:.2f}'})
            pbar.update(1)

    return results_log

def rejection_overhall_metadesr(model, x, y, reject_rates, selection_thresholds:list[float]=[0.5], methods=['avg','median','min','max']):
    results_log = pd.DataFrame(columns=['Rejection Rate','BM Selection Threshold','Method','Accuracy','Precision','Recall','F1-Score'])
    rejection_history = pd.DataFrame(columns=['idx','Rejection Rate','BM Selection Threshold','Method'])
    
    reject_rates = np.asarray(reject_rates)
    if(any(reject_rates<0) or any(reject_rates>1)):
        raise ValueError("Apenas valores entre 0 e 1")
    if(isinstance(y, pd.DataFrame) or isinstance(y, pd.Series)):
        y = y.values
    total_iterations = len(selection_thresholds) * len(reject_rates) * len(methods)

    with tqdm(total=total_iterations, desc="Processing") as pbar:
        for method, rej_rate, sel_thresh in product(methods, reject_rates, selection_thresholds):
            model.set_thresholds(selection_threshold=sel_thresh)
            model.rejection_method = method
            model.reject_rate = rej_rate
            y_pred = model.predict(x)

            idx_rej = np.isnan(y_pred)#.index
            idx = ~idx_rej
            rejected = rejection_history.loc[(rejection_history['Method']==method)&(rejection_history['BM Selection Threshold']==sel_thresh),'idx']
            idx_rej = list(set(idx_rej) - set(rejected))
            for i in range(len(idx_rej)):
                rejection_history.loc[rejection_history.shape[0],:] = [idx_rej[i], rej_rate, sel_thresh, method]
            
            if(len(idx)>0):
                acc = accuracy_score(y[idx], y_pred[idx])
                if(len(set(y))>2):
                    pre = precision_score(y[idx], y_pred[idx], zero_division=0, average='macro')
                    rec = recall_score(y[idx], y_pred[idx], zero_division=0, average='macro')
                    f1s = f1_score(y[idx], y_pred[idx], zero_division=0, average='macro')
                else:
                    pre = precision_score(y[idx], y_pred[idx], zero_division=0)
                    rec = recall_score(y[idx], y_pred[idx], zero_division=0)
                    f1s = f1_score(y[idx], y_pred[idx], zero_division=0)
                results_log.loc[results_log.shape[0]] = [rej_rate,sel_thresh,method,acc,pre,rec,f1s]
                
            pbar.set_postfix({'Rejection by Uncertainty': f'{100*rej_rate:.2f}%', 'Selection by Competence': f'{sel_thresh:.2f}', 'Method': method})
            pbar.update(1)

    return results_log, rejection_history