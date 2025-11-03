import pandas as pd
import numpy as np
from tqdm import tqdm
from train_models import CONFIG, load_model_by_name
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

def rejection_overhall(rejector, x, x_norm, y, rej_rates, methods=['avg','median','min','max']):
    results_log = pd.DataFrame(columns=['Method','Rejection Rate','Accuracy','Precision','Recall','F1-Score'])
    rejection_history = pd.DataFrame(columns=['idx','Method','Rejection Rate'])
    #rejected = {}
    rej_rates = np.asarray(rej_rates)
    if(any(rej_rates<0) or any(rej_rates>1)):
        raise ValueError("Apenas valores entre 0 e 1")
    if(rej_rates.max() != 1):
        rej_rates = np.append(rej_rates, [1])
    
    total_iterations = len(methods) * len(rej_rates)

    with tqdm(total=total_iterations, desc="Processing") as pbar:
        for method in methods:
            for reject_rate in rej_rates:
                y_pred = rejector.predictWithReject(x, x_norm, reject_rate, method, warnings=False)
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