import pandas as pd
import numpy as np
from math import floor
from scipy import stats as st
from sklearn.pipeline import NotFittedError

class Pool:
    def __init__(self, models):
        self.predictors = models
        #self.keys = list(models.keys())
        #self.predict_cache = None
        #self.predict_proba_cache = None
        self.tree_based = ['Decision_Tree','Random_Forest','Gradient_Boosting','AdaBoost','XGBoost']
    
    def __len__(self):
        return len(self.predictors)

    def fit(self, X, X_norm, y):
        for name, model in self.predictors.items():
            try:
                _ = model.predict(X)
            except NotFittedError:
                if(name in self.tree_based):
                    model.fit(X,y)
                else:
                    model.fit(X_norm,y)
            print(f'{name} fitted.')
        
        return self

    def predict(self, X, X_norm):
        pred = pd.DataFrame(columns=list(self.predictors.keys()))
        for name, model in self.predictors.items():
            if(name in self.tree_based):
                pred[name] = model.predict(X)
            else:
                pred[name] = model.predict(X_norm)
        return pred

    def predict_proba(self, X, X_norm):
        pred = {}
        for name, model in self.predictors.items():
            if(name in self.tree_based):
                pred[name] = model.predict_proba(X)
            else:
                pred[name] = model.predict_proba(X_norm)
        return pred
    
    def drop(self, *args):
        new_models = {}
        for name, model in self.predictors.items():
            if(name not in args):
                new_models[name] = model
        newPool = Pool(new_models)
        return newPool

class Rejector:
    def __init__(self, model_pool:Pool):
        self.predictors = model_pool
        self.cache_input = None
        self.cache_output_preds = None
        self.cache_output_proba = None

    def predictWithReject(self, X, X_norm, reject_rate, reject_method:str='avg', warnings:bool=True):
        # reject_method deve ser igual a 'avg', 'median', 'min' ou 'max'
        if(reject_method not in ['avg', 'mean', 'median', 'min','max']):
            raise ValueError("reject_method deve ser igual a 'avg', 'mean, 'median', 'min' ou 'max'")
        
        if(isinstance(X, pd.DataFrame)):
            X = X.to_numpy()
        if((self.cache_input is not None) and (X.shape == self.cache_input.shape) and ((X == self.cache_input).all() and (self.cache_output_proba.shape[1]==len(self.predictors)))):
            predictions = self.cache_output_preds
            poolProb = self.cache_output_proba
        else:
            poolProb = pd.DataFrame()
            poolPreds = pd.DataFrame()
            results = self.predictors.predict_proba(X, X_norm)

            for name, probas in results.items():
                poolPreds[name] = np.argmax(probas, axis=1)
                poolProb[name] = 1 - np.max(probas, axis=1)

            predictions = pd.DataFrame(np.array(st.mode(poolPreds.values, axis=1))[0])

            self.cache_input = X
            self.cache_output_preds = predictions
            self.cache_output_proba = poolProb

        if(reject_method=='max'):
            predictions['score'] = poolProb.apply(lambda x: max(x), axis=1)
        elif(reject_method=='median'):
            predictions['score'] = poolProb.apply(lambda x: np.median(x), axis=1)
        elif(reject_method=='min'):
            predictions['score'] = poolProb.apply(lambda x: min(x), axis=1)
        else: #avg or mean
            predictions['score'] = poolProb.apply(lambda x: np.mean(x), axis=1)

        if(reject_rate > 1):
            reject_rate = reject_rate/100

        n_reject = floor(poolProb.shape[0] * reject_rate)
        if((n_reject==0) and (warnings)):
            print('Warning: Number of rejections equals 0. Increase the rejection rate.')
        elif((n_reject==poolProb.shape[0]) and (warnings)):
            print('Warning: All examples will be rejected. Decrease the rejection rate.')

        predictions = predictions.sort_values(by='score', ascending=False)
        predictions.iloc[:n_reject, :] = np.nan # REJECTED
        predictions = predictions.drop(columns=['score']).sort_index()

        return predictions[0]