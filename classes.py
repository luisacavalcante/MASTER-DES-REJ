import pandas as pd
import numpy as np
from math import floor
from scipy import stats as st
from sklearn.pipeline import NotFittedError
from sklearn.preprocessing import StandardScaler

class SpecificScaler(StandardScaler):
    def __init__(self, *, copy = True, with_mean = True, with_std = True):
        super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)
        self.features = None

    def _prep_data(self, X, features:list=None):
        if(not isinstance(X, pd.DataFrame)):
            X = pd.DataFrame(X)
        if(features is not None):
            self.features = features
        elif(self.features is None):
            self.features = X.columns
        return X

    def fit(self, X:pd.DataFrame, features:list=None):
        '''Fit the scaler on specific columns of data'''
        X = self._prep_data(X.copy(), features)
        super().fit(X.loc[:,self.features])
        return self

    def fit_transform(self, X:pd.DataFrame, features:list=None):
        '''Fit the scaler on specific columns of data and transform it'''
        X = self._prep_data(X.copy(), features)
        X.loc[:,self.features] = super().fit_transform(X.loc[:,self.features])

        return X

    def transform(self, X:pd.DataFrame):
        '''Use the fitted scaler to transform specific columns of data'''
        X = X.copy()
        if(not isinstance(X, pd.DataFrame)):
            X = pd.DataFrame(X)
        X.loc[:,self.features] = super().transform(X.loc[:,self.features])
        return X

class Pool:
    def __init__(self, models):
        self.predictors = models
        self.cache_input_pred = None
        self.cache_input_proba = None
        self.cache_output_pred = None
        self.cache_output_proba = None
    
    def __len__(self):
        return len(self.predictors)
    
    def __getitem__(self, key):
        return self.predictors[key]

    def fit(self, X, y):
        for name, model in self.predictors.items():
            try:
                _ = model.predict(X)
            except NotFittedError:
                model.fit(X,y)
            print(f'{name} fitted.')
        
        return self

    def predict(self, X) -> pd.DataFrame:
        pred = pd.DataFrame(columns=list(self.predictors.keys()))
        
        if((self.cache_input_pred is not None) and (X.shape == self.cache_input_pred.shape) and (X == self.cache_input_pred).all().all()):
            pred = self.cache_output_pred
        else:
            for name, model in self.predictors.items():
                pred[name] = model.predict(X)
            self.cache_input_pred = X
            self.cache_output_pred = pred
            
        return pred

    def predict_proba(self, X) -> dict:
        pred = {}
        
        if((self.cache_input_proba is not None) and (X.shape == self.cache_input_proba.shape) and (X == self.cache_input_proba).all().all()):
            pred = self.cache_output_proba
        else:
            for name, model in self.predictors.items():
                pred[name] = model.predict_proba(X)
            self.cache_input_proba = X
            self.cache_output_proba = pred

        return pred
    
    def drop(self, *args):
        new_models = {}
        list_pred = []
        keys = list(self.predictors.keys())
        for i in range(len(keys)):
            if(keys[i] not in args):
                new_models[keys[i]] = self.predictors[keys[i]]
                list_pred.append(i)

        newPool = Pool(new_models)
        if(self.cache_output_pred):
            newPool.cache_input_pred = self.cache_input_pred
            newPool.cache_output_pred = self.cache_output_pred[list_pred]
        if(self.cache_output_proba):
            newPool.cache_input_proba = self.cache_input_proba
            newPool.cache_output_proba = self.cache_output_proba[list_pred]
                    
        return newPool

    def reject_rate_predict(self, X, reject_rate, reject_method:str='avg', warnings:bool=True):
        # reject_method deve ser igual a 'avg', 'median', 'min' ou 'max'
        if(reject_method not in ['avg', 'mean', 'median', 'min','max']):
            raise ValueError("reject_method deve ser igual a 'avg', 'mean, 'median', 'min' ou 'max'")
        
        poolProb = pd.DataFrame()
        predictions = self.predict(X)
        proba_results = self.predict_proba(X)

        for name, probas in proba_results.items():
            poolProb[name] = 1 - np.max(probas, axis=1)

        predictions = pd.DataFrame(np.array(st.mode(predictions.values, axis=1))[0])

        match reject_method:
            case 'max':
                predictions['score'] = poolProb.apply(lambda x: max(x), axis=1)
            case 'median':
                predictions['score'] = poolProb.apply(lambda x: np.median(x), axis=1)
            case 'min':
                predictions['score'] = poolProb.apply(lambda x: min(x), axis=1)
            case _: #avg or mean
                predictions['score'] = poolProb.apply(lambda x: np.mean(x), axis=1)

        if(reject_rate > 1):
            reject_rate = reject_rate/100

        n_reject = floor(poolProb.shape[0] * reject_rate)
        if(warnings):
            if(n_reject==0):
                print('Warning: Number of rejections equals 0. Increase the rejection rate.')
            elif(n_reject==poolProb.shape[0]):
                print('Warning: All examples will be rejected. Decrease the rejection rate.')

        predictions = predictions.sort_values(by='score', ascending=False)
        predictions.iloc[:n_reject, :] = np.nan # REJECTED
        predictions = predictions.drop(columns=['score']).sort_index()

        return predictions[0]

    def reject_threshold_predict(self, X, reject_threshold, reject_method:str='avg', assessor=None, warnings:bool=True):
        # reject_method deve ser igual a 'avg', 'median', 'min' ou 'max'
        if(reject_method not in ['avg', 'mean', 'median', 'min','max', 'assessor']):
            raise ValueError("reject_method deve ser igual a 'avg', 'mean, 'median', 'min' ou 'max'")
        
        #if(assessor is None):
        poolProb = pd.DataFrame()
        predictions = pd.DataFrame()
        results = self.predict_proba(X)

        for name, probas in results.items():
            poolProb[name] = 1 - np.max(probas, axis=1)

        predictions = pd.DataFrame(np.array(st.mode(predictions.values, axis=1))[0])

        match reject_method:
            case 'max':
                predictions['score'] = poolProb.apply(lambda x: max(x), axis=1)
            case 'median':
                predictions['score'] = poolProb.apply(lambda x: np.median(x), axis=1)
            case 'min':
                predictions['score'] = poolProb.apply(lambda x: min(x), axis=1)
            case _: #avg or mean
                predictions['score'] = poolProb.apply(lambda x: np.mean(x), axis=1)
            # Vou deixar assim mesmo só para ficar mais fácil de adicionar outros métodos depois
        
        # TODO: Parte do assessor como DES
        #else:
        #    poolProb = pd.DataFrame(assessor.predict(X), columns=list(self.predictors.keys()))

        if(reject_threshold > 1):
            reject_threshold = reject_threshold/100

        predictions.loc[predictions['score']<reject_threshold,:] = np.nan
        predictions = predictions.drop(columns=['score'])
        
        if(warnings):
            if(predictions.iloc[:,0].notna().all()):
                print('Warning: Number of rejections equals 0. Increase the rejection threshold.')
            elif(predictions.iloc[:,0].isna().all()):
                print('Warning: All examples will be rejected. Decrease the rejection threshold.')

        return predictions[0]