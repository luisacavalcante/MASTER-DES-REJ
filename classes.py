import pandas as pd
import numpy as np
from math import floor
from deslib.des import METADES as MetaDES
from scipy import stats as st
from sklearn.pipeline import NotFittedError
from sklearn.preprocessing import StandardScaler
from deslib.util.aggregation import (aggregate_proba_ensemble_weighted,
                                     sum_votes_per_class,
                                     get_weighted_votes)

class SpecificScaler(StandardScaler):
    def __init__(self, *, copy = True, with_mean = True, with_std = True):
        super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)
        self.chosen_features_ = None
        self.all_features_ = None

    def _prep_data(self, X, features:list=None):
        if(not isinstance(X, pd.DataFrame)):
            X = pd.DataFrame(X)
        if(features is not None):
            self.chosen_features_ = features
        elif(self.chosen_features_ is None):
            self.chosen_features_ = X.columns
        self.all_features_ = X.columns
        return X

    def fit(self, X:pd.DataFrame, features:list=None):
        '''Fit the scaler on specific columns of data'''
        X = self._prep_data(X.copy(), features)
        super().fit(X.loc[:,self.chosen_features_])
        return self

    def fit_transform(self, X:pd.DataFrame, features:list=None):
        '''Fit the scaler on specific columns of data and transform it'''
        X = self._prep_data(X.copy(), features)
        X.loc[:,self.chosen_features_] = super().fit_transform(X.loc[:,self.chosen_features_])

        return X

    def transform(self, X:pd.DataFrame):
        '''Use the fitted scaler to transform specific columns of data'''
        X = X.copy()
        if(not isinstance(X, pd.DataFrame)):
            X = pd.DataFrame(X, columns=self.all_features_)
        X.loc[:,self.chosen_features_] = super().transform(X.loc[:,self.chosen_features_])
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

        # No survey de reject option, uma instância era rejeitada se:
        #         =>      confiança < rejection threshold
        # Ou seja, como confiança é igual a 1-incerteza, ent o novo critério de rejeição seria:
        #         =>      incerteza > rejection threshold
        # O que tecnicamente poderia ser considerado um threshold de aceitação em ambos os casos, mas enfim né
        predictions.loc[predictions['score']>reject_threshold,:] = np.nan
        predictions = predictions.drop(columns=['score'])
        
        if(warnings):
            if(predictions.iloc[:,0].notna().all()):
                print('Warning: Number of rejections equals 0. Increase the rejection threshold.')
            elif(predictions.iloc[:,0].isna().all()):
                print('Warning: All examples will be rejected. Decrease the rejection threshold.')

        return predictions[0]

# TODO
class METADESR(MetaDES): # META-DES.Rejector (nome ainda não definido)
    """Meta learning for dynamic ensemble selection (META-DES).

    The META-DES framework is based on the assumption that the dynamic ensemble
    selection problem can be considered as a meta-problem. This meta-problem
    uses different criteria regarding the behavior of a base classifier
    :math:`c_{i}`, in order to decide whether it is competent enough to
    classify a given test sample.

    The framework performs a meta-training stage, in which, the meta-features
    are extracted from each instance belonging to the training and the dynamic
    selection dataset (DSEL). Then, the extracted meta-features are used
    to train the meta-classifier :math:`\\lambda`. The meta-classifier is
    trained to predict whether or not a base classifier :math:`c_{i}` is
    competent enough to classify a given input sample.

    When an unknown sample is presented to the system, the meta-features for
    each base classifier :math:`c_{i}` in relation to the input sample are
    calculated and presented to the meta-classifier. The meta-classifier
    estimates the competence level of the base classifier :math:`c_{i}` for
    the classification of the query sample. Base classifiers with competence
    level higher than a pre-defined threshold are selected. If no base
    classifier is selected, the whole pool is used for classification.

    Parameters
    ----------
     pool_classifiers : list of classifiers (Default = None)
        The generated_pool of classifiers trained for the corresponding
        classification problem. Each base classifiers should support the method
        "predict". If None, then the pool of classifiers is a bagging
        classifier.

    meta_classifier :   sklearn.estimator (Default = None)
                        Classifier model used for the meta-classifier. If None,
                        a Multinomial naive Bayes classifier is used.

    k : int (Default = 7)
        Number of neighbors used to estimate the competence of the base
        classifiers.

    Kp : int (Default = 5)
         Number of output profiles used to estimate the competence of the
         base classifiers.

    Hc : float (Default = 1.0)
         Sample selection threshold.

    selection_threshold : float(Default = 0.5)
        Threshold used to select the base classifier. Only the base classifiers
        with competence level higher than the selection_threshold are selected
        to compose the ensemble.

    mode : String (Default = "selection")
        Determines the mode of META-des that is used
        (selection, weighting or hybrid).

    DFP : Boolean (Default = False)
        Determines if the dynamic frienemy pruning is applied.

    with_IH : Boolean (Default = False)
        Whether the hardness level of the region of competence is used to
        decide between using the DS algorithm or the KNN for classification of
        a given query sample.

    safe_k : int (default = None)
        The size of the indecision region.

    IH_rate : float (default = 0.3)
        Hardness threshold. If the hardness level of the competence region is
        lower than the IH_rate the KNN classifier is used. Otherwise, the DS
        algorithm is used for classification.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    knn_classifier : {'knn', 'faiss', None} (Default = 'knn')
         The algorithm used to estimate the region of competence:

         - 'knn' will use :class:`KNeighborsClassifier` from sklearn
          :class:`KNNE` available on `deslib.utils.knne`

         - 'faiss' will use Facebook's Faiss similarity search through the
           class :class:`FaissKNNClassifier`

         - None, will use sklearn :class:`KNeighborsClassifier`.

    knn_metric : {'minkowski', 'cosine', 'mahalanobis'} (Default = 'minkowski')
        The metric used by the k-NN classifier to estimate distances.

        - 'minkowski' will use minkowski distance.

        - 'cosine' will use the cosine distance.

        - 'mahalanobis' will use the mahalonibis distance.

        Note: This parameter only affects the neighborhood search applied in
        the feature space.

    knne : bool (Default=False)
        Whether to use K-Nearest Neighbor Equality (KNNE) for the region
        of competence estimation.

    DSEL_perc : float (Default = 0.5)
        Percentage of the input data used to fit DSEL.
        Note: This parameter is only used if the pool of classifier is None or
        unfitted.

    voting : {'hard', 'soft'}, default='hard'
            If 'hard', uses predicted class labels for majority rule voting.
            Else if 'soft', predicts the class label based on the argmax of
            the sums of the predicted probabilities, which is recommended for
            an ensemble of well-calibrated classifiers.

    n_jobs : int, default=-1
        The number of parallel jobs to run. None means 1 unless in
        a joblib.parallel_backend context. -1 means using all processors.
        Doesn’t affect fit method.

    References
    ----------
    Cruz, R.M., Sabourin, R., Cavalcanti, G.D. and Ren, T.I., 2015. META-DES:
    A dynamic ensemble selection framework using meta-learning.
    Pattern Recognition, 48(5), pp.1925-1935.

    Cruz, R.M., Sabourin, R. and Cavalcanti, G.D., 2015, July. META-des. H:
    a dynamic ensemble selection technique using meta-learning and a dynamic
    weighting approach. In Neural Networks (IJCNN), 2015 International Joint
    Conference on (pp. 1-8).

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier
    selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    """
    def __init__(self, pool_classifiers=None, meta_classifier=None, k=7, Kp=5, Hc=1, selection_threshold=0.5, rejection_threshold=0.5,  DFP=False, with_IH=False, safe_k=None, IH_rate=0.3, random_state=None, knn_classifier='knn', knne=False, knn_metric='minkowski', DSEL_perc=0.5, n_jobs=-1, voting='soft', rejection_method='median'):
        super().__init__(pool_classifiers, meta_classifier, k, Kp, Hc, selection_threshold, 'selection', DFP, with_IH, safe_k, IH_rate, random_state, knn_classifier, knne, knn_metric, DSEL_perc, n_jobs, voting)
        self.rejection_method = rejection_method
        self.rejection_threshold = rejection_threshold

    def _validate_parameters(self):
        """Check if the parameters passed as argument are correct.

        Raises
        -------
        ValueError
            If any of the hyper-parameters are invalid.
        """
        if not isinstance(self.Hc, (float, int)):
            raise ValueError(
                'Parameter Hc should be either a number.'
                ' Currently Hc = {}'.format(type(self.Hc)))

        if self.Hc < 0.5:
            raise ValueError(
                'Parameter Hc should be higher than 0.5.'
                ' Currently Hc = {}'.format(self.Hc))

        if not isinstance(self.selection_threshold, float):
            raise ValueError(
                'Parameter Hc should be either a float.'
                ' Currently Hc = {}'.format(type(self.Hc)))

        # v Qualquer valor válido para selection_threshold
        #if self.selection_threshold < 0.5:
        #    raise ValueError(
        #        'Parameter selection_threshold should be higher than 0.5. '
        #        'Currently selection_threshold = {}'.format(
        #            self.selection_threshold))

        if (self.meta_classifier is not None and
                not hasattr(self.meta_classifier, "predict_proba")):

            raise ValueError(
                "The meta-classifier should output probability estimates")

        if self.Kp is not None:
            if not isinstance(self.Kp, int):
                raise TypeError("parameter Kp should be an integer.")
            if self.Kp <= 0:
                raise ValueError("parameter Kp must be equal orhigher than 1."
                                 "input Kp is {} .".format(self.Kp))
        else:
            raise ValueError("Parameter Kp is 'None'.")

        super()._validate_parameters()

    def predict(self, X):
        """Predict the class label for each sample in X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        predicted_labels : array of shape (n_samples)
                           Predicted class label for each sample in X.
        """
        probas = self.predict_proba(X)
        preds = probas.argmax(axis=1).astype(float)
        # Rejected by uncertainty
        subgroup_mask = np.isnan(probas[:,0]) # Sempre que uma instância for rejeitada, todas as probabilidades de classe serão np.nan
        preds[subgroup_mask] = np.nan
        # Accepted
        subgroup_mask = ~subgroup_mask
        preds[subgroup_mask] = self.classes_.take(preds[subgroup_mask].astype(int))
        return preds

    def predict_proba(self, X):
        """Estimates the posterior probabilities for sample in X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        predicted_proba : array of shape (n_samples, n_classes)
                          Probabilities estimates for each sample in X.
        """
        X = self._check_predict(X)

        self._check_predict_proba()
        probas = np.zeros((X.shape[0], self.n_classes_))
        base_preds, base_probas = self._preprocess_predictions(X, True)
        # predict all agree
        ind_disagreement, ind_all_agree = self._split_agreement(base_preds)
        if ind_all_agree.size:
            probas[ind_all_agree] = base_probas[ind_all_agree].mean(axis=1)
        # predict with IH
        if ind_disagreement.size:
            distances, ind_ds_classifier, neighbors = self._IH_prediction(
                    X, ind_disagreement, probas, is_proba=True)
            # Predict with DS - Check if there are still samples to be labeled.
            if ind_ds_classifier.size:
                DFP_mask = self._get_DFP_mask(neighbors)
                inds, sel_preds, sel_probas = self._prepare_indices_DS(
                    base_preds, base_probas, ind_disagreement,
                    ind_ds_classifier)
                probas_ds = self.predict_proba_with_ds(sel_preds,
                                                       sel_probas,
                                                       neighbors, distances,
                                                       DFP_mask)
                probas[inds] = probas_ds

        # Rejection by uncertainty
        #rej_indices = (1 - probas.max(axis=1)) > self.rejection_threshold
        #probas[rej_indices] = np.full(shape=probas.shape[1], fill_value=np.nan)
        return probas

    def classify_with_ds(self, predictions, probabilities=None,
                         competence_region=None, distances=None,
                         DFP_mask=None):
        """Predicts the label of the corresponding query sample.

        If self.mode == "selection", the selected ensemble is combined using
        the majority voting rule

        If self.mode == "weighting", all base classifiers are used for
        classification, however their influence in the final decision are
        weighted according to their estimated competence level. The weighted
        majority voting scheme is used to combine the decisions of the
        base classifiers.

        If self.mode == "hybrid",  A hybrid Dynamic selection and weighting
        approach is used. First an ensemble with the competent base classifiers
        are selected. Then, their decisions are aggregated using the weighted
        majority voting rule according to its competence level estimates.

        Parameters
        ----------
        predictions : array of shape (n_samples, n_classifiers)
                      Predictions of the base classifier for all test examples.

        probabilities : array of shape (n_samples, n_classifiers, n_classes)
            Probabilities estimates of each base classifier for all test
            examples. (For methods that always require probabilities from
            the base classifiers).

        competence_region : array of shape (n_samples, n_neighbors)
            Indices of the k nearest neighbors according for each test sample.

        distances : array of shape (n_samples, n_neighbors)
                        Distances from the k nearest neighbors to the query

        DFP_mask : array of shape (n_samples, n_classifiers)
            Mask containing 1 for the selected base classifier and 0 otherwise.

        Returns
        -------
        predicted_label : array of shape (n_samples)
                          Predicted class label for each test example.
        """
        probas = self.predict_proba_with_ds(predictions, probabilities,
                                            competence_region, distances,
                                            DFP_mask)

        na_indices = np.all(np.isnan(probas), axis=1)
        probas = probas.argmax(axis=1).astype(float)
        probas[na_indices] = np.nan
        return probas

    def select(self, competences):
        """Selects the base classifiers that obtained a competence level higher
        than the predefined threshold defined in self.selection_threshold.

        Parameters
        ----------
        competences : array of shape (n_samples, n_classifiers)
            The competence level estimated for each base classifier and test
            example.

        Returns
        -------
        selected_classifiers : array of shape (n_samples, n_classifiers)
            Boolean matrix containing True if the base classifier is selected,
            False otherwise.

        """
        if competences.ndim < 2:
            competences = competences.reshape(1, -1)

        # Rejection by model competence
        selected_classifiers = (competences > self.selection_threshold)
        match self.rejection_method:
            case 'median':
                competences = np.median(competences, axis=1)
            case 'max':
                competences = np.max(competences, axis=1)
            case 'min':
                competences = np.min(competences, axis=1)
            case _:
                competences = np.mean(competences, axis=1)
        selected_instances = (competences > self.selection_threshold)
        # For the rows that are all False (i.e., no base classifier was
        # selected, select all classifiers (all True)
        #selected_classifiers[~np.any(selected_classifiers, axis=1), :] = True # < Desligado

        return selected_classifiers, selected_instances
    
    def _dynamic_selection(self, competences, predictions, probabilities):
        """ Combine models using dynamic ensemble selection. """
        selected_classifiers, selected_instances = self.select(competences)
        if self.voting == 'hard':
            votes = np.ma.MaskedArray(predictions, ~selected_classifiers)
            votes = sum_votes_per_class(votes, self.n_classes_)
            predicted_proba = votes / votes.sum(axis=1)[:, None]
        else:
            masked_proba = self._mask_proba(probabilities,
                                            selected_classifiers)
            #match self.rejection_method:
            #    case 'median':
            #        predicted_proba = np.median(masked_proba, axis=1)
            #    case 'max':
            #        predicted_proba = np.max(masked_proba, axis=1)
            #    case 'min':
            #        predicted_proba = np.min(masked_proba, axis=1)
            #    case _: # average/mean
            #        predicted_proba = np.mean(masked_proba, axis=1)
            predicted_proba = np.ma.MaskedArray(np.mean(masked_proba, axis=1), ~selected_instances)
        return predicted_proba
    
    def _hybrid(self, competences, predictions, probabilities):
        """ Combine models using a hybrid dynamic selection + weighting. """
        selected_classifiers, selected_instances = self.select(competences)
        if self.voting == 'hard':
            votes = np.ma.MaskedArray(predictions, ~selected_classifiers)
            w_votes, _ = get_weighted_votes(votes, competences,
                                            np.arange(self.n_classes_))
            predicted_proba = w_votes / w_votes.sum(axis=1)[:, None]
        else:
            masked_proba = self._mask_proba(probabilities,
                                            selected_classifiers)
            predicted_proba = aggregate_proba_ensemble_weighted(
                masked_proba, competences)
            predicted_proba = np.ma.MaskedArray(predicted_proba, ~selected_instances)
        return predicted_proba
    
    def set_thresholds(self, rejection_threshold:float=None, selection_threshold:float=None):
        if(rejection_threshold is not None):
            self.rejection_threshold = rejection_threshold
        if(selection_threshold is not None):
            self.selection_threshold = selection_threshold