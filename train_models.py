import json
import pandas as pd
import numpy as np
import re
import mlflow
import mlflow.sklearn
import optuna
#import logging
from sklearn.pipeline import Pipeline
from classes import SpecificScaler
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC#, SVR # kernels: 'linear', 'poly' e 'rbf'
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier#, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.calibration import CalibratedClassifierCV
from deslib.des import METADES

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')

# =================================================

with open('config.json', 'r') as f:
    CONFIG = json.load(f)

#with open('config.json', 'w') as f:
#    CONFIG['SEED'] = randint(0, 4294967295)
#    json.dump(CONFIG, f)
#    print(CONFIG['SEED'])

# ==================================================

def auroc_score(model, x, y):
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(x)
            # Para classificação binária
            if y_proba.shape[1] == 2:
                auroc = roc_auc_score(y, y_proba[:, 1])
            # Para classificação multiclasse
            else:
                auroc = roc_auc_score(y, y_proba)
        elif hasattr(model, 'decision_function'):
            y_scores = model.decision_function(x)
            auroc = roc_auc_score(y, y_scores)
        else:
            auroc = None
    except Exception as e:
        auroc = None
    return auroc
    

def log_model_to_mlflow(model, model_name, hyperparams, X_train, y_train, X_test, y_test):
    """
    Treina, avalia e registra um modelo no MLflow.
    
    Parameters:
    -----------
    model : estimator
        Modelo do scikit-learn ou compatível
    model_name : str
        Nome do modelo para registro
    hyperparams : dict
        Dicionário com os hiperparâmetros do modelo
    X_train, y_train : array-like
        Dados de treino
    X_test, y_test : array-like
        Dados de teste
    """
    with mlflow.start_run(run_name=model_name):
        # Fazer predições
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calcular probabilidades para AUROC (se disponível)
        train_auroc = auroc_score(model, X_train, y_train)
        test_auroc = auroc_score(model, X_test, y_test)
        
        # Calcular métricas
        train_accuracy = accuracy_score(y_train, y_pred_train)
        if(len(set(y_train)) > 2):
            train_precision = precision_score(y_train, y_pred_train, zero_division=0, average='macro')
            train_recall = recall_score(y_train, y_pred_train, zero_division=0, average='macro')
            train_f1 = f1_score(y_train, y_pred_train, zero_division=0, average='macro')
        else:
            train_precision = precision_score(y_train, y_pred_train, zero_division=0)
            train_recall = recall_score(y_train, y_pred_train, zero_division=0)
            train_f1 = f1_score(y_train, y_pred_train, zero_division=0)
        
        test_accuracy = accuracy_score(y_test, y_pred_test)
        if(len(set(y_test)) > 2):
            test_precision = precision_score(y_test, y_pred_test, zero_division=0, average='macro')
            test_recall = recall_score(y_test, y_pred_test, zero_division=0, average='macro')
            test_f1 = f1_score(y_test, y_pred_test, zero_division=0, average='macro')
        else:
            test_precision = precision_score(y_test, y_pred_test, zero_division=0)
            test_recall = recall_score(y_test, y_pred_test, zero_division=0)
            test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
        
        # Registrar hiperparâmetros
        mlflow.log_params(hyperparams)
        
        # Registrar métricas
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_precision", train_precision)
        mlflow.log_metric("train_recall", train_recall)
        mlflow.log_metric("train_f1_score", train_f1)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1_score", test_f1)
        if train_auroc is not None:
            mlflow.log_metric("train_auroc", train_auroc)
        if test_auroc is not None:
            mlflow.log_metric("test_auroc", test_auroc)
        
        # Registrar modelo
        mlflow.sklearn.log_model(model, name=model_name)
    
def load_model_by_name(experiment_name, run_name):
    """
    Carrega um modelo específico do MLflow pelo nome da run.
    
    Parameters:
    -----------
    experiment_name : str
        Nome do experimento MLflow
    run_name : str
        Nome da run (ex: "Random_Forest", "XGBoost")
    
    Returns:
    --------
    model : estimator
        Modelo carregado
    run_info : dict
        Informações da run (hiperparâmetros e métricas)
    """
    # Buscar experimento
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experimento '{experiment_name}' não encontrado!")
    
    # Buscar runs do experimento
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'"
    )
    
    if runs.empty:
        raise ValueError(f"Run '{run_name}' não encontrada no experimento '{experiment_name}'!")
    
    # Pegar a run mais recente se houver múltiplas
    run = runs.iloc[0]
    run_id = run.run_id
    print(run_id)
    # Carregar modelo
    model_uri = f"runs:/{run_id}/{run_name}"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Extrair informações da run
    run_info = {
        'params': {k.replace('params.', ''): v for k, v in run.items() if k.startswith('params.')},
        'metrics': {k.replace('metrics.', ''): v for k, v in run.items() if k.startswith('metrics.')}
    }
    
    return model, run_info

# ======================================================

from sklearn.preprocessing import OneHotEncoder, StandardScaler#, MinMaxScaler
from sklearn.model_selection import train_test_split

def transform_property(x):
    x = re.sub('(Private room( in )?)|(Shared room( in )?)|(Entire )|(Room in )', '', x).lower()
    if(x=='casa particular'):
        x='home'
    
    if(x not in ['rental unit','home','condo','loft','serviced apartment']):
        x='other'

    return x

def get_data(dataset:str):
    global CONFIG
    if(dataset in ['twomoons','circles','aniso','blobs','varied']):
        if(dataset=='twomoons'):
            X, y = make_moons(n_samples=CONFIG['TWO_MOONS']['N_SAMPLES'], 
                            noise=CONFIG['TWO_MOONS']['NOISE'],
                            random_state=CONFIG['SEED'])
        elif(dataset=='circles'):
            X, y = make_circles(n_samples=CONFIG['CIRCLES']['N_SAMPLES'], 
                            noise=CONFIG['CIRCLES']['NOISE'], 
                            factor=CONFIG['CIRCLES']['FACTOR'],
                            random_state=CONFIG['SEED'])
        elif(dataset=='aniso'):
            X, y = make_blobs(n_samples=CONFIG['ANISO']['N_SAMPLES'], 
                                       centers=CONFIG['ANISO']['CENTERS'],
                                       cluster_std=CONFIG['ANISO']['CLUSTER_STD'],
                                       random_state=CONFIG['SEED'])
            X = np.dot(X, CONFIG['ANISO']['TRANSFORM_VECTOR'])
        elif(dataset=='blobs'):
            X, y = make_blobs(n_samples=CONFIG['BLOBS']['N_SAMPLES'], 
                                       centers=CONFIG['BLOBS']['CENTERS'],
                                       cluster_std=CONFIG['BLOBS']['CLUSTER_STD'],
                                       random_state=CONFIG['SEED'])
        elif(dataset=='varied'):
            X, y = make_blobs(n_samples=CONFIG['VARIED']['N_SAMPLES'],
                                       centers=CONFIG['VARIED']['CENTERS'],
                                       cluster_std=CONFIG['VARIED']['CLUSTER_STD'],
                                       random_state=CONFIG['SEED'])
            
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.67, shuffle=True, random_state=CONFIG['SEED'])
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=CONFIG['SEED'])
        scaler = SpecificScaler().fit(X_train)

    elif(dataset=='covid'):
        df_covid = pd.read_csv('data/hosp1_v8 (1).csv') # 526 exemplos
        X_train, y_train = df_covid.drop(columns=['severity']), df_covid['severity']

        df_covid = pd.read_csv('data/hospital2 (2).csv').drop(columns=['creatino.fosfoquinase.cpk.plasma.ck',
                                                                        'troponina.i.plasma.troponina.i']) # 134 exemplos
        X_temp, y_temp = df_covid.drop(columns=['severity']), df_covid['severity']
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=CONFIG['SEED'])

        nmrc_cols = X_train.columns[1:]
        scaler = SpecificScaler().fit(X_train, nmrc_cols)

        return X_train, X_val, X_test, y_train, y_val, y_test, scaler

    elif(dataset=='airbnb'):
        df = pd.read_csv('data/listings.csv')

        bathrooms = df['bathrooms_text'].str.extract('([0-9\.]+)?([- A-Za-z]+)')#[[0,2]]
        bathrooms[1] = bathrooms[1].apply(lambda x: x if pd.isna(x) else x.strip().lower().replace('baths','bath'))
        bathrooms.columns = ['n_baths', 'bath_type']

        for i in range(len(bathrooms)):
            bt = bathrooms.at[i,'bath_type']
            if(pd.notna(bt)):
                if(re.search('half', bt)):
                    bt = re.sub('half-', '', bt)
                    bathrooms.loc[i,:] = [0.5, bt]

                if(bt=='bath'):
                    bathrooms.at[i,'bath_type'] = 'regular bath'

        df['bathrooms'] = bathrooms['n_baths'].astype(float)
        df['bathroom_type'] = bathrooms['bath_type']

        df = df[[
            'host_response_time',
            'host_response_rate',
            'host_is_superhost',
            'host_total_listings_count',
            'host_identity_verified',
            'latitude',
            'longitude',
            'property_type',
            'room_type',
            'accommodates',
            'bathrooms',
            'bathroom_type',
            'bedrooms',
            'beds',
            'number_of_reviews',
            'review_scores_rating',
            'review_scores_checkin',
            'review_scores_communication',
            'review_scores_location',
            'minimum_nights',
            'maximum_nights',
            'availability_30',
            'price'
        ]].dropna()

        df['host_response_rate'] = df['host_response_rate'].str.replace('%', '').astype(float)
        df['host_response_time'] = df['host_response_time'].astype('category').cat.reorder_categories(['within an hour', 'within a few hours', 'within a day', 'a few days or more']).cat.codes
        df[['host_is_superhost','host_identity_verified']] = df[['host_is_superhost','host_identity_verified']].map(lambda x: x=='t')
        df['property_type'] = df['property_type'].apply(transform_property)
        df['price'] = df['price'].str.replace('[,\$]','', regex=True).astype(float)>300
        
        X, y = df.drop(columns=['price']), df['price'].astype(int)

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.67, shuffle=True, random_state=CONFIG['SEED'])
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=CONFIG['SEED'])

        onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        onehot = onehot.set_output(transform='pandas')
        X_train = pd.concat([X_train.drop(columns=['property_type','room_type','bathroom_type']), onehot.fit_transform(X_train[['property_type','room_type','bathroom_type']], y_train)], axis=1)
        X_val = pd.concat([X_val.drop(columns=['property_type','room_type','bathroom_type']), onehot.transform(X_val[['property_type','room_type','bathroom_type']])], axis=1)
        X_test = pd.concat([X_test.drop(columns=['property_type','room_type','bathroom_type']), onehot.transform(X_test[['property_type','room_type','bathroom_type']])], axis=1)

        nmrc_cols = ['host_response_time','host_response_rate','host_total_listings_count',
                    'latitude','longitude','accommodates','bathrooms','bedrooms','beds',
                    'number_of_reviews','review_scores_rating','review_scores_checkin',
                    'review_scores_communication','review_scores_location',
                    'minimum_nights','maximum_nights','availability_30']

        scaler = SpecificScaler().fit(X_train, nmrc_cols)

    elif(dataset=='heloc'):
        df = pd.read_csv('data/heloc_dataset_v1 (1).csv')

        X, y = df.drop(columns=['RiskPerformance']), df['RiskPerformance'].replace({'Bad':0, 'Good':1}).astype(int)

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.67, random_state=CONFIG['SEED'], shuffle=True)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=CONFIG['SEED'], shuffle=True)

        scaler = StandardScaler().fit(X_train)

    elif(dataset=='covertype'):
        df = pd.read_csv('data/covertype.csv')
        target = 'Cover_Type'

        df_sample, _ = train_test_split(
            df, 
            test_size=0.9,
            stratify=df[target],
            random_state=CONFIG['SEED'],
            shuffle=True
        )

        X = df_sample.drop(columns=[target])
        y = df_sample[target].astype(int)-1

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.67, stratify=y, random_state=CONFIG['SEED'], shuffle=True)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=CONFIG['SEED'], shuffle=True)

        nmrc_cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                    'Horizontal_Distance_To_Fire_Points']
        
        scaler = StandardScaler().fit(X_train, nmrc_cols)

    elif(dataset=='churn'):
        df = pd.read_csv(f'data/customer_churn_telecom_services.csv', header=0)

        continuous_cols = []
        cat_cols = []

        for col in df.drop(columns=['Churn']).columns:
            unique_values = df[col].value_counts()
            if(len(unique_values) <= 4):
                cat_cols.append(col)
            else:
                continuous_cols.append(col)

        df['TotalCharges'] = df['TotalCharges'].fillna(0)

        rdict = {'gender': {'Male': 0, 'Female': 1},
                'Partner': {'No': 0, 'Yes': 1},
                'Dependents': {'No': 0, 'Yes': 1},
                'PhoneService': {'No': 0, 'Yes': 1},
                'PaperlessBilling': {'No': 0, 'Yes': 1},
                'Churn': {'No': 0, 'Yes': 1},
                }

        rdict['MultipleLines'] = {'No phone service': 0, 'No': 1, 'Yes': 2}

        cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies']

        for col in cols:
            rdict[col] = {'No internet service': 0, 'No': 1, 'Yes': 2}

        rdict['InternetService'] = {'No': 0, 'DSL': 1, 'Fiber optic': 2}
        rdict['Contract'] = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        rdict['PaymentMethod'] = {'Credit card (automatic)': 0, 'Bank transfer (automatic)': 1,
                                'Mailed check': 2, 'Electronic check': 3}

        df = df.replace(rdict)
        df = df.rename(columns={'Churn': 'target'}) 
        cols = df.drop(columns=['target']).columns

        temp = df[df.target==1]
        train_pos, temp_pos = train_test_split(temp, test_size=0.67, shuffle=True, random_state=CONFIG['SEED'])
        val_pos, test_pos = train_test_split(temp_pos, test_size=0.5, shuffle=True, random_state=CONFIG['SEED'])

        temp = df[df.target==0]
        train_neg, temp_neg = train_test_split(temp, test_size=0.67, shuffle=True, random_state=CONFIG['SEED'])
        val_neg, test_neg = train_test_split(temp_neg, test_size=0.5, shuffle=True, random_state=CONFIG['SEED'])

        X_train = pd.concat([train_pos[cols], train_neg[cols]], ignore_index=True)
        y_train = pd.concat([train_pos['target'], train_neg['target']], ignore_index=True)

        X_val = pd.concat([val_pos[cols], val_neg[cols]], ignore_index=True)
        y_val = pd.concat([val_pos['target'], val_neg['target']], ignore_index=True)

        X_test = pd.concat([test_pos[cols], test_neg[cols]], ignore_index=True)
        y_test = pd.concat([test_pos['target'], test_neg['target']], ignore_index=True)

        scaler = SpecificScaler().fit(X_train, continuous_cols)

        o_sampler = RandomOverSampler(random_state=CONFIG['SEED'])
        X_train, y_train = o_sampler.fit_resample(X_train, y_train)

    else:
        raise ValueError('Dataset Not Usable')
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

# ======================================================

def searchAndTrain(dataset, experiment_name, num_trials, load=False):

    mlflow.set_experiment(experiment_name=experiment_name)

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = get_data(dataset)
    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)

    scorer_string = 'f1_macro' if len(set(y_train))>2 else 'f1'
    if len(X_val) < 750:
        cal_method = 'isotonic'
        num_cv_folds = 5
    else:
        cal_method = 'sigmoid'
        num_cv_folds = 10

    # 1. Define an objective function to be maximized.
    def dtree_objective(trial:optuna.trial._trial.Trial):
        
        # 2. Suggest values for the hyperparameters using a trial object.
        max_depth = trial.suggest_int('max_depth', 5, 100, log=True)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
        min_samples_split = trial.suggest_int('min_samples_split', 2, 60)
        min_samples_leaf = trial.suggest_int('min_samples_leaf',1, 30)
        
        clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf)
        score = cross_val_score(clf, X_train, y_train, scoring=scorer_string, n_jobs=-1, cv=num_cv_folds).mean()
        
        return score

    def sgd_objective(trial):
        loss = trial.suggest_categorical('loss', ['log_loss', 'modified_huber'])
        penalty = trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet'])
        alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
        learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive'])
        eta0 = trial.suggest_float('eta0', 1e-5, 1e-1, log=True)
        max_iter = trial.suggest_int('max_iter', 500, 2000)
        
        clf = SGDClassifier(
            loss=loss, penalty=penalty, alpha=alpha, 
            learning_rate=learning_rate, eta0=eta0, max_iter=max_iter,
            random_state=CONFIG['SEED']
        )
        
        score = cross_val_score(clf, X_train_norm, y_train, scoring=scorer_string, n_jobs=-1, cv=num_cv_folds)
        return score.mean()

    def logreg_objective(trial):
        #penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
        C = trial.suggest_float('C', 1e-3, 100, log=True)
        solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga'])
        max_iter = trial.suggest_int('max_iter', 100, 1000)
        
        # Ajusta solver baseado no penalty
        params = {
            'penalty': 'l2', 'C': C, 'solver': solver, 
            'max_iter': max_iter, 'random_state': CONFIG['SEED']
        }
        
        clf = LogisticRegression(**params)
        score = cross_val_score(clf, X_train_norm, y_train, scoring=scorer_string, n_jobs=-1, cv=num_cv_folds)
        return score.mean()

    def knn_objective(trial):
        n_neighbors = trial.suggest_int('n_neighbors', 1, 50)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])
        p = trial.suggest_int('p', 1, 5) if metric == 'minkowski' else 2
        
        clf = KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights, 
            metric=metric, p=p
        )
        
        score = cross_val_score(clf, X_train_norm, y_train, scoring=scorer_string, n_jobs=-1, cv=num_cv_folds)
        return score.mean()

    def svm_linear_objective(trial):
        C = trial.suggest_float('C', 1e-3, 100, log=True)
        max_iter = trial.suggest_int('max_iter', 500, 3000)
        
        clf = SVC(
            kernel='linear', C=C, max_iter=max_iter, random_state=CONFIG['SEED']
        )
        
        score = cross_val_score(clf, X_train_norm, y_train, scoring=scorer_string, n_jobs=-1, cv=num_cv_folds)
        return score.mean()

    def svm_poly_objective(trial):
        C = trial.suggest_float('C', 1e-3, 100, log=True)
        degree = trial.suggest_int('degree', 2, 5)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        coef0 = trial.suggest_float('coef0', 0, 10)
        max_iter = trial.suggest_int('max_iter', 500, 3000)
        
        clf = SVC(
            kernel='poly', C=C, degree=degree, gamma=gamma, 
            coef0=coef0, max_iter=max_iter, random_state=CONFIG['SEED']
        )
        
        score = cross_val_score(clf, X_train_norm, y_train, scoring=scorer_string, n_jobs=-1, cv=num_cv_folds)
        return score.mean()

    def svm_rbf_objective(trial):
        C = trial.suggest_float('C', 1e-3, 100, log=True)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        max_iter = trial.suggest_int('max_iter', 500, 3000)
        
        clf = SVC(
            kernel='rbf', C=C, gamma=gamma, max_iter=max_iter, random_state=CONFIG['SEED']
        )
        
        score = cross_val_score(clf, X_train_norm, y_train, scoring=scorer_string, n_jobs=-1, cv=num_cv_folds)
        return score.mean()

    def mlp_objective(trial):
        
        hidden_layer_sizes = trial.suggest_int('hidden_layer_sizes', 50, 500, step=5)
        activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])
        solver = trial.suggest_categorical('solver', ['adam', 'sgd'])
        alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
        learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'adaptive'])
        learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
        #max_iter = trial.suggest_int('max_iter', 200, 1000)
        
        clf = MLPClassifier(max_iter=10000, early_stopping=True, 
            n_iter_no_change=20, shuffle=True,
            hidden_layer_sizes=(hidden_layer_sizes,), activation=activation,
            solver=solver, alpha=alpha, learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            random_state=CONFIG['SEED']
        )
        
        score = cross_val_score(clf, X_train_norm, y_train, scoring=scorer_string, n_jobs=-1, cv=num_cv_folds)
        return score.mean()

    def rf_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 500, log=True)
        max_depth = trial.suggest_int('max_depth', 5, 100, log=True)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 60)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 30)
        #max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
        
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, criterion=criterion,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            #max_features=max_features, 
            random_state=CONFIG['SEED'], n_jobs=-1
        )
        
        score = cross_val_score(clf, X_train, y_train, scoring=scorer_string, n_jobs=-1, cv=num_cv_folds)
        return score.mean()

    def gb_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 500, log=True)
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 1, log=True)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 60)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 30)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf, subsample=subsample,
            max_features=max_features, random_state=CONFIG['SEED']
        )
        
        score = cross_val_score(clf, X_train, y_train, scoring=scorer_string, n_jobs=-1, cv=num_cv_folds)
        return score.mean()

    def ada_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 500, log=True)
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 2, log=True)
        #algorithm = trial.suggest_categorical('algorithm', ['SAMME', 'SAMME.R'])
        
        clf = AdaBoostClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            #algorithm=algorithm, 
            random_state=CONFIG['SEED']
        )
        
        score = cross_val_score(clf, X_train, y_train, scoring=scorer_string, n_jobs=-1, cv=num_cv_folds)
        return score.mean()

    def xgb_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 500, log=True)
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 1, log=True)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
        gamma = trial.suggest_float('gamma', 0, 5)
        #subsample = trial.suggest_float('subsample', 0.5, 1.0)
        #colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        reg_alpha = trial.suggest_float('reg_alpha', 1e-5, 100, log=True)
        reg_lambda = trial.suggest_float('reg_lambda', 1e-5, 100, log=True)
        
        clf = XGBClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, min_child_weight=min_child_weight,
            gamma=gamma, #1subsample=subsample, colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha, reg_lambda=reg_lambda, 
            random_state=CONFIG['SEED'], n_jobs=-1, eval_metric='logloss'
        )
        
        score = cross_val_score(clf, X_train, y_train, scoring=scorer_string, n_jobs=1, cv=num_cv_folds)
        return score.mean()

    # ============================================

    loaded_models = []
    
    # Decision Tree
    try:
        model = load_model_by_name(experiment_name=experiment_name, run_name='Decision_Tree')[0]
        loaded_models.append(['Decision_Tree', model])
    except ValueError:
        dtree_study = optuna.create_study(direction='maximize')
        dtree_study.optimize(dtree_objective, n_trials=num_trials)
        dtree_params = dtree_study.best_params
        base_model = DecisionTreeClassifier(**dtree_params, random_state=CONFIG['SEED']).fit(X_train, y_train)
        model = CalibratedClassifierCV(FrozenEstimator(base_model), method=cal_method, cv=num_cv_folds).fit(X_val, y_val)
        log_model_to_mlflow(
            model, "Decision_Tree", dtree_params, 
            X_train, y_train, X_test, y_test
        )
        loaded_models.append(['Decision_Tree', model])

# SGD
    try:
        model = load_model_by_name(experiment_name=experiment_name, run_name='SGD')[0]
        loaded_models.append(['SGD', model])
    except ValueError:
        sgd_study = optuna.create_study(direction='maximize')
        sgd_study.optimize(sgd_objective, n_trials=num_trials)
        sgd_params = sgd_study.best_params
        # 1. Treina o modelo base antes
        base_clf = SGDClassifier(**sgd_params, random_state=CONFIG['SEED']).fit(X_train_norm, y_train)
        # 2. Calibra o modelo treinado e inclui no Pipeline
        calibrated_clf = CalibratedClassifierCV(FrozenEstimator(base_clf), method=cal_method, cv=num_cv_folds).fit(X_val, y_val)
        model = Pipeline([
            ('scaler', scaler),
            ('clf', calibrated_clf)
        ])
        log_model_to_mlflow(model, "SGD", sgd_params, X_train_norm, y_train, X_test_norm, y_test)
        loaded_models.append(['SGD', model])

    # Logistic Regression
    try:
        model = load_model_by_name(experiment_name=experiment_name, run_name='Logistic_Regression')[0]
        loaded_models.append(['Logistic_Regression', model])
    except ValueError:
        logreg_study = optuna.create_study(direction='maximize')
        logreg_study.optimize(logreg_objective, n_trials=num_trials)
        logreg_params = logreg_study.best_params
        base_clf = LogisticRegression(**logreg_params, random_state=CONFIG['SEED']).fit(X_train_norm, y_train)
        calibrated_clf = CalibratedClassifierCV(FrozenEstimator(base_clf), method=cal_method, cv=num_cv_folds).fit(X_val, y_val)
        model = Pipeline([
            ('scaler', scaler),
            ('clf', calibrated_clf)
        ])
        log_model_to_mlflow(model, "Logistic_Regression", logreg_params, X_train_norm, y_train, X_test_norm, y_test)
        loaded_models.append(['Logistic_Regression', model])

    # KNN
    try:
        model = load_model_by_name(experiment_name=experiment_name, run_name='KNN')[0]
        loaded_models.append(['KNN', model])
    except ValueError:
        knn_study = optuna.create_study(direction='maximize')
        knn_study.optimize(knn_objective, n_trials=num_trials)
        knn_params = knn_study.best_params
        base_clf = KNeighborsClassifier(**knn_params).fit(X_train_norm, y_train)
        calibrated_clf = CalibratedClassifierCV(FrozenEstimator(base_clf), method=cal_method, cv=num_cv_folds).fit(X_val, y_val)
        model = Pipeline([
            ('scaler', scaler),
            ('clf', calibrated_clf)
        ])
        log_model_to_mlflow(model, "KNN", knn_params, X_train_norm, y_train, X_test_norm, y_test)
        loaded_models.append(['KNN', model])

    # SVM Linear
    try:
        model = load_model_by_name(experiment_name=experiment_name, run_name='SVM_Linear')[0]
        loaded_models.append(['SVM_Linear', model])
    except ValueError:
        svm_linear_study = optuna.create_study(direction='maximize')
        svm_linear_study.optimize(svm_linear_objective, n_trials=num_trials)
        svm_linear_params = svm_linear_study.best_params
        base_clf = SVC(kernel='linear', **svm_linear_params, random_state=CONFIG['SEED'], probability=True).fit(X_train_norm, y_train)
        calibrated_clf = CalibratedClassifierCV(FrozenEstimator(base_clf), method=cal_method, cv=num_cv_folds).fit(X_val, y_val)
        model = Pipeline([
            ('scaler', scaler),
            ('clf', calibrated_clf)
        ])
        log_model_to_mlflow(model, "SVM_Linear", svm_linear_params, X_train_norm, y_train, X_test_norm, y_test)
        loaded_models.append(['SVM_Linear', model])

    # SVM Polynomial
    try:
        model = load_model_by_name(experiment_name=experiment_name, run_name='SVM_Polynomial')[0]
        loaded_models.append(['SVM_Polynomial', model])
    except ValueError:
        svm_poly_study = optuna.create_study(direction='maximize')
        svm_poly_study.optimize(svm_poly_objective, n_trials=num_trials)
        svm_poly_params = svm_poly_study.best_params
        base_clf = SVC(kernel='poly', **svm_poly_params, random_state=CONFIG['SEED'], probability=True).fit(X_train_norm, y_train)
        calibrated_clf = CalibratedClassifierCV(FrozenEstimator(base_clf), method=cal_method, cv=num_cv_folds).fit(X_val, y_val)
        model = Pipeline([
            ('scaler', scaler),
            ('clf', calibrated_clf)
        ])
        log_model_to_mlflow(model, "SVM_Polynomial", svm_poly_params, X_train_norm, y_train, X_test_norm, y_test)
        loaded_models.append(['SVM_Polynomial', model])

    # SVM RBF
    try:
        model = load_model_by_name(experiment_name=experiment_name, run_name='SVM_RBF')[0]
        loaded_models.append(['SVM_RBF', model])
    except ValueError:
        svm_rbf_study = optuna.create_study(direction='maximize')
        svm_rbf_study.optimize(svm_rbf_objective, n_trials=num_trials)
        svm_rbf_params = svm_rbf_study.best_params
        base_clf = SVC(kernel='rbf', **svm_rbf_params, random_state=CONFIG['SEED'], probability=True).fit(X_train_norm, y_train)
        calibrated_clf = CalibratedClassifierCV(FrozenEstimator(base_clf), method=cal_method, cv=num_cv_folds).fit(X_val, y_val)
        model = Pipeline([
            ('scaler', scaler),
            ('clf', calibrated_clf)
        ])
        log_model_to_mlflow(model, "SVM_RBF", svm_rbf_params, X_train_norm, y_train, X_test_norm, y_test)
        loaded_models.append(['SVM_RBF', model])

    # MLP
    try:
        model = load_model_by_name(experiment_name=experiment_name, run_name='MLP')[0]
        loaded_models.append(['MLP', model])
    except ValueError:
        mlp_study = optuna.create_study(direction='maximize')
        mlp_study.optimize(mlp_objective, n_trials=num_trials)
        mlp_params = mlp_study.best_params
        mlp_params['hidden_layer_sizes'] = (mlp_params['hidden_layer_sizes'],)
        base_clf = MLPClassifier(**mlp_params, random_state=CONFIG['SEED']).fit(X_train_norm, y_train)
        calibrated_clf = CalibratedClassifierCV(FrozenEstimator(base_clf), method=cal_method, cv=num_cv_folds).fit(X_val, y_val)
        model = Pipeline([
            ('scaler', scaler),
            ('clf', calibrated_clf)
        ])
        log_model_to_mlflow(model, "MLP", mlp_params, X_train_norm, y_train, X_test_norm, y_test)
        loaded_models.append(['MLP', model])

    # Random Forest
    try:
        model = load_model_by_name(experiment_name=experiment_name, run_name='Random_Forest')[0]
        loaded_models.append(['Random_Forest', model])
    except ValueError:
        rf_study = optuna.create_study(direction='maximize')
        rf_study.optimize(rf_objective, n_trials=num_trials)
        rf_params = rf_study.best_params
        base_model = RandomForestClassifier(**rf_params, random_state=CONFIG['SEED'], n_jobs=-1).fit(X_train, y_train)
        model = CalibratedClassifierCV(FrozenEstimator(base_model), method=cal_method, cv=num_cv_folds).fit(X_val, y_val)
        log_model_to_mlflow(
            model, "Random_Forest", rf_params,
            X_train, y_train, X_test, y_test
        )
        loaded_models.append(['Random_Forest', model])

    # Gradient Boosting
    try:
        model = load_model_by_name(experiment_name=experiment_name, run_name='Gradient_Boosting')[0]
        loaded_models.append(['Gradient_Boosting', model])
    except ValueError:
        gb_study = optuna.create_study(direction='maximize')
        gb_study.optimize(gb_objective, n_trials=num_trials)
        gb_params = gb_study.best_params
        base_model = GradientBoostingClassifier(**gb_params, random_state=CONFIG['SEED']).fit(X_train, y_train)
        model = CalibratedClassifierCV(FrozenEstimator(base_model), method=cal_method, cv=num_cv_folds).fit(X_val, y_val)
        log_model_to_mlflow(
            model, "Gradient_Boosting", gb_params,
            X_train, y_train, X_test, y_test
        )
        loaded_models.append(['Gradient_Boosting', model])

    # AdaBoost
    try:
        model = load_model_by_name(experiment_name=experiment_name, run_name='AdaBoost')[0]
        loaded_models.append(['AdaBoost', model])
    except ValueError:
        ada_study = optuna.create_study(direction='maximize')
        ada_study.optimize(ada_objective, n_trials=num_trials)
        ada_params = ada_study.best_params
        base_model = AdaBoostClassifier(**ada_params, random_state=CONFIG['SEED']).fit(X_train, y_train)
        model = CalibratedClassifierCV(FrozenEstimator(base_model), method=cal_method, cv=num_cv_folds).fit(X_val, y_val)
        log_model_to_mlflow(
            model, "AdaBoost", ada_params,
            X_train, y_train, X_test, y_test
        )
        loaded_models.append(['AdaBoost', model])

    # XGBoost
    try:
        model = load_model_by_name(experiment_name=experiment_name, run_name='XGBoost')[0]
        loaded_models.append(['XGBoost', model])
    except ValueError:
        xgb_study = optuna.create_study(direction='maximize')
        xgb_study.optimize(xgb_objective, n_trials=num_trials)
        xgb_params = xgb_study.best_params
        base_model = XGBClassifier(**xgb_params, random_state=CONFIG['SEED'], n_jobs=-1, eval_metric='logloss', enable_categorical=True).fit(X_train, y_train)
        model = CalibratedClassifierCV(FrozenEstimator(base_model), method=cal_method, cv=num_cv_folds).fit(X_val, y_val)
        log_model_to_mlflow(
            model, "XGBoost", xgb_params,
            X_train, y_train, X_test, y_test
        )
        loaded_models.append(['XGBoost', model])

    # METADES

    pool_classifiers = [x[1] for x in loaded_models]

    #def mdes_objective(trial):
    #    k = trial.suggest_int('k', 3, 20)
    #    kp = trial.suggest_int('kp', 3, 20)
    #    selection_threshold = trial.suggest_int('selection_threshold', 0.5, 0.8)
    #    knn_metric = trial.suggest_categorical('knn_metric', ['minkowski', 'cosine', 'mahalanobis'])
    #    
    #    clf = METADES(pool_classifiers, k=k, Kp=kp, selection_threshold=selection_threshold, 
    #                  knn_metric=knn_metric, random_state=CONFIG['SEED'])
    #    
    #    score = cross_val_score(clf, X_val, y_val, scoring=scorer_string, n_jobs=-1, cv=num_cv_folds)
    #    return score.mean()

    try:
        metades = load_model_by_name(experiment_name=experiment_name, run_name='METADES')[0]
    except ValueError:
        # Juro que tentei fazer HPO no META-DES mas essa DESLib é QUEBRADA e eu não aguento mais tentar fazer isso funcionar
        #mdes_study = optuna.create_study(direction='maximize')
        #mdes_study.optimize(mdes_objective, n_trials=num_trials)
        #mdes_params = mdes_study.best_params
        metades = METADES(pool_classifiers, random_state=CONFIG['SEED']).fit(X_val, y_val)
        log_model_to_mlflow(
            metades, "METADES", metades.get_params(),
            X_val, y_val, X_test, y_test
        )

    if(load):
        return dict(loaded_models), metades

def getExpName(dataset):
    global CONFIG
    dataset = re.sub('[-_ ]', '', dataset).lower()
    return f"{dataset}_{CONFIG['VERSION']}_{CONFIG['SEED']}"

if(__name__=='__main__'):
    NUM_TRIALS = 25
    #DATASET = 'circles'
    for DATASET in ['covid','airbnb','heloc','churn','covertype']: # 'twomoons',
        experiment_name = getExpName(DATASET)

        searchAndTrain(dataset=DATASET, 
                    experiment_name=experiment_name, 
                    num_trials=NUM_TRIALS)