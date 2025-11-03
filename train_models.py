import json
import pandas as pd
import numpy as np
import re
import mlflow
import mlflow.sklearn
import optuna
#import logging
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
        # Treinar modelo
        model.fit(X_train, y_train)
        
        # Fazer predições
        y_pred = model.predict(X_test)
        
        # Calcular probabilidades para AUROC (se disponível)
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                # Para classificação binária
                if y_proba.shape[1] == 2:
                    auroc = roc_auc_score(y_test, y_proba[:, 1])
                # Para classificação multiclasse
                else:
                    auroc = roc_auc_score(y_test, y_proba)
            elif hasattr(model, 'decision_function'):
                y_scores = model.decision_function(X_test)
                if len(np.unique(y_test)) == 2:
                    auroc = roc_auc_score(y_test, y_scores)
                else:
                    auroc = roc_auc_score(y_test, y_scores)
            else:
                auroc = None
        except Exception as e:
            #print(f"Aviso: Não foi possível calcular AUROC para {model_name}: {e}")
            auroc = None
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        if(len(set(y_test)) > 2):
            precision = precision_score(y_test, y_pred, zero_division=0, average='macro')
            recall = recall_score(y_test, y_pred, zero_division=0, average='macro')
            f1 = f1_score(y_test, y_pred, zero_division=0, average='macro')
        else:
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Registrar hiperparâmetros
        mlflow.log_params(hyperparams)
        
        # Registrar métricas
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        if auroc is not None:
            mlflow.log_metric("auroc", auroc)
        
        # Registrar modelo
        mlflow.sklearn.log_model(model, name=model_name)
        
        #print(f"\n{'='*60}")
        #print(f"Modelo: {model_name}")
        #print(f"{'='*60}")
        #print(f"Acurácia:  {accuracy:.4f}")
        #print(f"Precisão:  {precision:.4f}")
        #print(f"Recall:    {recall:.4f}")
        #print(f"F1-Score:  {f1:.4f}")
        #if auroc is not None:
            #print(f"AUROC:     {auroc:.4f}")
        #print(f"{'='*60}\n")
        
        return model
    
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
        'run_id': run_id,
        'run_name': run_name,
        'start_time': run.start_time,
        'params': {k.replace('params.', ''): v for k, v in run.items() if k.startswith('params.')},
        'metrics': {k.replace('metrics.', ''): v for k, v in run.items() if k.startswith('metrics.')}
    }
    
    #print(f"✓ Modelo '{run_name}' carregado com sucesso!")
    #print(f"  Run ID: {run_id}")
    ##print(f"  Métricas: F1={run_info['metrics'].get('f1_score', 'N/A'):.4f}")
    
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
        scaler = StandardScaler()
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
            
        #df = pd.DataFrame(dict(x0=X[:,0], x1=X[:,1], label=y))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=CONFIG['SEED'])
        X_train_norm = scaler.fit_transform(X_train)
        X_test_norm = scaler.transform(X_test)

    elif(dataset=='covid'):
        df_covid = pd.read_csv('data/hosp1_v8 (1).csv') # 526 exemplos
        X_train, y_train = df_covid.drop(columns=['severity']), df_covid['severity']

        df_covid = pd.read_csv('data/hospital2 (2).csv').drop(columns=['creatino.fosfoquinase.cpk.plasma.ck',
                                                                        'troponina.i.plasma.troponina.i']) # 134 exemplos
        X_test, y_test = df_covid.drop(columns=['severity']), df_covid['severity']

        scaler = StandardScaler()
        nmrc_cols = X_train.columns[1:]
        X_train_norm = X_train.copy()
        X_test_norm = X_test.copy()
        X_train_norm.loc[:,nmrc_cols] = scaler.fit_transform(X_train_norm[nmrc_cols])
        X_test_norm.loc[:,nmrc_cols] = scaler.transform(X_test_norm[nmrc_cols])

        return X_train, X_train_norm, X_test, X_test_norm, y_train, y_test

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
                #else:
                #    bathrooms.at[i,'bath_type'] = re.sub(' bath', '', bt)

        df['bathrooms'] = bathrooms['n_baths'].astype(float)
        df['bathroom_type'] = bathrooms['bath_type']

        df = df[[
            'host_response_time', #ok
            'host_response_rate', #ok
            'host_is_superhost', #ok
            'host_total_listings_count', #ok
            'host_identity_verified', #ok
            'latitude', #ok
            'longitude', #ok
            'property_type',
            'room_type', #ok
            'accommodates', #ok
            'bathrooms', #ok (o atualizado, vindo de bathrooms_text)
            'bathroom_type', #ok
            'bedrooms', #ok
            'beds', #ok
            'number_of_reviews', #ok
            #'number_of_reviews_l30d', #ok
            'review_scores_rating', #ok
            'review_scores_checkin', #ok
            'review_scores_communication', #ok
            'review_scores_location', #ok
            'minimum_nights',#ok (como o preço é apenas no momento, então vou deixar as noites apenas do momento também)
            'maximum_nights',#ok (como o preço é apenas no momento, então vou deixar as noites apenas do momento também)
            #'has_availability',#ok
            'availability_30',#ok
            #'availability_60',#ok
            #'availability_90',#ok
            #'availability_365',#ok
            'price'
        ]].dropna()

        df['host_response_rate'] = df['host_response_rate'].str.replace('%', '').astype(float)
        df['host_response_time'] = df['host_response_time'].astype('category').cat.reorder_categories(['within an hour', 'within a few hours', 'within a day', 'a few days or more']).cat.codes
        df[['host_is_superhost','host_identity_verified']] = df[['host_is_superhost','host_identity_verified']].map(lambda x: x=='t')
        df['property_type'] = df['property_type'].apply(transform_property)
        df['price'] = df['price'].str.replace('[,\$]','', regex=True).astype(float)>300
        
        X, y = df.drop(columns=['price']), df['price'].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=CONFIG['SEED'])

        onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        scaler = StandardScaler() 
        onehot = onehot.set_output(transform='pandas')
        X_train = pd.concat([X_train.drop(columns=['property_type','room_type','bathroom_type']), onehot.fit_transform(X_train[['property_type','room_type','bathroom_type']], y_train)], axis=1)
        X_test = pd.concat([X_test.drop(columns=['property_type','room_type','bathroom_type']), onehot.transform(X_test[['property_type','room_type','bathroom_type']])], axis=1)

        X_train_norm = X_train.copy()
        X_test_norm = X_test.copy()

        nmrc_cols = ['host_response_time','host_response_rate','host_total_listings_count',
                    'latitude','longitude','accommodates','bathrooms','bedrooms','beds',
                    'number_of_reviews','review_scores_rating','review_scores_checkin',
                    'review_scores_communication','review_scores_location',
                    'minimum_nights','maximum_nights','availability_30']

        X_train_norm.loc[:,nmrc_cols] = scaler.fit_transform(X_train_norm[nmrc_cols])
        X_test_norm.loc[:,nmrc_cols] = scaler.transform(X_test_norm[nmrc_cols])

    elif(dataset=='heloc'):
        df = pd.read_csv('data/heloc_dataset_v1 (1).csv')

        X, y = df.drop(columns=['RiskPerformance']), df['RiskPerformance'].replace({'Bad':0, 'Good':1}).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=CONFIG['SEED'], shuffle=True)

        scaler = StandardScaler() 
        X_train_norm = scaler.fit_transform(X_train.copy())
        X_test_norm = scaler.transform(X_test.copy())

    #elif(dataset=='machinefailure'):
    #    df = df.drop(columns=['UDI','Product ID'])

    #    df['Label'] = df.apply(lambda x: np.nan if x[['TWF','HDF','PWF','OSF']].sum()>1 else 0 if x[['TWF','HDF','PWF','OSF']].sum()==0 else x[['TWF','HDF','PWF','OSF']].values.argmax()+1, axis=1)#.astype(int)
    #    df = df.dropna()

    #    df['Type'] = df['Type'].replace({'L':0,'M':1,'H':2}).astype(int)

    #    X, y = df.drop(columns=['Label']), df['Label'].astype(int)

    #    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=CONFIG['SEED'], shuffle=True)

    #    scaler = StandardScaler()
    #    X_train_norm = scaler.fit_transform(X_train.copy())
    #    X_test_norm = scaler.transform(X_test.copy())

    elif(dataset=='covertype'):
        df = pd.read_csv('data/covertype.csv')
        # Carregar dataset
        target = 'Cover_Type'

        # Amostragem estratificada de 10%
        df_sample, _ = train_test_split(
            df, 
            test_size=0.9,
            stratify=df[target],
            random_state=CONFIG['SEED'],
            shuffle=True
        )

        # Separar features e target
        X = df_sample.drop(columns=[target])
        y = df_sample[target].astype(int)-1

        # Dividir em treino (70%) e teste (30%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=CONFIG['SEED'], shuffle=True)
 
        X_train_norm = X_train.copy()
        X_test_norm = X_test.copy()
        nmrc_cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                    'Horizontal_Distance_To_Fire_Points']
        
        scaler = StandardScaler()
        X_train_norm.loc[:,nmrc_cols] = scaler.fit_transform(X_train_norm[nmrc_cols])
        X_test_norm.loc[:,nmrc_cols] = scaler.transform(X_test_norm[nmrc_cols])

    elif(dataset=='churn'):
        df = pd.read_csv(f'data/customer_churn_telecom_services.csv', header=0)

        # Quantiades de cada valor único por coluna
        continuous_cols = []
        cat_cols = []

        for col in df.drop(columns=['Churn']).columns:
            unique_values = df[col].value_counts()
            if(len(unique_values) <= 4):
                cat_cols.append(col)
            else:
                continuous_cols.append(col)

        df['TotalCharges'] = df['TotalCharges'].fillna(0)

        # Alterando colunas categóricas binárias para int

        rdict = {'gender': {'Male': 0, 'Female': 1},
                'Partner': {'No': 0, 'Yes': 1},
                'Dependents': {'No': 0, 'Yes': 1},
                'PhoneService': {'No': 0, 'Yes': 1},
                'PaperlessBilling': {'No': 0, 'Yes': 1},
                'Churn': {'No': 0, 'Yes': 1},
                }

        # Alterando colunas que são parcialmente dummy
        # Exp.: OnlineSecurity: ("No internet service", "No", "Yes") -> (0, 1, 2)
        rdict['MultipleLines'] = {'No phone service': 0, 'No': 1, 'Yes': 2}

        cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies']

        for col in cols:
            rdict[col] = {'No internet service': 0, 'No': 1, 'Yes': 2}

        # Alterando colunas não-dummy

        rdict['InternetService'] = {'No': 0, 'DSL': 1, 'Fiber optic': 2}
        rdict['Contract'] = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        rdict['PaymentMethod'] = {'Credit card (automatic)': 0, 'Bank transfer (automatic)': 1,
                                'Mailed check': 2, 'Electronic check': 3}

        df = df.replace(rdict)
        df = df.rename(columns={'Churn': 'target'}) 
        cols = df.drop(columns=['target']).columns

        temp = df[df.target==1]
        train_pos, test_pos = train_test_split(temp, test_size=0.3, shuffle=True, random_state=CONFIG['SEED'])

        temp = df[df.target==0]
        train_neg, test_neg = train_test_split(temp, test_size=0.3, shuffle=True, random_state=CONFIG['SEED'])


        X_train = pd.concat([train_pos[cols], train_neg[cols]], ignore_index=True)
        y_train = pd.concat([train_pos['target'], train_neg['target']], ignore_index=True)

        X_test = pd.concat([test_pos[cols], test_neg[cols]], ignore_index=True)
        y_test = pd.concat([test_pos['target'], test_neg['target']], ignore_index=True)

        # Normalização baseada no conjunto de treinamento
        scaler1 = StandardScaler()

        X_train_norm = X_train.copy()
        X_train_norm.loc[:,continuous_cols] = scaler1.fit_transform(X_train_norm.loc[:,continuous_cols], y_train)

        # Normalização nos conjuntos de validação e teste, com base nos dados de treinamento
        X_test_norm = X_test.copy()
        X_test_norm.loc[:,continuous_cols] = scaler1.transform(X_test_norm.loc[:,continuous_cols])

        # Balanceamento no conjunto de treinamento
        o_sampler = RandomOverSampler(random_state=CONFIG['SEED'])

        X_train_norm, _ = o_sampler.fit_resample(X_train_norm, y_train) #yb_train == yb_train_norm
        X_train, y_train = o_sampler.fit_resample(X_train, y_train)

    else:
        raise ValueError('Dataset Not Usable')
    
    return X_train, X_train_norm, X_test, X_test_norm, y_train, y_test

# ======================================================

def searchAndTrain(dataset, experiment_name, num_trials, load=False):

    mlflow.set_experiment(experiment_name=experiment_name)

    X_train, X_train_norm, X_test, X_test_norm, y_train, y_test = get_data(dataset)

    scorer_string = 'f1_macro' if len(set(y_train))>2 else 'f1'

    # 1. Define an objective function to be maximized.
    def dtree_objective(trial:optuna.trial._trial.Trial):
        
        # 2. Suggest values for the hyperparameters using a trial object.
        max_depth = trial.suggest_int('max_depth', 5, 100, log=True)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
        min_samples_split = trial.suggest_int('min_samples_split', 2, 60)
        min_samples_leaf = trial.suggest_int('min_samples_leaf',1, 30)
        
        clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf).fit(X_train, y_train)
        score = cross_val_score(clf, X_train, y_train, scoring=scorer_string, n_jobs=-1, cv=10).mean()
        
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
        ).fit(X_train_norm, y_train)
        
        score = cross_val_score(clf, X_train_norm, y_train, scoring=scorer_string, n_jobs=-1, cv=10)
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
        
        clf = LogisticRegression(**params).fit(X_train_norm, y_train)
        score = cross_val_score(clf, X_train_norm, y_train, scoring=scorer_string, n_jobs=-1, cv=10)
        return score.mean()

    def knn_objective(trial):
        n_neighbors = trial.suggest_int('n_neighbors', 1, 50)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])
        p = trial.suggest_int('p', 1, 5) if metric == 'minkowski' else 2
        
        clf = KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights, 
            metric=metric, p=p
        ).fit(X_train_norm, y_train)
        
        score = cross_val_score(clf, X_train_norm, y_train, scoring=scorer_string, n_jobs=-1, cv=10)
        return score.mean()

    def svm_linear_objective(trial):
        C = trial.suggest_float('C', 1e-3, 100, log=True)
        max_iter = trial.suggest_int('max_iter', 500, 3000)
        
        clf = SVC(
            kernel='linear', C=C, max_iter=max_iter, random_state=CONFIG['SEED']
        ).fit(X_train_norm, y_train)
        
        score = cross_val_score(clf, X_train_norm, y_train, scoring=scorer_string, n_jobs=-1, cv=10)
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
        ).fit(X_train_norm, y_train)
        
        score = cross_val_score(clf, X_train_norm, y_train, scoring=scorer_string, n_jobs=-1, cv=10)
        return score.mean()

    def svm_rbf_objective(trial):
        C = trial.suggest_float('C', 1e-3, 100, log=True)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        max_iter = trial.suggest_int('max_iter', 500, 3000)
        
        clf = SVC(
            kernel='rbf', C=C, gamma=gamma, max_iter=max_iter, random_state=CONFIG['SEED']
        ).fit(X_train_norm, y_train)
        
        score = cross_val_score(clf, X_train_norm, y_train, scoring=scorer_string, n_jobs=-1, cv=10)
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
        ).fit(X_train_norm, y_train)
        
        score = cross_val_score(clf, X_train_norm, y_train, scoring=scorer_string, n_jobs=-1, cv=10)
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
        ).fit(X_train, y_train)
        
        score = cross_val_score(clf, X_train, y_train, scoring=scorer_string, n_jobs=-1, cv=10)
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
        ).fit(X_train, y_train)
        
        score = cross_val_score(clf, X_train, y_train, scoring=scorer_string, n_jobs=-1, cv=10)
        return score.mean()

    def ada_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 500, log=True)
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 2, log=True)
        #algorithm = trial.suggest_categorical('algorithm', ['SAMME', 'SAMME.R'])
        
        clf = AdaBoostClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            #algorithm=algorithm, 
            random_state=CONFIG['SEED']
        ).fit(X_train, y_train)
        
        score = cross_val_score(clf, X_train, y_train, scoring=scorer_string, n_jobs=-1, cv=10)
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
        ).fit(X_train, y_train)
        
        score = cross_val_score(clf, X_train, y_train, scoring=scorer_string, n_jobs=1, cv=10)
        return score.mean()

    # ============================================

    # 3. Create a study object and optimize the objective function.
    try:
        _ = load_model_by_name(experiment_name=experiment_name, run_name='Decision_Tree')
    except ValueError:
        dtree_study = optuna.create_study(direction='maximize')
        dtree_study.optimize(dtree_objective, n_trials=num_trials)
        #print(dtree_study.best_trial)

        dtree_params = dtree_study.best_params
        dtree_model = DecisionTreeClassifier(**dtree_params, random_state=CONFIG['SEED'])
        dtree_model = log_model_to_mlflow(
            dtree_model, "Decision_Tree", dtree_params, 
            X_train, y_train, X_test, y_test
        )

    try:
        _ = load_model_by_name(experiment_name=experiment_name, run_name='SGD')
    except ValueError:
        sgd_study = optuna.create_study(direction='maximize')
        sgd_study.optimize(sgd_objective, n_trials=num_trials)
        #print("SGD Best Trial:", sgd_study.best_trial)
        sgd_params = sgd_study.best_params
        sgd_model = SGDClassifier(**sgd_params, random_state=CONFIG['SEED'])
        sgd_model = log_model_to_mlflow(
            sgd_model, "SGD", sgd_params,
            X_train_norm, y_train, X_test_norm, y_test
        )

    try:
        _ = load_model_by_name(experiment_name=experiment_name, run_name='Logistic_Regression')
    except ValueError:
        logreg_study = optuna.create_study(direction='maximize')
        logreg_study.optimize(logreg_objective, n_trials=num_trials)
        #print("Logistic Regression Best Trial:", logreg_study.best_trial)
        logreg_params = logreg_study.best_params
        logreg_model = LogisticRegression(**logreg_params, random_state=CONFIG['SEED'])
        logreg_model = log_model_to_mlflow(
            logreg_model, "Logistic_Regression", logreg_params,
            X_train, y_train, X_test_norm, y_test
        )

    try:
        _ = load_model_by_name(experiment_name=experiment_name, run_name='KNN')
    except ValueError:
        knn_study = optuna.create_study(direction='maximize')
        knn_study.optimize(knn_objective, n_trials=num_trials)
        #print("KNN Best Trial:", knn_study.best_trial)
        knn_params = knn_study.best_params
        knn_model = KNeighborsClassifier(**knn_params)
        knn_model = log_model_to_mlflow(
            knn_model, "KNN", knn_params,
            X_train_norm, y_train, X_test_norm, y_test
        )

    try:
        _ = load_model_by_name(experiment_name=experiment_name, run_name='SVM_Linear')
    except ValueError:
        svm_linear_study = optuna.create_study(direction='maximize')
        svm_linear_study.optimize(svm_linear_objective, n_trials=num_trials)
        #print("SVM Linear Best Trial:", svm_linear_study.best_trial)
        svm_linear_params = svm_linear_study.best_params
        svm_linear_model = SVC(kernel='linear', **svm_linear_params, random_state=CONFIG['SEED'], probability=True)
        svm_linear_model = log_model_to_mlflow(
            svm_linear_model, "SVM_Linear", svm_linear_params,
            X_train, y_train, X_test_norm, y_test
        )

    try:
        _ = load_model_by_name(experiment_name=experiment_name, run_name='SVM_Polynomial')
    except ValueError:
        svm_poly_study = optuna.create_study(direction='maximize')
        svm_poly_study.optimize(svm_poly_objective, n_trials=num_trials)
        #print("SVM Poly Best Trial:", svm_poly_study.best_trial)
        svm_poly_params = svm_poly_study.best_params
        svm_poly_model = SVC(kernel='poly', **svm_poly_params, random_state=CONFIG['SEED'], probability=True)
        svm_poly_model = log_model_to_mlflow(
            svm_poly_model, "SVM_Polynomial", svm_poly_params,
            X_train_norm, y_train, X_test_norm, y_test
        )

    try:
        _ = load_model_by_name(experiment_name=experiment_name, run_name='SVM_RBF')
    except ValueError:
        svm_rbf_study = optuna.create_study(direction='maximize')
        svm_rbf_study.optimize(svm_rbf_objective, n_trials=num_trials)
        #print("SVM RBF Best Trial:", svm_rbf_study.best_trial)
        svm_rbf_params = svm_rbf_study.best_params
        svm_rbf_model = SVC(kernel='rbf', **svm_rbf_params, random_state=CONFIG['SEED'], probability=True)
        svm_rbf_model = log_model_to_mlflow(
            svm_rbf_model, "SVM_RBF", svm_rbf_params,
            X_train_norm, y_train, X_test_norm, y_test
        )

    try:
        _ = load_model_by_name(experiment_name=experiment_name, run_name='MLP')
    except ValueError:
        mlp_study = optuna.create_study(direction='maximize')
        mlp_study.optimize(mlp_objective, n_trials=num_trials)
        #print("MLP Best Trial:", mlp_study.best_trial)
        mlp_params = mlp_study.best_params
        mlp_params['hidden_layer_sizes'] = (mlp_params['hidden_layer_sizes'],)
        mlp_model = MLPClassifier(**mlp_params, random_state=CONFIG['SEED'])
        mlp_model = log_model_to_mlflow(
            mlp_model, "MLP", mlp_params,
            X_train_norm, y_train, X_test_norm, y_test
        )

    try:
        _ = load_model_by_name(experiment_name=experiment_name, run_name='Random_Forest')
    except ValueError:
        rf_study = optuna.create_study(direction='maximize')
        rf_study.optimize(rf_objective, n_trials=num_trials)
        #print("Random Forest Best Trial:", rf_study.best_trial)
        rf_params = rf_study.best_params
        rf_model = RandomForestClassifier(**rf_params, random_state=CONFIG['SEED'], n_jobs=-1)
        rf_model = log_model_to_mlflow(
            rf_model, "Random_Forest", rf_params,
            X_train, y_train, X_test, y_test
        )

    try:
        _ = load_model_by_name(experiment_name=experiment_name, run_name='Gradient_Boosting')
    except ValueError:
        gb_study = optuna.create_study(direction='maximize')
        gb_study.optimize(gb_objective, n_trials=num_trials)
        #print("Gradient Boosting Best Trial:", gb_study.best_trial)
        gb_params = gb_study.best_params
        gb_model = GradientBoostingClassifier(**gb_params, random_state=CONFIG['SEED'])
        gb_model = log_model_to_mlflow(
            gb_model, "Gradient_Boosting", gb_params,
            X_train, y_train, X_test, y_test
        )

    try:
        _ = load_model_by_name(experiment_name=experiment_name, run_name='AdaBoost')
    except ValueError:
        ada_study = optuna.create_study(direction='maximize')
        ada_study.optimize(ada_objective, n_trials=num_trials)
        #print("AdaBoost Best Trial:", ada_study.best_trial)
        ada_params = ada_study.best_params
        ada_model = AdaBoostClassifier(**ada_params, random_state=CONFIG['SEED'])
        ada_model = log_model_to_mlflow(
            ada_model, "AdaBoost", ada_params,
            X_train, y_train, X_test, y_test
        )

    try:
        _ = load_model_by_name(experiment_name=experiment_name, run_name='XGBoost')
    except ValueError:
        xgb_study = optuna.create_study(direction='maximize')
        xgb_study.optimize(xgb_objective, n_trials=num_trials)
        #print("XGBoost Best Trial:", xgb_study.best_trial)
        xgb_params = xgb_study.best_params
        xgb_model = XGBClassifier(**xgb_params, random_state=CONFIG['SEED'], n_jobs=-1, eval_metric='logloss', enable_categorical=True)
        xgb_model = log_model_to_mlflow(
            xgb_model, "XGBoost", xgb_params,
            X_train, y_train, X_test, y_test
        )

    if(load):
        return {'Decision_Tree': dtree_model,
                'SGD': sgd_model,
                'Logistic_Regression': logreg_model,
                'KNN':knn_model,
                'SVM_Linear':svm_linear_model,
                'SVM_Polynomial':svm_poly_model,
                'SVM_RBF':svm_rbf_model,
                'MLP':mlp_model,
                'Random_Forest':rf_model,
                'Grandient_Boosting':gb_model,
                'AdaBoost':ada_model,
                'XGBoost':xgb_model}

def getExpName(dataset):
    global CONFIG
    dataset = re.sub('[-_ ]', '', dataset).lower()
    return f"{dataset}_{CONFIG['VERSION']}_{CONFIG['SEED']}"

if(__name__=='__main__'):
    NUM_TRIALS = 20
    #DATASET = 'circles'
    for DATASET in ['covertype']:
        experiment_name = getExpName(DATASET)

        searchAndTrain(dataset=DATASET, 
                    experiment_name=experiment_name, 
                    num_trials=NUM_TRIALS)