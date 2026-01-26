import json
import pandas as pd
import numpy as np
import re
import mlflow
import mlflow.sklearn
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from classes import SpecificScaler
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from deslib.des import METADES
from classes import METADESR

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')

# =================================================

with open('config.json', 'r') as f:
    CONFIG = json.load(f)

# ==================================================

def auroc_score(model, x, y):
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(x)
            if y_proba.shape[1] == 2:
                auroc = roc_auc_score(y, y_proba[:, 1])
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
    """
    if(model_name=='METADESR'):
        model.set_predict_params(rejection_rate=0.0, selection_threshold=0.0)
    
    with mlflow.start_run(run_name=model_name):
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_auroc = auroc_score(model, X_train, y_train)
        test_auroc = auroc_score(model, X_test, y_test)
        
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
        
        mlflow.log_params(hyperparams)
        
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
        
        mlflow.sklearn.log_model(model, name=model_name)
    
def load_model_by_name(experiment_name, run_name):
    """
    Carrega um modelo específico do MLflow pelo nome da run.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experimento '{experiment_name}' não encontrado!")
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'"
    )
    
    if runs.empty:
        raise ValueError(f"Run '{run_name}' não encontrada no experimento '{experiment_name}'!")
    
    run = runs.iloc[0]
    run_id = run.run_id
    print(run_id)
    
    model_uri = f"runs:/{run_id}/{run_name}"
    model = mlflow.sklearn.load_model(model_uri)
    
    run_info = {
        'params': {k.replace('params.', ''): v for k, v in run.items() if k.startswith('params.')},
        'metrics': {k.replace('metrics.', ''): v for k, v in run.items() if k.startswith('metrics.')}
    }
    
    return model, run_info

# ======================================================

from sklearn.preprocessing import OneHotEncoder, StandardScaler

def transform_property(x):
    x = re.sub('(Private room( in )?)|(Shared room( in )?)|(Entire )|(Room in )', '', x).lower()
    if(x=='casa particular'):
        x='home'
    
    if(x not in ['rental unit','home','condo','loft','serviced apartment']):
        x='other'

    return x

def get_data(dataset:str):
    """
    Retorna os 4 subconjuntos de dados:
    - T (25%): Pool de classificadores
    - T_lambda (25%): Meta-treinamento
    - DSEL (25%): Seleção dinâmica
    - G (25%): Teste
    """
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
            
        # Primeira divisão: 25% para T, 75% restante
        X_T, X_temp, y_T, y_temp = train_test_split(X, y, test_size=0.75, shuffle=True, random_state=CONFIG['SEED'])
        
        # Segunda divisão: 33.33% do restante para T_lambda (25% do total), 66.67% restante
        X_T_lambda, X_temp2, y_T_lambda, y_temp2 = train_test_split(X_temp, y_temp, test_size=0.6667, shuffle=True, random_state=CONFIG['SEED'])
        
        # Terceira divisão: 50% do restante para DSEL e 50% para G (cada um com 25% do total)
        X_DSEL, X_G, y_DSEL, y_G = train_test_split(X_temp2, y_temp2, test_size=0.5, shuffle=True, random_state=CONFIG['SEED'])
        
        scaler = SpecificScaler().fit(X_T)

    elif(dataset=='covid'):
        df_covid = pd.read_csv('data/hosp1_v8 (1).csv')
        X, y = df_covid.drop(columns=['severity']), df_covid['severity']
        
        # Primeira divisão: 25% para T, 75% restante
        X_T, X_temp, y_T, y_temp = train_test_split(X, y, test_size=0.75, shuffle=True, random_state=CONFIG['SEED'])
        
        # Segunda divisão: 33.33% do restante para T_lambda (25% do total), 66.67% restante
        X_T_lambda, X_temp2, y_T_lambda, y_temp2 = train_test_split(X_temp, y_temp, test_size=0.6667, shuffle=True, random_state=CONFIG['SEED'])
        
        # Terceira divisão: 50% do restante para DSEL e 50% para G
        X_DSEL, X_G, y_DSEL, y_G = train_test_split(X_temp2, y_temp2, test_size=0.5, shuffle=True, random_state=CONFIG['SEED'])

        nmrc_cols = X_T.columns[1:]
        scaler = SpecificScaler().fit(X_T, nmrc_cols)

        return X_T, X_T_lambda, X_DSEL, X_G, y_T, y_T_lambda, y_DSEL, y_G, scaler

    elif(dataset=='airbnb'):
        df = pd.read_csv('data/listings.csv')

        bathrooms = df['bathrooms_text'].str.extract('([0-9\.]+)?([- A-Za-z]+)')
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

        # Primeira divisão: 25% para T, 75% restante
        X_T, X_temp, y_T, y_temp = train_test_split(X, y, test_size=0.75, shuffle=True, random_state=CONFIG['SEED'])
        
        # Segunda divisão: 33.33% do restante para T_lambda (25% do total), 66.67% restante
        X_T_lambda, X_temp2, y_T_lambda, y_temp2 = train_test_split(X_temp, y_temp, test_size=0.6667, shuffle=True, random_state=CONFIG['SEED'])
        
        # Terceira divisão: 50% do restante para DSEL e 50% para G
        X_DSEL, X_G, y_DSEL, y_G = train_test_split(X_temp2, y_temp2, test_size=0.5, shuffle=True, random_state=CONFIG['SEED'])

        onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        onehot = onehot.set_output(transform='pandas')
        
        X_T = pd.concat([X_T.drop(columns=['property_type','room_type','bathroom_type']), 
                         onehot.fit_transform(X_T[['property_type','room_type','bathroom_type']], y_T)], axis=1)
        X_T_lambda = pd.concat([X_T_lambda.drop(columns=['property_type','room_type','bathroom_type']), 
                                onehot.transform(X_T_lambda[['property_type','room_type','bathroom_type']])], axis=1)
        X_DSEL = pd.concat([X_DSEL.drop(columns=['property_type','room_type','bathroom_type']), 
                            onehot.transform(X_DSEL[['property_type','room_type','bathroom_type']])], axis=1)
        X_G = pd.concat([X_G.drop(columns=['property_type','room_type','bathroom_type']), 
                         onehot.transform(X_G[['property_type','room_type','bathroom_type']])], axis=1)

        nmrc_cols = ['host_response_time','host_response_rate','host_total_listings_count',
                    'latitude','longitude','accommodates','bathrooms','bedrooms','beds',
                    'number_of_reviews','review_scores_rating','review_scores_checkin',
                    'review_scores_communication','review_scores_location',
                    'minimum_nights','maximum_nights','availability_30']

        scaler = SpecificScaler().fit(X_T, nmrc_cols)

    elif(dataset=='heloc'):
        df = pd.read_csv('data/heloc_dataset_v1 (1).csv')

        X, y = df.drop(columns=['RiskPerformance']), df['RiskPerformance'].replace({'Bad':0, 'Good':1}).astype(int)

        # Primeira divisão: 25% para T, 75% restante
        X_T, X_temp, y_T, y_temp = train_test_split(X, y, test_size=0.75, random_state=CONFIG['SEED'], shuffle=True)
        
        # Segunda divisão: 33.33% do restante para T_lambda (25% do total), 66.67% restante
        X_T_lambda, X_temp2, y_T_lambda, y_temp2 = train_test_split(X_temp, y_temp, test_size=0.6667, random_state=CONFIG['SEED'], shuffle=True)
        
        # Terceira divisão: 50% do restante para DSEL e 50% para G
        X_DSEL, X_G, y_DSEL, y_G = train_test_split(X_temp2, y_temp2, test_size=0.5, random_state=CONFIG['SEED'], shuffle=True)

        scaler = StandardScaler().fit(X_T)

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

        # Primeira divisão: 25% para T, 75% restante
        X_T, X_temp, y_T, y_temp = train_test_split(X, y, test_size=0.75, stratify=y, random_state=CONFIG['SEED'], shuffle=True)
        
        # Segunda divisão: 33.33% do restante para T_lambda (25% do total), 66.67% restante
        X_T_lambda, X_temp2, y_T_lambda, y_temp2 = train_test_split(X_temp, y_temp, test_size=0.6667, stratify=y_temp, random_state=CONFIG['SEED'], shuffle=True)
        
        # Terceira divisão: 50% do restante para DSEL e 50% para G
        X_DSEL, X_G, y_DSEL, y_G = train_test_split(X_temp2, y_temp2, test_size=0.5, stratify=y_temp2, random_state=CONFIG['SEED'], shuffle=True)

        nmrc_cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                    'Horizontal_Distance_To_Fire_Points']
        
        scaler = StandardScaler().fit(X_T, nmrc_cols)

    elif(dataset=='churn'):
        df = pd.read_csv(f'data/customer_churn_telecom_services.csv', header=0)

        continuous_cols = []
        cat_cols = []

        cols = df.drop(columns=['Churn']).columns
        for col in cols:
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

        truple_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies']

        for col in truple_cols:
            rdict[col] = {'No internet service': 0, 'No': 1, 'Yes': 2}

        rdict['InternetService'] = {'No': 0, 'DSL': 1, 'Fiber optic': 2}
        rdict['Contract'] = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        rdict['PaymentMethod'] = {'Credit card (automatic)': 0, 'Bank transfer (automatic)': 1,
                                'Mailed check': 2, 'Electronic check': 3}

        df = df.replace(rdict)
        X = df[cols]
        y = df['Churn']

        # Primeira divisão: 25% para T, 75% restante
        X_T, X_temp, y_T, y_temp = train_test_split(X, y, test_size=0.75, stratify=y, random_state=CONFIG['SEED'], shuffle=True)
        
        # Segunda divisão: 33.33% do restante para T_lambda (25% do total), 66.67% restante
        X_T_lambda, X_temp2, y_T_lambda, y_temp2 = train_test_split(X_temp, y_temp, test_size=0.6667, stratify=y_temp, random_state=CONFIG['SEED'], shuffle=True)
        
        # Terceira divisão: 50% do restante para DSEL e 50% para G
        X_DSEL, X_G, y_DSEL, y_G = train_test_split(X_temp2, y_temp2, test_size=0.5, stratify=y_temp2, random_state=CONFIG['SEED'], shuffle=True)

        scaler = SpecificScaler().fit(X_T, continuous_cols)

        o_sampler = RandomOverSampler(random_state=CONFIG['SEED'])
        X_T, y_T = o_sampler.fit_resample(X_T, y_T)

    elif(dataset=='adult'):
        # UCI Adult Dataset - Predição de renda (>50K ou <=50K)
        df = pd.read_csv('data/adult.csv')
        
        # Remove espaços extras das colunas categóricas
        cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                    'relationship', 'race', 'sex', 'native-country']
        for col in cat_cols:
            df[col] = df[col].str.strip()
        
        # Remove linhas com valores faltantes (marcados como '?')
        df = df.replace('?', np.nan).dropna()
        
        # Target: target >50K (1) ou <=50K (0)
        y = (df['target'].str.strip() == '>50K').astype(int)
        X = df.drop(columns=['target'])
        
        # Label encoding para variáveis categóricas
        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
        
        # Primeira divisão: 25% para T, 75% restante
        X_T, X_temp, y_T, y_temp = train_test_split(X, y, test_size=0.75, stratify=y, 
                                                     random_state=CONFIG['SEED'], shuffle=True)
        
        # Segunda divisão: 33.33% do restante para T_lambda
        X_T_lambda, X_temp2, y_T_lambda, y_temp2 = train_test_split(X_temp, y_temp, test_size=0.6667, 
                                                                     stratify=y_temp, random_state=CONFIG['SEED'], shuffle=True)
        
        # Terceira divisão: 50% do restante para DSEL e 50% para G
        X_DSEL, X_G, y_DSEL, y_G = train_test_split(X_temp2, y_temp2, test_size=0.5, 
                                                     stratify=y_temp2, random_state=CONFIG['SEED'], shuffle=True)
        
        nmrc_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        scaler = SpecificScaler().fit(X_T, nmrc_cols)

    elif(dataset=='magic_gamma'):
        # UCI MAGIC Gamma Telescope - Classificação de raios gamma vs hadrons
        df = pd.read_csv('data/magic04.data', names=[
            'fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 
            'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class'
        ])
        
        # Target: gamma (1) vs hadron (0)
        y = (df['class'] == 'g').astype(int)
        X = df.drop(columns=['class'])
        
        # Primeira divisão: 25% para T, 75% restante
        X_T, X_temp, y_T, y_temp = train_test_split(X, y, test_size=0.75, stratify=y, 
                                                     random_state=CONFIG['SEED'], shuffle=True)
        
        # Segunda divisão: 33.33% do restante para T_lambda
        X_T_lambda, X_temp2, y_T_lambda, y_temp2 = train_test_split(X_temp, y_temp, test_size=0.6667, 
                                                                     stratify=y_temp, random_state=CONFIG['SEED'], shuffle=True)
        
        # Terceira divisão: 50% do restante para DSEL e 50% para G
        X_DSEL, X_G, y_DSEL, y_G = train_test_split(X_temp2, y_temp2, test_size=0.5, 
                                                     stratify=y_temp2, random_state=CONFIG['SEED'], shuffle=True)
        
        scaler = StandardScaler().fit(X_T)

    elif(dataset=='credit'):
        # UCI Default of Credit Card Clients
        df = pd.read_csv('data/default_credit_card.csv', header=1)
        
        # Target: default payment (1=yes, 0=no)
        y = df['default payment next month']
        X = df.drop(columns=['ID', 'default payment next month'])
        
        # Primeira divisão: 25% para T, 75% restante
        X_T, X_temp, y_T, y_temp = train_test_split(X, y, test_size=0.75, stratify=y, 
                                                     random_state=CONFIG['SEED'], shuffle=True)
        
        # Segunda divisão: 33.33% do restante para T_lambda
        X_T_lambda, X_temp2, y_T_lambda, y_temp2 = train_test_split(X_temp, y_temp, test_size=0.6667, 
                                                                     stratify=y_temp, random_state=CONFIG['SEED'], shuffle=True)
        
        # Terceira divisão: 50% do restante para DSEL e 50% para G
        X_DSEL, X_G, y_DSEL, y_G = train_test_split(X_temp2, y_temp2, test_size=0.5, 
                                                     stratify=y_temp2, random_state=CONFIG['SEED'], shuffle=True)
        
        scaler = StandardScaler().fit(X_T)
        
        # Oversampling para balancear classes
        o_sampler = RandomOverSampler(random_state=CONFIG['SEED'])
        X_T, y_T = o_sampler.fit_resample(X_T, y_T)

    elif(dataset=='marketing'):
        # UCI Bank Marketing Dataset
        df = pd.read_csv('data/bank-additional-full.csv', sep=';')
        
        # Target: subscribed to term deposit (yes=1, no=0)
        y = (df['y'] == 'yes').astype(int)
        X = df.drop(columns=['y'])
        
        # Codificação de variáveis categóricas
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        
        # Primeira divisão: 25% para T, 75% restante
        X_T, X_temp, y_T, y_temp = train_test_split(X, y, test_size=0.75, stratify=y, 
                                                     random_state=CONFIG['SEED'], shuffle=True)
        
        # Segunda divisão: 33.33% do restante para T_lambda
        X_T_lambda, X_temp2, y_T_lambda, y_temp2 = train_test_split(X_temp, y_temp, test_size=0.6667, 
                                                                     stratify=y_temp, random_state=CONFIG['SEED'], shuffle=True)
        
        # Terceira divisão: 50% do restante para DSEL e 50% para G
        X_DSEL, X_G, y_DSEL, y_G = train_test_split(X_temp2, y_temp2, test_size=0.5, 
                                                     stratify=y_temp2, random_state=CONFIG['SEED'], shuffle=True)
        
        # Escalar apenas colunas numéricas originais
        nmrc_cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 
                     'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
        scaler = SpecificScaler().fit(X_T, nmrc_cols)
        
        # Oversampling para balancear classes
        o_sampler = RandomOverSampler(random_state=CONFIG['SEED'])
        X_T, y_T = o_sampler.fit_resample(X_T, y_T)

    elif(dataset=='qsar'):
        # UCI QSAR Biodegradation Dataset
        df = pd.read_csv('data/biodeg.csv', sep=';')
        
        # Target: ready biodegradable (RB=1) vs not ready biodegradable (NRB=0)
        y = (df['experimental class'] == 'RB').astype(int)
        X = df.drop(columns=['experimental class'])
        
        # Primeira divisão: 25% para T, 75% restante
        X_T, X_temp, y_T, y_temp = train_test_split(X, y, test_size=0.75, stratify=y, 
                                                     random_state=CONFIG['SEED'], shuffle=True)
        
        # Segunda divisão: 33.33% do restante para T_lambda
        X_T_lambda, X_temp2, y_T_lambda, y_temp2 = train_test_split(X_temp, y_temp, test_size=0.6667, 
                                                                     stratify=y_temp, random_state=CONFIG['SEED'], shuffle=True)
        
        # Terceira divisão: 50% do restante para DSEL e 50% para G
        X_DSEL, X_G, y_DSEL, y_G = train_test_split(X_temp2, y_temp2, test_size=0.5, 
                                                     stratify=y_temp2, random_state=CONFIG['SEED'], shuffle=True)
        
        scaler = StandardScaler().fit(X_T)

    elif(dataset=='drive'):
        # UCI Sensorless Drive Diagnosis Dataset
        df = pd.read_csv('data/Sensorless_drive_diagnosis.txt', sep=' ', header=None)
        
        # Target: tipo de falha (última coluna, convertida para 0-indexed)
        y = df.iloc[:, -1] - 1
        X = df.iloc[:, :-1]
        
        # Primeira divisão: 25% para T, 75% restante
        X_T, X_temp, y_T, y_temp = train_test_split(X, y, test_size=0.75, stratify=y, 
                                                     random_state=CONFIG['SEED'], shuffle=True)
        
        # Segunda divisão: 33.33% do restante para T_lambda
        X_T_lambda, X_temp2, y_T_lambda, y_temp2 = train_test_split(X_temp, y_temp, test_size=0.6667, 
                                                                     stratify=y_temp, random_state=CONFIG['SEED'], shuffle=True)
        
        # Terceira divisão: 50% do restante para DSEL e 50% para G
        X_DSEL, X_G, y_DSEL, y_G = train_test_split(X_temp2, y_temp2, test_size=0.5, 
                                                     stratify=y_temp2, random_state=CONFIG['SEED'], shuffle=True)
        
        scaler = StandardScaler().fit(X_T)

    else:
        raise ValueError('Dataset Not Usable')
    
    return X_T, X_T_lambda, X_DSEL, X_G, y_T, y_T_lambda, y_DSEL, y_G, scaler

# ======================================================

def searchAndTrain(dataset, experiment_name, num_trials, load=False):

    mlflow.set_experiment(experiment_name=experiment_name)

    # Obter os 4 subconjuntos
    X_T, X_T_lambda, X_DSEL, X_G, y_T, y_T_lambda, y_DSEL, y_G, scaler = get_data(dataset)
    
    # Normalizar os dados
    X_T_norm = scaler.transform(X_T)
    X_T_lambda_norm = scaler.transform(X_T_lambda)
    X_DSEL_norm = scaler.transform(X_DSEL)
    X_G_norm = scaler.transform(X_G)

    print(f"Tamanhos dos conjuntos:")
    print(f"  T (Pool): {len(X_T)} amostras")
    print(f"  T_lambda (Meta-treino): {len(X_T_lambda)} amostras")
    print(f"  DSEL (Seleção dinâmica): {len(X_DSEL)} amostras")
    print(f"  G (Teste): {len(X_G)} amostras")

    scorer_string = 'f1_macro' if len(set(y_T))>2 else 'f1'
    num_cv_folds = 10 if len(X_T)>500 else 5

    # Funções objetivo permanecem as mesmas, mas usam X_T e y_T
    def dtree_objective(trial:optuna.trial._trial.Trial):
        max_depth = trial.suggest_int('max_depth', 5, 100, log=True)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
        min_samples_split = trial.suggest_int('min_samples_split', 2, 60)
        min_samples_leaf = trial.suggest_int('min_samples_leaf',1, 30)
        
        clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf)
        score = cross_val_score(clf, X_T, y_T, scoring=scorer_string, n_jobs=CONFIG['NUM_WORKERS'], cv=num_cv_folds).mean()
        
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
        
        score = cross_val_score(clf, X_T_norm, y_T, scoring=scorer_string, n_jobs=CONFIG['NUM_WORKERS'], cv=num_cv_folds)
        return score.mean()

    def logreg_objective(trial):
        C = trial.suggest_float('C', 1e-3, 100, log=True)
        solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga'])
        max_iter = trial.suggest_int('max_iter', 100, 1000)
        
        params = {
            'penalty': 'l2', 'C': C, 'solver': solver, 
            'max_iter': max_iter, 'random_state': CONFIG['SEED']
        }
        
        clf = LogisticRegression(**params)
        score = cross_val_score(clf, X_T_norm, y_T, scoring=scorer_string, n_jobs=CONFIG['NUM_WORKERS'], cv=num_cv_folds)
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
        
        score = cross_val_score(clf, X_T_norm, y_T, scoring=scorer_string, n_jobs=CONFIG['NUM_WORKERS'], cv=num_cv_folds)
        return score.mean()

    def svm_linear_objective(trial):
        C = trial.suggest_float('C', 1e-3, 100, log=True)
        max_iter = trial.suggest_int('max_iter', 500, 3000)
        
        clf = SVC(
            kernel='linear', C=C, max_iter=max_iter, random_state=CONFIG['SEED']
        )
        
        score = cross_val_score(clf, X_T_norm, y_T, scoring=scorer_string, n_jobs=CONFIG['NUM_WORKERS'], cv=num_cv_folds)
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
        
        score = cross_val_score(clf, X_T_norm, y_T, scoring=scorer_string, n_jobs=CONFIG['NUM_WORKERS'], cv=num_cv_folds)
        return score.mean()

    def svm_rbf_objective(trial):
        C = trial.suggest_float('C', 1e-3, 100, log=True)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        max_iter = trial.suggest_int('max_iter', 500, 3000)
        
        clf = SVC(
            kernel='rbf', C=C, gamma=gamma, max_iter=max_iter, random_state=CONFIG['SEED']
        )
        
        score = cross_val_score(clf, X_T_norm, y_T, scoring=scorer_string, n_jobs=CONFIG['NUM_WORKERS'], cv=num_cv_folds)
        return score.mean()

    def mlp_objective(trial):
        hidden_layer_sizes = trial.suggest_int('hidden_layer_sizes', 50, 500, step=5)
        activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])
        solver = trial.suggest_categorical('solver', ['adam', 'sgd'])
        alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
        learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'adaptive'])
        learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
        
        clf = MLPClassifier(max_iter=10000, early_stopping=True, 
            n_iter_no_change=20, shuffle=True,
            hidden_layer_sizes=(hidden_layer_sizes,), activation=activation,
            solver=solver, alpha=alpha, learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            random_state=CONFIG['SEED']
        )
        
        score = cross_val_score(clf, X_T_norm, y_T, scoring=scorer_string, n_jobs=CONFIG['NUM_WORKERS'], cv=num_cv_folds)
        return score.mean()

    def rf_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 500, log=True)
        max_depth = trial.suggest_int('max_depth', 5, 100, log=True)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 60)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 30)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
        
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, criterion=criterion,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            random_state=CONFIG['SEED'], n_jobs=CONFIG['NUM_WORKERS']
        )
        
        score = cross_val_score(clf, X_T, y_T, scoring=scorer_string, n_jobs=CONFIG['NUM_WORKERS'], cv=num_cv_folds)
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
        
        score = cross_val_score(clf, X_T, y_T, scoring=scorer_string, n_jobs=CONFIG['NUM_WORKERS'], cv=num_cv_folds)
        return score.mean()

    def ada_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 500, log=True)
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 2, log=True)
        
        clf = AdaBoostClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            random_state=CONFIG['SEED']
        )
        
        score = cross_val_score(clf, X_T, y_T, scoring=scorer_string, n_jobs=CONFIG['NUM_WORKERS'], cv=num_cv_folds)
        return score.mean()

    def xgb_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 500, log=True)
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 1, log=True)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
        gamma = trial.suggest_float('gamma', 0, 5)
        reg_alpha = trial.suggest_float('reg_alpha', 1e-5, 100, log=True)
        reg_lambda = trial.suggest_float('reg_lambda', 1e-5, 100, log=True)
        
        clf = XGBClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, min_child_weight=min_child_weight,
            gamma=gamma, reg_alpha=reg_alpha, reg_lambda=reg_lambda, 
            random_state=CONFIG['SEED'], n_jobs=CONFIG['NUM_WORKERS'], eval_metric='logloss'
        )
        
        score = cross_val_score(clf, X_T, y_T, scoring=scorer_string, n_jobs=1, cv=num_cv_folds)
        return score.mean()

    # ============================================

    loaded_models = {}

    # Treinamento dos modelos usando T (pool)
    model_name = 'Decision_Tree'
    try:
        loaded_models[model_name] = load_model_by_name(experiment_name=experiment_name, run_name=model_name)[0]
    except ValueError:
        dtree_study = optuna.create_study(direction='maximize')
        dtree_study.optimize(dtree_objective, n_trials=num_trials)
        dtree_params = dtree_study.best_params
        loaded_models[model_name] = DecisionTreeClassifier(**dtree_params, random_state=CONFIG['SEED']).fit(X_T, y_T)
        log_model_to_mlflow(
            loaded_models[model_name], model_name, dtree_params, 
            X_T, y_T, X_G, y_G
        )

    model_name = 'SGD'
    try:
        loaded_models[model_name] = load_model_by_name(experiment_name=experiment_name, run_name=model_name)[0]
    except ValueError:
        sgd_study = optuna.create_study(direction='maximize')
        sgd_study.optimize(sgd_objective, n_trials=num_trials)
        sgd_params = sgd_study.best_params
        loaded_models[model_name] = Pipeline([
            ('scaler', scaler),
            ('clf', SGDClassifier(**sgd_params, random_state=CONFIG['SEED']).fit(X_T_norm, y_T))
        ])
        log_model_to_mlflow(
            loaded_models[model_name], model_name, sgd_params,
            X_T_norm, y_T, X_G_norm, y_G
        )

    model_name = 'Logistic_Regression'
    try:
        loaded_models[model_name] = load_model_by_name(experiment_name=experiment_name, run_name=model_name)[0]
    except ValueError:
        logreg_study = optuna.create_study(direction='maximize')
        logreg_study.optimize(logreg_objective, n_trials=num_trials)
        logreg_params = logreg_study.best_params
        loaded_models[model_name] = Pipeline([
            ('scaler', scaler),
            ('clf', LogisticRegression(**logreg_params, random_state=CONFIG['SEED']).fit(X_T_norm, y_T))
        ])
        log_model_to_mlflow(
            loaded_models[model_name], model_name, logreg_params,
            X_T_norm, y_T, X_G_norm, y_G
        )

    model_name = 'KNN'
    try:
        loaded_models[model_name] = load_model_by_name(experiment_name=experiment_name, run_name=model_name)[0]
    except ValueError:
        knn_study = optuna.create_study(direction='maximize')
        knn_study.optimize(knn_objective, n_trials=num_trials)
        knn_params = knn_study.best_params
        loaded_models[model_name] = Pipeline([
            ('scaler', scaler),
            ('clf', KNeighborsClassifier(**knn_params).fit(X_T_norm, y_T))
        ])
        log_model_to_mlflow(
            loaded_models[model_name], model_name, knn_params,
            X_T_norm, y_T, X_G_norm, y_G
        )

    model_name = 'SVM_Linear'
    try:
        loaded_models[model_name] = load_model_by_name(experiment_name=experiment_name, run_name=model_name)[0]
    except ValueError:
        svm_linear_study = optuna.create_study(direction='maximize')
        svm_linear_study.optimize(svm_linear_objective, n_trials=num_trials)
        svm_linear_params = svm_linear_study.best_params
        loaded_models[model_name] = Pipeline([
            ('scaler', scaler),
            ('clf', SVC(kernel='linear', **svm_linear_params, random_state=CONFIG['SEED'], probability=True).fit(X_T_norm, y_T))
        ])
        log_model_to_mlflow(
            loaded_models[model_name], model_name, svm_linear_params,
            X_T_norm, y_T, X_G_norm, y_G
        )

    model_name = 'SVM_Polynomial'
    try:
        loaded_models[model_name] = load_model_by_name(experiment_name=experiment_name, run_name=model_name)[0]
    except ValueError:
        svm_poly_study = optuna.create_study(direction='maximize')
        svm_poly_study.optimize(svm_poly_objective, n_trials=num_trials)
        svm_poly_params = svm_poly_study.best_params
        loaded_models[model_name] = Pipeline([
            ('scaler', scaler),
            ('clf', SVC(kernel='poly', **svm_poly_params, random_state=CONFIG['SEED'], probability=True).fit(X_T_norm, y_T))
        ])
        log_model_to_mlflow(
            loaded_models[model_name], model_name, svm_poly_params,
            X_T_norm, y_T, X_G_norm, y_G
        )

    model_name = 'SVM_RBF'
    try:
        loaded_models[model_name] = load_model_by_name(experiment_name=experiment_name, run_name=model_name)[0]
    except ValueError:
        svm_rbf_study = optuna.create_study(direction='maximize')
        svm_rbf_study.optimize(svm_rbf_objective, n_trials=num_trials)
        svm_rbf_params = svm_rbf_study.best_params
        loaded_models[model_name] = Pipeline([
            ('scaler', scaler),
            ('clf', SVC(kernel='rbf', **svm_rbf_params, random_state=CONFIG['SEED'], probability=True).fit(X_T_norm, y_T))
        ])
        log_model_to_mlflow(
            loaded_models[model_name], model_name, svm_rbf_params,
            X_T_norm, y_T, X_G_norm, y_G
        )

    model_name = 'MLP'
    try:
        loaded_models[model_name] = load_model_by_name(experiment_name=experiment_name, run_name=model_name)[0]
    except ValueError:
        mlp_study = optuna.create_study(direction='maximize')
        mlp_study.optimize(mlp_objective, n_trials=num_trials)
        mlp_params = mlp_study.best_params
        mlp_params['hidden_layer_sizes'] = (mlp_params['hidden_layer_sizes'],)
        loaded_models[model_name] = Pipeline([
            ('scaler', scaler),
            ('clf', MLPClassifier(**mlp_params, random_state=CONFIG['SEED']).fit(X_T_norm, y_T))
        ])
        log_model_to_mlflow(
            loaded_models[model_name], model_name, mlp_params,
            X_T_norm, y_T, X_G_norm, y_G
        )

    model_name = 'Random_Forest'
    try:
        loaded_models[model_name] = load_model_by_name(experiment_name=experiment_name, run_name=model_name)[0]
    except ValueError:
        rf_study = optuna.create_study(direction='maximize')
        rf_study.optimize(rf_objective, n_trials=num_trials)
        rf_params = rf_study.best_params
        loaded_models[model_name] = RandomForestClassifier(**rf_params, random_state=CONFIG['SEED'], n_jobs=CONFIG['NUM_WORKERS']).fit(X_T, y_T)
        log_model_to_mlflow(
            loaded_models[model_name], model_name, rf_params,
            X_T, y_T, X_G, y_G
        )

    model_name = 'Gradient_Boosting'
    try:
        loaded_models[model_name] = load_model_by_name(experiment_name=experiment_name, run_name=model_name)[0]
    except ValueError:
        gb_study = optuna.create_study(direction='maximize')
        gb_study.optimize(gb_objective, n_trials=num_trials)
        gb_params = gb_study.best_params
        loaded_models[model_name] = GradientBoostingClassifier(**gb_params, random_state=CONFIG['SEED']).fit(X_T, y_T)
        log_model_to_mlflow(
            loaded_models[model_name], model_name, gb_params,
            X_T, y_T, X_G, y_G
        )

    model_name = 'AdaBoost'
    try:
        loaded_models[model_name] = load_model_by_name(experiment_name=experiment_name, run_name=model_name)[0]
    except ValueError:
        ada_study = optuna.create_study(direction='maximize')
        ada_study.optimize(ada_objective, n_trials=num_trials)
        ada_params = ada_study.best_params
        loaded_models[model_name] = AdaBoostClassifier(**ada_params, random_state=CONFIG['SEED']).fit(X_T, y_T)
        log_model_to_mlflow(
            loaded_models[model_name], model_name, ada_params,
            X_T, y_T, X_G, y_G
        )

    model_name = 'XGBoost'
    try:
        loaded_models[model_name] = load_model_by_name(experiment_name=experiment_name, run_name=model_name)[0]
    except ValueError:
        xgb_study = optuna.create_study(direction='maximize')
        xgb_study.optimize(xgb_objective, n_trials=num_trials)
        xgb_params = xgb_study.best_params
        loaded_models[model_name] = XGBClassifier(**xgb_params, random_state=CONFIG['SEED'], n_jobs=CONFIG['NUM_WORKERS'], eval_metric='logloss', enable_categorical=True).fit(X_T, y_T)
        log_model_to_mlflow(
            loaded_models[model_name], model_name, xgb_params,
            X_T, y_T, X_G, y_G
        )

    # METADES - treina em T_lambda (meta-treinamento) e valida em DSEL
    pool_classifiers = list(loaded_models.values())

    try:
        metades = load_model_by_name(experiment_name=experiment_name, run_name='METADES')[0]
    except ValueError:
        metades = METADES(pool_classifiers, random_state=CONFIG['SEED'], voting='soft').fit(X_T_lambda, y_T_lambda)
        mdes_params = metades.get_params()
        mdes_params.pop('pool_classifiers')
        log_model_to_mlflow(
            metades, "METADES", mdes_params,
            X_DSEL, y_DSEL, X_G, y_G
        )

    def mdesr_objective(trial):
        k = trial.suggest_int('k', 3, 25, log=True)
        kp = trial.suggest_int('Kp', 3, 25, log=True)

        n_estimators = trial.suggest_int('n_estimators', 10, 500, log=True)
        max_depth = trial.suggest_int('max_depth', 5, 100, log=True)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 60)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 30)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
        
        meta_clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, criterion=criterion,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            random_state=CONFIG['SEED'], n_jobs=CONFIG['NUM_WORKERS']
        )

        clf = METADESR(pool_classifiers, meta_classifier=meta_clf, k=k, Kp=kp, random_state=CONFIG['SEED'], voting='soft', selection_threshold=0.5, rejection_rate=0.0).fit(X_T_lambda, y_T_lambda)
        y_pred = clf.predict(X_DSEL)
        score = f1_score(y_DSEL[~y_pred.mask], y_pred[~y_pred.mask], zero_division=0, average='macro')
        #score = cross_val_score(clf, X_T, y_T, scoring=scorer_string, n_jobs=1, cv=num_cv_folds)
        return score#.mean()

    try:
        metadesr = load_model_by_name(experiment_name=experiment_name, run_name='METADESR')[0]
    except ValueError:
        mdesr_study = optuna.create_study(direction='maximize')
        mdesr_study.optimize(mdesr_objective, n_trials=num_trials)
        mdesr_params = mdesr_study.best_params
        meta_clf = RandomForestClassifier(
            n_estimators=mdesr_params['n_estimators'], max_depth=mdesr_params['max_depth'], criterion=mdesr_params['criterion'],
            min_samples_split=mdesr_params['min_samples_split'], min_samples_leaf=mdesr_params['min_samples_leaf'],
            random_state=CONFIG['SEED'], n_jobs=CONFIG['NUM_WORKERS']
        )
        metadesr = METADESR(pool_classifiers, meta_classifier=meta_clf, 
                            k=mdesr_params['k'], Kp=mdesr_params['Kp'], random_state=CONFIG['SEED'], 
                            n_jobs=1, voting='soft', selection_threshold=0.0, rejection_rate=0.0).fit(X_T_lambda, y_T_lambda)
        mdesr_params = metadesr.get_params()
        mdesr_params.pop('pool_classifiers')
        log_model_to_mlflow(
            metadesr, "METADESR", mdesr_params,
            X_T_lambda, y_T_lambda, X_G, y_G
        )

    if(load):
        return {
            'pool_classifiers': loaded_models, 
            'METADES': metades, 
            'METADESR': metadesr
        }

def getExpName(dataset):
    global CONFIG
    dataset = re.sub('[-_ ]', '', dataset).lower()
    return f"{dataset}_{CONFIG['VERSION']}_{CONFIG['SEED']}"

if(__name__=='__main__'):
    NUM_TRIALS = 25
    for DATASET in ['qsar', 'adult', 'magic_gamma', 'credit', 'marketing', 'drive']:
        experiment_name = getExpName(DATASET)

        searchAndTrain(dataset=DATASET, 
                    experiment_name=experiment_name, 
                    num_trials=NUM_TRIALS)