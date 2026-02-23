import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier


BASE_MODEL_LIBRARY = [
    ("logreg", LogisticRegression(max_iter=1000)),
    ("tree", DecisionTreeClassifier(max_depth=8)),
    ("knn", KNeighborsClassifier(n_neighbors=7)),
    ("svc_rbf", SVC(probability=True, kernel="rbf")),
    ("rf", RandomForestClassifier(n_estimators=300, n_jobs=-1)),
    ("extra_trees", ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=42)),
    ("gradient_boosting", GradientBoostingClassifier(random_state=42)),
    ("naive_bayes", GaussianNB()),
    ("qda", QuadraticDiscriminantAnalysis()),
    ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)),
]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )


ROOT_DIR = Path(__file__).resolve().parent.parent  # raiz do projeto


def load_dataset(dataset_cfg: dict, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    dataset_name = dataset_cfg.get("name", "dataset_sem_nome")
    print(f"\n[DATASET] Iniciando carregamento: '{dataset_name}'")

    if "path" not in dataset_cfg:
        print("[DATASET] Fonte: sintética (make_classification)")
        X, y = make_classification(
            n_samples=2000,
            n_features=20,
            n_informative=10,
            n_redundant=2,
            random_state=seed,
        )
        print(f"[DATASET] Concluído: X={X.shape}, y={y.shape}")
        return pd.DataFrame(X), pd.Series(y)
    print(f"[DATASET] Fonte: arquivo CSV -> {dataset_cfg['path']}")
    df = pd.read_csv(
        dataset_cfg["path"],
        sep=None,
        engine="python"
    )
    df.columns = df.columns.str.strip()
    target_col = dataset_cfg["target"]
    print(f"[DATASET] Colunas ({len(df.columns)}): {list(df.columns)}")
    print(f"[DATASET] Coluna alvo: {target_col}")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    print(f"[DATASET] Concluído: X={X.shape}, y={y.shape}")
    return X, y


def borda_count_prediction(probas: np.ndarray, class_labels: np.ndarray) -> np.ndarray:
    n_models, n_samples, n_classes = probas.shape
    borda_scores = np.zeros((n_samples, n_classes), dtype=float)

    for model_idx in range(n_models):
        model_proba = probas[model_idx]
        order = np.argsort(model_proba, axis=1)

        model_points = np.zeros_like(model_proba, dtype=float)
        for rank in range(n_classes):
            cls_idx = order[:, rank]
            model_points[np.arange(n_samples), cls_idx] = rank

        borda_scores += model_points

    return class_labels[np.argmax(borda_scores, axis=1)]


def metrics_no_reject(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }




def print_dataset_summary(dataset_name: str, rows: list[dict]) -> None:
    if not rows:
        print(f"[RESUMO] Dataset '{dataset_name}' sem resultados para imprimir.")
        return

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["method", "ensemble_size"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            n_execucoes=("accuracy", "count"),
        )
        .sort_values(["accuracy_mean", "f1_macro_mean"], ascending=False)
    )

    print(f"\n[RESUMO] Dataset '{dataset_name}' - desempenho por método")
    for _, row in summary.iterrows():
        acc_std = 0.0 if pd.isna(row["accuracy_std"]) else row["accuracy_std"]
        f1_std = 0.0 if pd.isna(row["f1_macro_std"]) else row["f1_macro_std"]
        print(
            "  - "
            f"method={row['method']:<18} "
            f"ensemble_size={int(row['ensemble_size']):<2d} "
            f"accuracy={row['accuracy_mean']:.4f} ± {acc_std:.4f} "
            f"f1_macro={row['f1_macro_mean']:.4f} ± {f1_std:.4f} "
            f"(n={int(row['n_execucoes'])})"
        )

    best_row = summary.iloc[0]
    print(
        "[RESUMO] Melhor método no dataset: "
        f"{best_row['method']} (ensemble_size={int(best_row['ensemble_size'])}) "
        f"com accuracy={best_row['accuracy_mean']:.4f} e f1_macro={best_row['f1_macro_mean']:.4f}"
    )

def run_single_dataset(dataset_cfg: dict, cfg: dict) -> list[dict]:
    seed = cfg["seed"]
    dataset_name = dataset_cfg.get("name", "dataset_sem_nome")
    print(f"\n[ETAPA 1/6] Preparando dataset '{dataset_name}'")
    X, y = load_dataset(dataset_cfg, seed)

    print(f"[ETAPA 2/6] Split treino/teste (test_size={cfg['test_size']}, seed={seed})")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["test_size"],
        random_state=seed,
        stratify=y,
    )
    print(f"[ETAPA 2/6] Tamanhos -> treino={X_train.shape}, teste={X_test.shape}")

    print("[ETAPA 3/6] Construindo pré-processador")
    preprocessor = build_preprocessor(X_train)
    library = []
    print("[ETAPA 4/6] Treinando biblioteca de modelos base")
    for name, model in BASE_MODEL_LIBRARY:
        print(f"  - Treinando modelo: {name}")
        pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clone(model))])
        pipe.fit(X_train, y_train)
        library.append((name, pipe))

    results = []

    print("[ETAPA 5/6] Avaliando ensembles sem rejeição")
    for ensemble_size in cfg["ensemble_sizes"]:
        print(f"  - Ensemble size: {ensemble_size}")
        selected_models = library[: min(ensemble_size, len(library))]
        model_names = [name for name, _ in selected_models]
        print(f"    · Modelos usados ({len(model_names)}): {model_names}")

        # `probas` guarda as probabilidades previstas por cada modelo base.
        # Formato: (n_modelos, n_amostras, n_classes)
        probas = np.stack([model.predict_proba(X_test) for _, model in selected_models], axis=0)

        # Mapeamento de índices para rótulos reais das classes.
        class_labels = selected_models[0][1].named_steps["clf"].classes_

        # `votes` guarda as classes previstas por cada modelo base.
        # Formato: (n_modelos, n_amostras)
        votes = np.stack([model.predict(X_test) for _, model in selected_models], axis=0)

        # 1) SUM RULE (Soma):
        # Soma as probabilidades por classe entre os modelos e escolhe a classe com maior soma.
        pred_sum = class_labels[np.argmax(np.sum(probas, axis=0), axis=1)]

        # 2) PRODUCT RULE (Produto):
        # Multiplica as probabilidades por classe entre os modelos e escolhe a classe com maior produto.
        pred_product = class_labels[np.argmax(np.prod(probas, axis=0), axis=1)]

        # 3) MAX RULE (Máximo):
        # Para cada classe, pega a maior probabilidade atribuída por qualquer modelo.
        # Em seguida escolhe a classe com maior valor máximo.
        pred_max = class_labels[np.argmax(np.max(probas, axis=0), axis=1)]

        # 4) MIN RULE (Mínimo):
        # Para cada classe, pega a menor probabilidade atribuída entre os modelos.
        # Em seguida escolhe a classe com maior desses mínimos.
        pred_min = class_labels[np.argmax(np.min(probas, axis=0), axis=1)]

        # 5) MEDIAN RULE (Mediana):
        # Para cada classe, calcula a mediana das probabilidades entre modelos
        # e escolhe a classe com maior mediana.
        pred_median = class_labels[np.argmax(np.median(probas, axis=0), axis=1)]

        # 6) BORDA COUNT:
        # Converte probabilidades em ranking por modelo e soma pontos de ranking por classe.
        # A classe com maior pontuação final é selecionada.
        pred_borda = borda_count_prediction(probas, class_labels)

        # 7) MAJORITY VOTING (Voto majoritário):
        # Em cada amostra, conta os votos de classe dos modelos e escolhe a classe mais votada.
        pred_majority = []
        for col in votes.T:
            values, counts = np.unique(col, return_counts=True)
            pred_majority.append(values[counts.argmax()])
        pred_majority = np.asarray(pred_majority)

        for method_name, pred in [
            ("majority_voting", pred_majority),
            ("sum_rule", pred_sum),
            ("product_rule", pred_product),
            ("max_rule", pred_max),
            ("min_rule", pred_min),
            ("median_rule", pred_median),
            ("borda_count", pred_borda),
        ]:
            row = {
                "dataset": dataset_cfg["name"],
                "method": method_name,
                "ensemble_size": len(selected_models),
            }
            row.update(metrics_no_reject(np.asarray(y_test), pred))
            results.append(row)

    print(f"[ETAPA 6/6] Dataset '{dataset_name}' finalizado com {len(results)} linhas de resultado")
    print_dataset_summary(dataset_name, results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="study_config.json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    print(f"[INÍCIO] Configuração carregada de: {args.config}")
    print(f"[INÍCIO] Datasets na execução: {[d.get('name', 'dataset_sem_nome') for d in cfg['datasets']]}")

    all_rows = []

    for dataset_cfg in cfg["datasets"]:
        all_rows.extend(run_single_dataset(dataset_cfg, cfg))

    out_path = Path(cfg["results_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"Resultados salvos em: {out_path}")


if __name__ == "__main__":
    main()
