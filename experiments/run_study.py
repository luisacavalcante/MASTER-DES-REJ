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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from classes import Pool


BASE_MODEL_LIBRARY = [
    ("logreg", LogisticRegression(max_iter=1000)),
    ("tree", DecisionTreeClassifier(max_depth=8)),
    ("knn", KNeighborsClassifier(n_neighbors=7)),
    ("svc_rbf", SVC(probability=True, kernel="rbf")),
    ("rf", RandomForestClassifier(n_estimators=300, n_jobs=-1)),
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


def load_dataset(dataset_cfg: dict, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    if "path" not in dataset_cfg:
        X, y = make_classification(
            n_samples=2000,
            n_features=20,
            n_informative=10,
            n_redundant=2,
            random_state=seed,
        )
        return pd.DataFrame(X), pd.Series(y)

    df = pd.read_csv(dataset_cfg["path"])
    target_col = dataset_cfg["target"]
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


def reject_from_proba(y_proba: np.ndarray, reject_rate: float) -> np.ndarray:
    confidence = np.max(y_proba, axis=1)
    k = int(len(confidence) * reject_rate)
    reject_mask = np.zeros(len(confidence), dtype=bool)
    if k > 0:
        idx = np.argsort(confidence)[:k]
        reject_mask[idx] = True
    return reject_mask


def metrics_with_reject(y_true: np.ndarray, y_pred: np.ndarray, reject_mask: np.ndarray) -> dict:
    accepted = ~reject_mask
    coverage = float(accepted.mean())
    if accepted.sum() == 0:
        return {
            "coverage": coverage,
            "accuracy_accept": np.nan,
            "f1_accept": np.nan,
            "reject_rate_observed": float(reject_mask.mean()),
        }

    return {
        "coverage": coverage,
        "accuracy_accept": float(accuracy_score(y_true[accepted], y_pred[accepted])),
        "f1_accept": float(f1_score(y_true[accepted], y_pred[accepted], average="macro")),
        "reject_rate_observed": float(reject_mask.mean()),
    }


def run_single_dataset(dataset_cfg: dict, cfg: dict) -> list[dict]:
    seed = cfg["seed"]
    X, y = load_dataset(dataset_cfg, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["test_size"],
        random_state=seed,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)
    library = []
    for name, model in BASE_MODEL_LIBRARY:
        pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clone(model))])
        pipe.fit(X_train, y_train)
        library.append((name, pipe))

    results = []

    for ensemble_size in cfg["ensemble_sizes"]:
        selected_models = library[: min(ensemble_size, len(library))]

        # 1) Ensemble simples: média das probabilidades
        probas = np.stack([model.predict_proba(X_test) for _, model in selected_models], axis=0)
        proba_mean = probas.mean(axis=0)
        class_labels = selected_models[0][1].named_steps["clf"].classes_
        pred_simple = class_labels[proba_mean.argmax(axis=1)]

        # 2) Combinação majoritária
        votes = np.stack([model.predict(X_test) for _, model in selected_models], axis=0)
        pred_majority = []
        for col in votes.T:
            values, counts = np.unique(col, return_counts=True)
            pred_majority.append(values[counts.argmax()])
        pred_majority = np.asarray(pred_majority)

        # 3) Plugin: usa classe Pool já existente
        plugin_pool = Pool({name: model for name, model in selected_models})
        pred_plugin = plugin_pool.predict(X_test).mode(axis=1).iloc[:, 0].to_numpy()

        for reject_rate in cfg["reject_rates"]:
            reject_simple = reject_from_proba(proba_mean, reject_rate)

            maj_as_proba = np.stack([model.predict_proba(X_test) for _, model in selected_models], axis=0).mean(axis=0)
            reject_majority = reject_from_proba(maj_as_proba, reject_rate)

            plugin_proba = np.stack([model.predict_proba(X_test) for _, model in selected_models], axis=0).mean(axis=0)
            reject_plugin = reject_from_proba(plugin_proba, reject_rate)

            for method_name, pred, reject_mask in [
                ("ensemble_simples", pred_simple, reject_simple),
                ("combinacao_majoritaria", pred_majority, reject_majority),
                ("plugin", pred_plugin, reject_plugin),
            ]:
                row = {
                    "dataset": dataset_cfg["name"],
                    "method": method_name,
                    "ensemble_size": ensemble_size,
                    "reject_rate_target": reject_rate,
                }
                row.update(metrics_with_reject(np.asarray(y_test), pred, reject_mask))
                results.append(row)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/study_config.json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    all_rows = []
    for dataset_cfg in cfg["datasets"]:
        all_rows.extend(run_single_dataset(dataset_cfg, cfg))

    out_path = Path(cfg["results_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"Resultados salvos em: {out_path}")


if __name__ == "__main__":
    main()
