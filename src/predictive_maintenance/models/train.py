
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

TARGET = "Machine failure"
RANDOM_STATE = 42


# ==================================================
# 2. CHARGEMENT DES DONNEES
# ==================================================

def load_data():

    X = pd.read_csv(DATA_DIR / "X_train.csv")
    y = pd.read_csv(DATA_DIR / "y_train.csv")[TARGET]

    # Nettoyage des noms de colonnes pour compatibilité XGBoost
    X.columns = (
        X.columns
        .str.replace("[", "_", regex=False)
        .str.replace("]", "_", regex=False)
        .str.replace("<", "_", regex=False)
    )

    # Séparation entraînement / validation
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_val, y_train, y_val


# ==================================================
# 3. DEFINITION DES ALGORITHMES
# ==================================================

def get_experiments():

    experiments = {

        # Logistic Regression - Baseline
        "Logistic_Regression_baseline": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            )),
        ]),

        # Logistic Regression - Balanced
        "Logistic_Regression_balanced": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),

        # Random Forest - Baseline
        "Random_Forest_baseline": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        # Random Forest - Balanced
        "Random_Forest_balanced": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        # XGBoost - Baseline
        "XGBoost_baseline": xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        # XGBoost - Balanced
        "XGBoost_balanced": xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=28.5,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    return experiments


# ==================================================
# 4. EVALUATION ET SUIVI MLFLOW
# ==================================================

def train_and_log(name, model, X_train, X_val, y_train, y_val):

    with mlflow.start_run(run_name=name):

        # Entraînement
        model.fit(X_train, y_train)

        # Prédictions
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        # Calcul des métriques
        metrics = {
            "precision": precision_score(
                y_val, y_pred, zero_division=0
            ),
            "recall": recall_score(
                y_val, y_pred, zero_division=0
            ),
            "f1_score": f1_score(
                y_val, y_pred, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_val, y_proba),
            "pr_auc": average_precision_score(y_val, y_proba),
        }

        # Enregistrement des paramètres et métriques dans MLflow
        mlflow.log_param("model_name", name)
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)

        # Matrice de confusion
        tn, fp, fn, tp = confusion_matrix(
            y_val,
            y_pred,
            labels=[0, 1],
        ).ravel()

        mlflow.log_dict(
            {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
            "confusion_matrix.json",
        )

        # Sauvegarde du modèle dans MLflow selon sa bibliothèque
        if isinstance(model, xgb.XGBClassifier):

            mlflow.xgboost.log_model(
                xgb_model=model,
                name="model",
            )

        else:

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

        # Affichage des résultats
        print(f"\n========== {name} ==========")

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")


# ==================================================
# 5. EXECUTION PRINCIPALE
# ==================================================

def main():

    # Configuration MLflow
    mlflow.set_tracking_uri(
        f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    )

    mlflow.set_experiment("maintenance-predictive")

    # Chargement des données
    X_train, X_val, y_train, y_val = load_data()

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_val.shape)

    print("\nTraining class distribution:")
    print(y_train.value_counts(normalize=True))

    # Récupération des expériences
    experiments = get_experiments()

    # Exécution des six expériences
    for name, model in experiments.items():

        train_and_log(
            name,
            model,
            X_train,
            X_val,
            y_train,
            y_val,
        )


if __name__ == "__main__":
    main()
