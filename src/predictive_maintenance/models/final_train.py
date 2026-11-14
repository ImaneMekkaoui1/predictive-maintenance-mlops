

import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from predictive_maintenance.models.train import (
    DATA_DIR,
    PROJECT_ROOT,
    TARGET,
    get_experiments,
)

# ==================================================
# 1. CONFIGURATION
# ==================================================

MODEL_NAME = "XGBoost_baseline"
THRESHOLD = 0.15

REPORTS_DIR = PROJECT_ROOT / "reports"


# ==================================================
# 2. CHARGEMENT DES DONNEES
# ==================================================

def load_final_data():

    # Données d'entraînement : 8 000 observations
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")[TARGET]

    # Données de test : 2 000 observations
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")[TARGET]

    # Même nettoyage des colonnes que dans train.py
    for X in [X_train, X_test]:

        X.columns = (
            X.columns
            .str.replace("[", "_", regex=False)
            .str.replace("]", "_", regex=False)
            .str.replace("<", "_", regex=False)
        )

    return X_train, X_test, y_train, y_test


# ==================================================
# 3. ENTRAINEMENT ET EVALUATION FINALE
# ==================================================

def main():

    # Configuration MLflow
    mlflow.set_tracking_uri(
        f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    )

    mlflow.set_experiment("maintenance-final-model")

    # Charger les données
    X_train, X_test, y_train, y_test = load_final_data()

    print("\n========== DONNEES ==========")
    print("Training shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Failures in training:", int(y_train.sum()))
    print("Failures in test:", int(y_test.sum()))

    # Récupérer exactement la configuration XGBoost expérimentée
    model = get_experiments()[MODEL_NAME]

    with mlflow.start_run(run_name="XGBoost_final_test"):

        # ------------------------------------------
        # ENTRAINEMENT SUR LES 8000 OBSERVATIONS
        # ------------------------------------------

        print("\n========== ENTRAINEMENT ==========")

        model.fit(X_train, y_train)

        print("Entraînement terminé.")

        # ------------------------------------------
        # PREDICTIONS SUR LE TEST
        # ------------------------------------------

        y_proba = model.predict_proba(X_test)[:, 1]

        # Application du seuil choisi sur validation
        y_pred = (y_proba > THRESHOLD).astype(int)

        # ------------------------------------------
        # CALCUL DES METRIQUES
        # ------------------------------------------

        precision = precision_score(
            y_test, y_pred, zero_division=0
        )

        recall = recall_score(
            y_test, y_pred, zero_division=0
        )

        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )

        roc_auc = roc_auc_score(y_test, y_proba)

        pr_auc = average_precision_score(y_test, y_proba)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        ).ravel()

        specificity = tn / (tn + fp) if (tn + fp) else 0

        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "specificity": float(specificity),
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
        }

        # ------------------------------------------
        # ENREGISTREMENT DANS MLFLOW
        # ------------------------------------------

        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("decision_threshold", THRESHOLD)
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("random_state", 42)
        mlflow.log_param("selection_objective", "recall")

        model_params = {
            key: value
            for key, value in model.get_params().items()
            if isinstance(value, (str, int, float, bool))
        }

        mlflow.log_params(model_params)
        mlflow.log_metrics(metrics)

        # Enregistrer la matrice de confusion
        mlflow.log_dict(
            {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
            "confusion_matrix.json",
        )

        # Enregistrer la configuration de prédiction
        mlflow.log_dict(
            {
                "model_name": MODEL_NAME,
                "threshold": THRESHOLD,
                "positive_class": 1,
                "decision_rule": "probability > threshold",
            },
            "prediction_config.json",
        )

        # Enregistrer le modèle XGBoost
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="model",
        )

        # ------------------------------------------
        # GENERATION DU RAPPORT CSV
        # ------------------------------------------

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        report_path = REPORTS_DIR / "final_test_metrics.csv"

        pd.DataFrame([metrics]).to_csv(
            report_path,
            index=False,
        )

        mlflow.log_artifact(str(report_path))

        # ------------------------------------------
        # AFFICHAGE DES RESULTATS
        # ------------------------------------------

        print("\n========== RESULTATS FINAUX SUR TEST ==========")

        for name, value in metrics.items():

            if isinstance(value, float):
                print(f"{name}: {value:.4f}")
            else:
                print(f"{name}: {value}")

        print("\nRapport enregistré :", report_path)
        print("Modèle enregistré dans MLflow.")


if __name__ == "__main__":
    main()
