

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Réutilisation des fonctions et des 9 modèles de train.py
from predictive_maintenance.models.train import (
    PROJECT_ROOT,
    get_experiments,
    get_smote_experiments,
    load_data,
)

# ==================================================
# 1. CONFIGURATION
# ==================================================

REPORTS_DIR = PROJECT_ROOT / "reports"

# Seuils testés : 0.10, 0.15, ..., 0.90
THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)

# Objectif de détection retenu pour l'analyse :
# rechercher les seuils avec un Recall >= 80 %
RECALL_TARGET = 0.80


# ==================================================
# 2. EVALUATION POUR UN SEUIL
# ==================================================

def evaluate_threshold(y_true, y_proba, threshold):
    """
    Transforme les probabilités en classes selon le seuil,
    puis calcule les métriques de classification.
    """

    y_pred = (y_proba > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "specificity": float(specificity),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


# ==================================================
# 3. OPTIMISATION DES SEUILS
# ==================================================

def optimize_model(name, model, X_train, X_val, y_train, y_val):

    print(f"\n{'=' * 65}")
    print(f"Modèle : {name}")
    print(f"{'=' * 65}")

    # Entraîner le modèle sur les données d'entraînement
    model.fit(X_train, y_train)

    # Récupérer les probabilités sur la validation
    # On ne modifie pas le modèle entre les différents seuils.
    y_proba = model.predict_proba(X_val)[:, 1]

    results = []

    # Un seul run MLflow par modèle
    with mlflow.start_run(run_name=f"threshold_{name}"):

        mlflow.log_param("model_name", name)
        mlflow.log_param("number_of_thresholds", len(THRESHOLDS))
        mlflow.log_param("threshold_min", float(THRESHOLDS.min()))
        mlflow.log_param("threshold_max", float(THRESHOLDS.max()))
        mlflow.log_param("threshold_step", 0.05)
        mlflow.log_param("recall_target", RECALL_TARGET)

        # Tester tous les seuils
        for threshold in THRESHOLDS:

            metrics = evaluate_threshold(
                y_val,
                y_proba,
                threshold,
            )

            results.append({
                "model": name,
                **metrics,
            })

            # Enregistrer les métriques dans MLflow
            suffix = str(threshold).replace(".", "_")

            mlflow.log_metrics({
                f"precision_at_{suffix}": metrics["precision"],
                f"recall_at_{suffix}": metrics["recall"],
                f"f1_at_{suffix}": metrics["f1_score"],
                f"specificity_at_{suffix}": metrics["specificity"],
                f"TP_at_{suffix}": metrics["TP"],
                f"FP_at_{suffix}": metrics["FP"],
                f"TN_at_{suffix}": metrics["TN"],
                f"FN_at_{suffix}": metrics["FN"],
            })

        # Tableau des résultats de ce modèle
        model_results = pd.DataFrame(results)

        # Meilleur seuil selon le F1-score
        best_f1 = model_results.loc[
            model_results["f1_score"].idxmax()
        ]

        # Meilleur seuil parmi ceux qui atteignent Recall >= 0.80
        recall_candidates = model_results[
            model_results["recall"] >= RECALL_TARGET
        ]

        if not recall_candidates.empty:
            best_recall_target = recall_candidates.loc[
                recall_candidates["precision"].idxmax()
            ]

            mlflow.log_metrics({
                "best_recall_target_threshold":
                    float(best_recall_target["threshold"]),
                "best_recall_target_precision":
                    float(best_recall_target["precision"]),
                "best_recall_target_recall":
                    float(best_recall_target["recall"]),
                "best_recall_target_f1":
                    float(best_recall_target["f1_score"]),
            })

        # Enregistrer le meilleur F1 dans MLflow
        mlflow.log_metrics({
            "best_f1_threshold": float(best_f1["threshold"]),
            "best_f1_precision": float(best_f1["precision"]),
            "best_f1_recall": float(best_f1["recall"]),
            "best_f1_score": float(best_f1["f1_score"]),
        })

    # Affichage des résultats
    print("\nRésultats par seuil :")
    print(
        model_results[
            ["threshold", "precision", "recall", "f1_score",
             "TP", "FP", "TN", "FN"]
        ].to_string(index=False)
    )

    print("\nMeilleur seuil selon F1-score :")
    print(best_f1.to_string())

    if not recall_candidates.empty:
        print("\nMeilleur seuil avec Recall >= 0.80 :")
        print(best_recall_target.to_string())
    else:
        print("\nAucun seuil n'atteint le Recall cible de 0.80.")

    return results


# ==================================================
# 4. EXECUTION PRINCIPALE
# ==================================================

def main():

    # Configuration MLflow
    mlflow.set_tracking_uri(
        f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    )

    mlflow.set_experiment("maintenance-threshold-optimization")

    # Charger les mêmes données train/validation que train.py
    X_train, X_val, y_train, y_val = load_data()

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Nombre de défaillances en validation:", int(y_val.sum()))
    print("Seuils testés:", THRESHOLDS)

    # Récupérer les 9 expériences
    experiments = get_experiments()
    experiments.update(get_smote_experiments())

    print(f"\nNombre de modèles à optimiser : {len(experiments)}")

    all_results = []

    for name, model in experiments.items():

        model_results = optimize_model(
            name,
            model,
            X_train,
            X_val,
            y_train,
            y_val,
        )

        all_results.extend(model_results)

    # ==================================================
    # 5. SAUVEGARDE DU RAPPORT
    # ==================================================

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_results)

    csv_path = REPORTS_DIR / "threshold_optimization.csv"

    results_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 65)
    print("OPTIMISATION TERMINEE")
    print("=" * 65)

    # Meilleurs résultats globaux selon F1
    best_models = (
        results_df.sort_values("f1_score", ascending=False)
        .groupby("model")
        .head(1)
        .sort_values("f1_score", ascending=False)
    )

    print("\nMeilleur seuil F1 pour chaque modèle :")
    print(
        best_models[
            ["model", "threshold", "precision", "recall",
             "f1_score", "FP", "FN"]
        ].to_string(index=False)
    )

    print(f"\nRapport CSV enregistré : {csv_path}")


if __name__ == "__main__":
    main()
