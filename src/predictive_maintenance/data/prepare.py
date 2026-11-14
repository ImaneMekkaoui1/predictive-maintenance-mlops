
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Chemins du projet
PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Paramètres
TARGET = "Machine failure"
TEST_SIZE = 0.2
RANDOM_STATE = 42

COLUMNS_TO_DROP = [
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

CATEGORICAL_FEATURES = ["Type"]


def prepare_data():

    # 1. Chargement
    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Dataset initial : {df.shape}")

    # 2. Nettoyage
    df = df.drop(columns=COLUMNS_TO_DROP)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 3. Séparation Train / Test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 4. Prétraitement
    numeric_features = [
        col for col in X_train.columns
        if col not in CATEGORICAL_FEATURES
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
        ]
    )

    # Fit uniquement sur Train
    X_train_encoded = preprocessor.fit_transform(X_train)

    # Transform sur Test
    X_test_encoded = preprocessor.transform(X_test)

    # Noms des colonnes après encodage
    categorical_names = (
        preprocessor
        .named_transformers_["categorical"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
        .tolist()
    )

    feature_names = categorical_names + numeric_features

    # Conversion en DataFrame
    if hasattr(X_train_encoded, "toarray"):
        X_train_encoded = X_train_encoded.toarray()

    if hasattr(X_test_encoded, "toarray"):
        X_test_encoded = X_test_encoded.toarray()

    X_train = pd.DataFrame(
        X_train_encoded,
        columns=feature_names,
    )

    X_test = pd.DataFrame(
        X_test_encoded,
        columns=feature_names,
    )

    # 5. Sauvegarde
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)

    y_train.to_frame(name=TARGET).to_csv(
        PROCESSED_DIR / "y_train.csv",
        index=False,
    )

    y_test.to_frame(name=TARGET).to_csv(
        PROCESSED_DIR / "y_test.csv",
        index=False,
    )

    print("\nPréparation terminée avec succès !")
    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")

    print("\nDistribution des défaillances dans Train :")
    print(y_train.value_counts(normalize=True))

    print("\nDistribution des défaillances dans Test :")
    print(y_test.value_counts(normalize=True))


if __name__ == "__main__":
    prepare_data()
