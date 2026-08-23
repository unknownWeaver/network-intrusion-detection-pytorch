"""Preprocessing for NSL-KDD: binary target + categorical/numeric transforms.

Transformers (encoder, scaler) are always fit on the training split only,
then applied unchanged to validation/test - fitting on data a model will
later be evaluated on is a leak, even if it "just" shifts a mean.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from data import load_train, load_test
from sklearn.model_selection import train_test_split
from data import CATEGORICAL_FEATURES

DEAD_COLUMNS = ["num_outbound_cmds"]
NON_FEATURE_COLUMNS = ["difficulty_level"]


def make_binary_target(df: pd.DataFrame) -> pd.Series:
    return (df["label"] != "normal").astype(int).rename("is_attack")


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = make_binary_target(df)
    X = df.drop(columns=["label", *NON_FEATURE_COLUMNS, *DEAD_COLUMNS])
    return X, y


def fit_transformers(X_train: pd.DataFrame) -> dict:
    numeric_cols = [c for c in X_train.columns if c not in CATEGORICAL_FEATURES]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[CATEGORICAL_FEATURES])

    scaler = StandardScaler()
    scaler.fit(X_train[numeric_cols])

    return {
        "encoder": encoder,
        "scaler": scaler,
        "categorical_cols": CATEGORICAL_FEATURES,
        "numeric_cols": numeric_cols,
    }


def transform(X: pd.DataFrame, fitted: dict) -> np.ndarray:
    encoded = fitted["encoder"].transform(X[fitted["categorical_cols"]])
    scaled = fitted["scaler"].transform(X[fitted["numeric_cols"]])
    return np.concatenate([scaled, encoded], axis=1).astype(np.float32)


def prepare_data(random_state: int = 42):
    train_df = load_train()
    X, y = split_features_target(train_df)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    fitted = fit_transformers(X_train)
    X_train = transform(X_train, fitted)
    X_val = transform(X_val, fitted)

    test_df = load_test()
    X_test, y_test = split_features_target(test_df)
    X_test = transform(X_test, fitted)

    return X_train, y_train, X_val, y_val, X_test, y_test