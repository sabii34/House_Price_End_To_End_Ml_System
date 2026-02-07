from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import load_yaml


@dataclass(frozen=True)
class FeatureSpec:
    numeric_features: List[str]
    categorical_features: List[str]


def infer_feature_types(df: pd.DataFrame, target: str) -> FeatureSpec:
    """
    Industry-friendly: infer feature types from dataframe dtypes.
    Later we can lock this down in config for strict schema control.
    """
    cols = [c for c in df.columns if c != target]

    numeric = []
    categorical = []

    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
        else:
            categorical.append(c)

    return FeatureSpec(numeric_features=numeric, categorical_features=categorical)


def build_preprocessor(feature_spec: FeatureSpec) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, feature_spec.numeric_features),
            ("cat", categorical_pipe, feature_spec.categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_feature_spec_from_config_or_infer(df: pd.DataFrame) -> Tuple[FeatureSpec, str]:
    """
    If config has validation.required_columns, we use that as a strict schema.
    Then we infer feature types from those columns.
    """
    cfg = load_yaml("configs/config.yaml")
    target = cfg["training"]["target"]

    required_cols = cfg["validation"]["required_columns"]
    # ensure df contains required columns
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")

    # use only required cols (clean schema)
    clean_df = df[required_cols].copy()
    spec = infer_feature_types(clean_df, target)
    return spec, target
