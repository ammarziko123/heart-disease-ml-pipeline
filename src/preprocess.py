import os, json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
 
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Expected: ensure 'target' present
    if "target" not in df.columns:
        raise ValueError("Expected 'target' column in dataset.")

    # Basic cleaning: numeric coercion
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df

def split_features_target(df, target="target", test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def infer_feature_types(X):
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical

def make_preprocessor(X, n_pca=None, k_best=None, rfe=False, rfe_estimator=None):
    numeric, categorical = infer_feature_types(X)

    # Transformers
    num_tf = Pipeline(steps=[("scaler", StandardScaler())])
    cat_tf = Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore"))])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_tf, numeric),
            ("cat", cat_tf, categorical),
        ]
    )

    steps = [("pre", pre)]

    if k_best is not None and k_best > 0:
        # Chi2 requires non-negative features; after OHE + StandardScaler numeric may be negative.
        # We'll run f_classif which works on standardized data, and fall back to chi2 only on non-negative.
        steps.append(("select_kbest", SelectKBest(score_func=f_classif, k=k_best)))

    if rfe:
        base = rfe_estimator if rfe_estimator is not None else LogisticRegression(max_iter=500)
        steps.append(("rfe", RFE(estimator=base, n_features_to_select=max(5, int(0.5)))))

    if n_pca is not None and n_pca > 0:
        steps.append(("pca", PCA(n_components=n_pca, random_state=42)))

    return Pipeline(steps=steps)

def persist_feature_info(X, path):
    numeric, categorical = infer_feature_types(X)
    info = {
        "feature_order": X.columns.tolist(),
        "numeric": numeric,
        "categorical": categorical,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
