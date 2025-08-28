import os, json
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.pipeline import Pipeline

from src.preprocess import load_dataset, split_features_target, make_preprocessor, persist_feature_info
from src.models import model_spaces
from src.utils import report_and_plots, learning_curve_plot, ARTIFACTS_DIR, REPORTS_DIR

DATA = os.path.join(os.path.dirname(__file__), "data", "heart.csv")
FEATURE_INFO = os.path.join(ARTIFACTS_DIR, "feature_info.json")
BEST_MODEL = os.path.join(ARTIFACTS_DIR, "best_supervised_pipeline.joblib")

def main():
    df = load_dataset(DATA)
    X_train, X_test, y_train, y_test = split_features_target(df)

    # Persist feature info for Streamlit
    persist_feature_info(X_train, FEATURE_INFO)

    # Build base preprocessor (with KBest & PCA knobs that we tune)
    pre = make_preprocessor(X_train)

    results = []
    best_score = -np.inf
    best_name = None
    best_pipe = None

    for name, (clf, grid) in model_spaces().items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])

        # Combine model grid with optional selectors
        search_space = {}
        for k, v in grid.items():
            search_space[k] = v
        # Try a couple PCA sizes and KBest sizes
        search_space.update({
            "pre__select_kbest__k": [None, 8, 10, 12],
            "pre__pca__n_components": [None, 5, 8, 10],
        })

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        gs = GridSearchCV(pipe, param_grid=search_space, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)

        # Evaluate on holdout
        y_proba = gs.predict_proba(X_test)[:,1] if hasattr(gs.best_estimator_["clf"], "predict_proba") else None
        y_pred = gs.predict(X_test)

        report_and_plots(y_test, y_proba, y_pred, model_name=name)

        # Learning curve
        sizes, tr, te = learning_curve(gs.best_estimator_, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1,
                                       train_sizes=np.linspace(0.2, 1.0, 5), shuffle=True, random_state=42)
        learning_curve_plot(sizes, tr, te, model_name=name)

        results.append({
            "model": name,
            "best_params": gs.best_params_,
            "cv_best_score": gs.best_score_,
        })

        if gs.best_score_ > best_score:
            best_score = gs.best_score_
            best_name = name
            best_pipe = gs.best_estimator_

    # Persist the best
    dump(best_pipe, BEST_MODEL)
    with open(os.path.join(ARTIFACTS_DIR, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "best_cv_auc": best_score, "results": results}, f, indent=2)

    print(f"Best model: {best_name} (CV ROC-AUC={best_score:.3f})")
    print(f"Saved pipeline => {BEST_MODEL}")
    print(f"Reports => {REPORTS_DIR}")

if __name__ == "__main__":
    main()
