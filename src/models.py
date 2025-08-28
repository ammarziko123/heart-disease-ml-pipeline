from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def model_spaces() -> Dict[str, Any]:
    return {
        "logreg": (
            LogisticRegression(max_iter=2000),
            {
                "clf__C": [0.01, 0.1, 1, 10],
                "clf__penalty": ["l2"],
                "clf__solver": ["lbfgs", "liblinear"],
            },
        ),
        "decision_tree": (
            DecisionTreeClassifier(random_state=42),
            {
                "clf__max_depth": [None, 3, 5, 8, 12],
                "clf__min_samples_split": [2, 5, 10],
                "clf__min_samples_leaf": [1, 2, 4],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42),
            {
                "clf__n_estimators": [100, 200, 400],
                "clf__max_depth": [None, 5, 10, 15],
                "clf__min_samples_split": [2, 5, 10],
                "clf__min_samples_leaf": [1, 2, 4],
            },
        ),
        "svm": (
            SVC(probability=True, random_state=42),
            {
                "clf__C": [0.1, 1, 10],
                "clf__kernel": ["rbf", "linear"],
                "clf__gamma": ["scale", "auto"],
            },
        ),
    }
