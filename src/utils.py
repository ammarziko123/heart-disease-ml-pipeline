import os, json, textwrap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
                             classification_report)

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
REPORTS_DIR = os.path.join(ARTIFACTS_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_fig(name):
    path = os.path.join(REPORTS_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path

def report_and_plots(y_true, y_proba, y_pred, model_name="model"):
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    save_json(report, os.path.join(REPORTS_DIR, f"{model_name}_classification_report.json"))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    save_fig(f"{model_name}_confusion_matrix.png")

    # ROC curve (if probabilities provided)
    if y_proba is not None:
        RocCurveDisplay.from_predictions(y_true, y_proba)
        save_fig(f"{model_name}_roc_curve.png")

def learning_curve_plot(train_sizes, train_scores, test_scores, model_name="model"):
    import numpy as np
    import matplotlib.pyplot as plt

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, marker="o", label="Training score")
    plt.plot(train_sizes, test_mean, marker="s", label="Validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title(f"Learning Curve — {model_name}")
    plt.legend()
    return save_fig(f"{model_name}_learning_curve.png")
