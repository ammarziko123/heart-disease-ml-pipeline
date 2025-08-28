import os, json
import pandas as pd
from src.preprocess import load_dataset, split_features_target
from src.unsupervised import run_unsupervised
from src.utils import ARTIFACTS_DIR

DATA = os.path.join(os.path.dirname(__file__), "data", "heart.csv")

def main():
    df = load_dataset(DATA)
    X = df.drop(columns=["target"])
    summary, km_labels, ag_labels = run_unsupervised(X, n_clusters=3)

    out = {
        "kmeans_labels_sample": km_labels[:20].tolist(),
        "agglomerative_labels_sample": ag_labels[:20].tolist(),
        **summary,
    }
    with open(os.path.join(ARTIFACTS_DIR, "unsupervised_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Unsupervised analysis complete. See artifacts/reports for plots.")

if __name__ == "__main__":
    main()
