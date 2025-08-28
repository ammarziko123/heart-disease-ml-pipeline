# Heart Disease UCI — End‑to‑End ML Pipeline

A complete, reproducible machine‑learning pipeline to analyze, predict, and visualize heart disease risk.

## ✨ What’s inside
- **Data preprocessing & cleaning** (missing values, encoding, scaling)
- **Feature selection**: Chi‑Square, ANOVA (f_classif), and **RFE**
- **Dimensionality reduction**: **PCA**
- **Supervised models**: Logistic Regression, Decision Tree, Random Forest, SVM
- **Unsupervised**: K‑Means & Hierarchical (Agglomerative) clustering
- **Hyperparameter tuning**: GridSearchCV & RandomizedSearchCV
- **Evaluation**: accuracy, precision/recall/F1, ROC‑AUC, confusion matrix, learning curves
- **Deployment**: Streamlit app (bonus), optional Ngrok tunnel (bonus)
- **Reproducible outputs**: trained pipeline + reports in `./artifacts`

> **Dataset**: Uses the commonly used, pre‑processed *Heart Disease* CSV (Kaggle/various mirrors) with columns like `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target` where `target` is 1 for disease, 0 otherwise.

## 🗂 Project structure
```
heart-disease-ml-pipeline/
├─ data/
│  └─ heart.csv                # Place dataset here (or run the downloader)
├─ src/
│  ├─ preprocess.py            # Cleaning, encoding, scaling, PCA, feature selection
│  ├─ models.py                # Model spaces, search grids, training helpers
│  ├─ unsupervised.py          # KMeans & Agglomerative + plots & metrics
│  ├─ utils.py                 # Utilities (plots, reports, I/O)
├─ artifacts/
│  ├─ best_supervised_pipeline.joblib
│  ├─ feature_info.json
│  └─ reports/                 # Auto-generated metrics and plots
├─ train.py                    # Orchestrates supervised training + tuning
├─ cluster.py                  # Runs unsupervised analysis & figures
├─ app.py                      # Streamlit UI for live inference
├─ download_data.py            # Convenience script to fetch heart.csv
├─ requirements.txt
└─ README.md
```

## ⬇️ Get the data
Option A — Put `heart.csv` into `./data` yourself.

Option B — Let the script try common mirrors:
```bash
python download_data.py
```

## ▶️ Train supervised models
```bash
pip install -r requirements.txt

# Train + tune (saves best pipeline to ./artifacts)
python train.py
```

Outputs in `./artifacts`:
- `best_supervised_pipeline.joblib` — the **fitted Pipeline** (preprocessing + model)
- `feature_info.json` — column metadata (dtypes, order, required inputs)
- `reports/` — classification report, confusion matrix, ROC curve, learning curves

## 🔍 Unsupervised exploration
```bash
python cluster.py
```
Generates cluster labels, silhouette score, and 2D PCA plots into `./artifacts/reports`.

## 🖥 Streamlit app (bonus)
```bash
streamlit run app.py
```
- Loads `artifacts/best_supervised_pipeline.joblib` and `feature_info.json`
- Web form for feature inputs
- Predicts probability & class, shows interpretation hints

### 🌐 Ngrok (bonus)
1) Install ngrok and get a token from ngrok.com.
2) In a second terminal:
```bash
ngrok http 8501
```
3) Share the forwarding URL.

## 🧪 Notes
- Reproducibility: random seeds are fixed.
- The Streamlit app hides technical columns (like one‑hot vectors); it asks for human‑readable inputs and maps them internally.
- The training code uses sensible defaults but is easy to extend (e.g., XGBoost, LightGBM).

## 📦 Requirements
See `requirements.txt`.
