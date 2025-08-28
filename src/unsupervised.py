import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from .utils import save_fig, REPORTS_DIR

def run_unsupervised(X, n_clusters=3, random_state=42):
    # Scale via z-score for PCA visualization purposes
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    km_labels = kmeans.fit_predict(X_scaled)
    km_sil = silhouette_score(X_scaled, km_labels)

    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    ag_labels = agg.fit_predict(X_scaled)
    ag_sil = silhouette_score(X_scaled, ag_labels)

    # PCA for 2D plot
    pca = PCA(n_components=2, random_state=random_state)
    X2 = pca.fit_transform(X_scaled)

    # Plot clusters (KMeans)
    plt.figure()
    plt.scatter(X2[:,0], X2[:,1], c=km_labels)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title(f"KMeans (k={n_clusters}) — PCA view")
    save_fig("kmeans_pca_scatter.png")

    # Plot clusters (Agglomerative)
    plt.figure()
    plt.scatter(X2[:,0], X2[:,1], c=ag_labels)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title(f"Agglomerative (k={n_clusters}) — PCA view")
    save_fig("agglomerative_pca_scatter.png")

    summary = {
        "kmeans_silhouette": float(km_sil),
        "agglomerative_silhouette": float(ag_sil),
        "explained_variance_ratio_2d": list(map(float, pca.explained_variance_ratio_)),
    }
    return summary, km_labels, ag_labels
