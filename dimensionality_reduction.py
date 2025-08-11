import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
from visualizations import save_pdf

# Custom palette (extend if you use more clusters)
custom_colors = [
    "#1f77b4", "#2ca02c", "#17becf", "#98df8a",
    "#aec7e8", "#006400", "#004c6d"
]


def kmeans_umap_visualization(X_scaled, n_clusters=3, fig_name="fig3"):
    """
    PCA -> KMeans clustering -> UMAP visualization.
    Returns: (X_umap, labels, silhouette_score)
    """
    # 1) Reduce to 2D with PCA (before UMAP)
    X_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    # 2) KMeans in PCA space
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)

    # 3) UMAP on PCA data
    X_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(X_pca)

    # 4) Silhouette (computed on UMAP space)
    sil = silhouette_score(X_umap, labels)
    print(f"[UMAP] Silhouette Score: {sil:.4f}")

    # 5) Plot
    plt.figure(figsize=(8, 6))
    for cluster_id in range(n_clusters):
        color = custom_colors[cluster_id % len(custom_colors)]
        plt.scatter(
            X_umap[labels == cluster_id, 0],
            X_umap[labels == cluster_id, 1],
            s=40, alpha=0.85, color=color, label=f"Cluster {cluster_id}"
        )

    # No title here
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Cluster", fontsize=10, loc='best', frameon=True)
    plt.tight_layout()
    save_pdf(fig_name)
    plt.close()

    return X_umap, labels, sil


def kmeans_pca_visualization(X_scaled, n_clusters=3, fig_name="fig3B"):
    """
    PCA -> KMeans clustering -> PCA scatter.
    Returns: (X_pca, labels, silhouette_score)
    """
    # 1) PCA
    X_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    # 2) KMeans in PCA space
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)

    # 3) Silhouette (on PCA space)
    sil = silhouette_score(X_pca, labels)
    print(f"[PCA] Silhouette Score: {sil:.4f}")

    # 4) Plot
    plt.figure(figsize=(8, 6))
    for cluster_id in range(n_clusters):
        color = custom_colors[cluster_id % len(custom_colors)]
        plt.scatter(
            X_pca[labels == cluster_id, 0],
            X_pca[labels == cluster_id, 1],
            s=40, alpha=0.85, color=color, label=f"Cluster {cluster_id}"
        )

    # No title here
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Cluster", fontsize=10, loc='best', frameon=True)
    plt.tight_layout()
    save_pdf(fig_name)
    plt.close()

    return X_pca, labels, sil
