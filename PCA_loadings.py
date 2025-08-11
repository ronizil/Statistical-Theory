# PCA_loadings.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from visualizations import save_pdf

def plot_all_pca_loadings(X_scaled, feature_names, n_components=2, prefix="fig5"):
    """
    Plots and saves loading plots of the first `n_components` PCA components.
    Saves them as fig5A.pdf, fig5B.pdf, ...
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_scaled)
    components = pca.components_

    feature_names = [str(f) for f in feature_names]
    suffix_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for i in range(n_components):
        loadings = components[i]
        sorted_idx = np.argsort(loadings)
        sorted_features = np.array(feature_names)[sorted_idx]
        sorted_loadings = loadings[sorted_idx]

        max_abs = np.max(np.abs(loadings))
        norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
        cmap = plt.cm.Blues

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(sorted_features, sorted_loadings, color=cmap(norm(sorted_loadings)))
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(f"PCA Component {i+1} Loadings", fontsize=14)
        ax.set_xlabel("Loading Value")
        ax.set_ylabel("Feature")
        plt.tight_layout()

        fig_name = f"{prefix}{suffix_letters[i]}"
        save_pdf(fig_name)
        plt.close(fig)
