# spearman_correlation.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from visualizations import save_pdf

def plot_spearman_correlation_heatmap(df: pd.DataFrame, fig_name: str = "fig2") -> str:
    """
    Plot and save Spearman correlation heatmap for numeric features.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame after anomaly removal (must include numeric columns and 'Gender').
    fig_name : str
        Base filename (without extension) to save the PDF, default 'fig2'.

    Returns
    -------
    str
        The path of the saved PDF (figures_pdf/<fig_name>.pdf)
    """
    # Select numeric columns (exclude label if present)
    exclude_cols = {"Gender"}
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found to compute Spearman correlation.")

    numeric_df = df[numeric_cols]

    # Compute Spearman correlation matrix
    corr = numeric_df.corr(method='spearman')

    # Define custom diverging colormap
    colors = [(0.0, "#e5f5e0"), (0.5, "white"), (1.0, "#31a354")]
    custom_cmap = LinearSegmentedColormap.from_list("blaugrana_diverging", colors)

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=custom_cmap,
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Spearman Correlation"},
        annot_kws={"size": 10}
    )
    plt.title("Spearman Correlation Heatmap", fontsize=14)
    plt.tight_layout()

    # Save as PDF explicitly as fig2 (or whatever fig_name is)
    save_pdf(fig_name)  # -> figures_pdf/<fig_name>.pdf
    plt.close()

    saved_path = f"figures_pdf/{fig_name}.pdf"
    print(f"✓ Spearman heatmap saved to: {saved_path}")
    return saved_path
