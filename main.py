# main.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

import os
os.makedirs("figures_pdf", exist_ok=True)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data & utilities
from data_loader import load_and_preprocess_data
from visualizations import save_pdf

# Anomaly detection (run cleaning right after load)
from anomaly_detection import detect_outliers_iqr

# Clustering visuals & stats
from clustering import (
    elbow_plot,
    kmeans_loss_heatmap,
    kmeans_silhouette_heatmap,
    gmm_silhouette_matrix,
    dbscan_coerced_heatmap,
    agglomerative_heatmap,
)
from cluster_analysis import (
    gender_distribution_by_cluster,
    mann_whitney_between_gender_dominated_clusters,
    kruskal_test_across_clusters,
    find_gender_dominated_clusters,
    summarize_feature_direction,
    save_direction_summary_table,
)
from analysis import (
    cluster_statistical_analysis,
    visualize_pvalue_table,
)
from dimensionality_reduction import kmeans_pca_visualization
from PCA_loadings import plot_all_pca_loadings  # <-- needed for fig5A/B

# Inference & tests
from gender_feature_stats import mann_whitney_by_gender
from gender_model_selection import model_selection               # saves fig6D
from gender_prediction_model_analysis import gender_prediction_model_analysis  # saves fig6A/B/C


def _normality_variance_on_df(
    df_numeric: pd.DataFrame,
    label_column: str = "Gender",
    features: list[str] | None = None
) -> pd.DataFrame:
    """
    Run Shapiro (per group) and Levene (between groups) on the CLEANED dataframe.
    Returns a tidy DataFrame with p-values (printed only, not saved).
    """
    from scipy.stats import shapiro, levene

    # Default feature set (fallback to all numeric except label)
    if features is None:
        candidates = [
            'Calories_Burned', 'BMI', 'Fat_Percentage', 'Resting_BPM',
            'Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM',
            'Session_Duration (hours)', 'Workout_Frequency (days/week)'
        ]
        features = [c for c in candidates if c in df_numeric.columns]
        if not features:  # fallback
            features = [c for c in df_numeric.select_dtypes(include=[np.number]).columns
                        if c != label_column]

    # Groups coded 0/1 (as produced by the loader)
    male_data = df_numeric[df_numeric[label_column] == 0]
    female_data = df_numeric[df_numeric[label_column] == 1]

    rows = []
    for feat in features:
        x_m = male_data[feat].dropna()
        x_f = female_data[feat].dropna()

        # Shapiro on each group (guard against tiny samples)
        p_m = shapiro(x_m).pvalue if len(x_m) >= 3 else np.nan
        p_f = shapiro(x_f).pvalue if len(x_f) >= 3 else np.nan

        # Levene between groups (guard against tiny samples)
        p_var = levene(x_m, x_f).pvalue if (len(x_m) >= 2 and len(x_f) >= 2) else np.nan

        rows.append({
            "Feature": feat,
            "Normality_Male_p": p_m,
            "Normality_Female_p": p_f,
            "Equal_Variance_p": p_var
        })

    df_assump = pd.DataFrame(rows)
    # Pretty printing
    with pd.option_context('display.float_format', lambda v: f"{v:.4g}"):
        print(df_assump)
    return df_assump


def run_all():
    # 0) Load raw data (original)
    X_scaled_raw, gender_labels_raw, df_numeric_raw, gender_mapping = load_and_preprocess_data(
        "gym_members_exercise_tracking.csv", label_column="Gender"
    )

    # 1) Anomaly cleaning (IQR) — everything else runs on CLEAN data
    print("\n=== Outlier Removal (IQR) on loaded dataframe ===")
    clean_res = detect_outliers_iqr(
        df_numeric=df_numeric_raw,
        label_column="Gender",
        features=None,          # or pass a custom list
        group_by_label=True,    # per-gender bounds to avoid bias
        k=1.5,                  # 1.5 (standard); use 3.0 for conservative trimming
    )
    df_numeric = clean_res["df_clean"]          # CLEAN dataframe
    removed_idx = clean_res["removed_indices"]  # Index of removed rows
    print(clean_res["summary"].to_string(index=False))  # print only

    # Recompute X_scaled and labels to match the cleaned rows
    features_only = df_numeric.drop(columns=["Gender"])
    X_scaled = StandardScaler().fit_transform(features_only)
    gender_labels = df_numeric["Gender"].to_numpy()

    print(f"Original rows: {len(df_numeric_raw)} | After cleaning: {len(df_numeric)} | Removed: {len(removed_idx)}")

    # 2) Normality & variance checks on CLEAN data (print only)
    print("\n=== Normality & Variance Test Results (CLEAN data) ===")
    assumptions_df = _normality_variance_on_df(
        df_numeric=df_numeric,
        label_column="Gender",
        features=None
    )

    # 3) Mann–Whitney U per feature vs gender (CLEAN)
    print("\n=== Mann-Whitney by Gender (CLEAN) ===")
    mwu_results = mann_whitney_by_gender(df_numeric, gender_column="Gender")
    print(mwu_results)

    # 4) Random Forest (behavioral vs combined) — saves 6A/6B/6C (CLEAN)
    print("\n=== Gender Prediction Models (RF) on CLEAN data ===")
    rf_summary = gender_prediction_model_analysis(
        df_numeric=df_numeric,
        gender_column="Gender",
        gender_mapping=gender_mapping,
        n_boot=500,
        save_name_confusion="fig6A",
        save_name_importance="fig6B",
        save_name_roc="fig6C"
    )
    print(rf_summary)

    # 5) Model selection benchmark (6 models) — saves 6D (CLEAN)
    print("\n=== Model Selection for Gender Prediction (CLEAN) ===")
    model_selection_results = model_selection(
        df_numeric=df_numeric,
        gender_column="Gender",
        gender_mapping=gender_mapping,
        save_name="fig6D"
    )
    print(model_selection_results)

    # 6) Clustering evaluation visuals (CLEAN)
    X_pca_2d = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    elbow_plot(X_pca_2d)
    kmeans_loss_heatmap(X_scaled)
    kmeans_silhouette_heatmap(X_scaled)
    gmm_silhouette_matrix(X_scaled)
    dbscan_coerced_heatmap(X_scaled)
    agglomerative_heatmap(X_scaled)

    # 7) Gender–cluster association tests (CLEAN)
    print("\n=== Cluster-Gender Statistical Test Results (CLEAN) ===")
    df_stats = cluster_statistical_analysis(X_scaled, gender_labels)
    print(df_stats)
    visualize_pvalue_table(df_stats, fig_name="Table_1")

    # 8) PCA loadings (CLEAN)
    feature_names = features_only.columns.tolist()
    plot_all_pca_loadings(X_scaled, feature_names, n_components=2, prefix="fig5")

    # 9) PCA + KMeans clustering (silhouette printed inside)
    X_pca, pca_labels, sil_pca = kmeans_pca_visualization(
        X_scaled, n_clusters=3, fig_name="fig3B"
    )

    # 10) Gender distribution by cluster — PRINT + PDF (fig3C)
    pca_gender_df = gender_distribution_by_cluster(
        pca_labels, gender_labels, "PCA + KMeans", gender_mapping
    )
    print("\n=== Gender distribution by cluster (counts and %) ===")
    print(pca_gender_df.to_string(index=False))

    # Save the distribution table as a PDF figure (fig3C)
    fig, ax = plt.subplots(figsize=(8.5, 0.6 + 0.4 * len(pca_gender_df)))
    ax.axis('off')
    tbl = ax.table(
        cellText=pca_gender_df.values,
        colLabels=pca_gender_df.columns,
        loc='center',
        cellLoc='center',
        colLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.2)
    fig.tight_layout()
    save_pdf("fig3C", dpi=300, close=True)

    # 11) Feature-level tests across clusters + direction summary (saved as fig5C)
    male_cluster, female_cluster = find_gender_dominated_clusters(pca_labels, gender_labels, gender_mapping)
    mann_df = mann_whitney_between_gender_dominated_clusters(
        df_numeric, pca_labels, cluster_a=male_cluster, cluster_b=female_cluster
    )
    kruskal_df = kruskal_test_across_clusters(df_numeric, pca_labels)
    sig_features = sorted(set(
        mann_df[mann_df["Significant"]]["Feature"].tolist() +
        kruskal_df[kruskal_df["Significant"]]["Feature"].tolist()
    ))
    direction_df = summarize_feature_direction(df_numeric, pca_labels, sig_features)
    save_direction_summary_table(direction_df, fig_name="fig5C")

    return {
        "assumptions": assumptions_df,
        "pca": {
            "gender_distribution": pca_gender_df,
            "mannwhitney": mann_df,
            "kruskal": kruskal_df,
            "direction_summary": direction_df
        },
        "gender_stats": df_stats,
        "mwu_by_gender": mwu_results,
        "rf_summary": rf_summary,
        "model_selection": model_selection_results
    }


if __name__ == "__main__":
    results = run_all()

    print("\n=== Kruskal-Wallis Test Results ===")
    print(results["pca"]["kruskal"])

    print("\n=== Mann-Whitney by Gender Results ===")
    print(results["mwu_by_gender"])

    print("\n=== Gender Prediction Results ===")
    print(results["rf_summary"])

    print("\n=== Model Selection Results ===")
    print(results["model_selection"])

    print("\n=== Gender distribution by cluster (summary) ===")
    print(results["pca"]["gender_distribution"].to_string(index=False))
