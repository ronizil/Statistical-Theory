# cluster_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, kruskal
from visualizations import save_pdf

# ✅ use KMeans-based visualization instead of the old hierarchical one
from dimensionality_reduction import kmeans_pca_visualization

# Features to exclude from the final PDF table only
EXCLUDE_FROM_TABLE = {
    "Water_Intake (liters)",
    "Max_BPM",
    "Gender",
    "Calories_Burned",
}


def _display_name(feat: str) -> str:
    """Tweak display name to avoid truncation in PDF tables."""
    text = feat.replace("_", " ")
    if text.lower().startswith("workout frequency"):
        return "Workout Frequency"
    return feat


def find_gender_dominated_clusters(cluster_labels, gender_labels, gender_mapping):
    """Identify male- and female-dominated clusters by gender ratios."""
    df = pd.DataFrame({"Cluster": cluster_labels, "Gender": gender_labels})
    gender_counts = pd.crosstab(df["Cluster"], df["Gender"], normalize='index')

    for g in gender_mapping.keys():
        if g not in gender_counts.columns:
            gender_counts[g] = 0.0

    male_code = [code for code, name in gender_mapping.items() if name.lower() == "male"]
    if not male_code:
        raise ValueError("Couldn't identify 'Male' in gender mapping.")
    male_code = male_code[0]

    gender_ratio = gender_counts[male_code]
    male_dominated = gender_ratio.idxmax()
    female_dominated = gender_ratio.idxmin()
    return male_dominated, female_dominated


def gender_distribution_by_cluster(cluster_labels, gender_labels, method_name, gender_mapping):
    """Compute gender counts and percentages per cluster and return a tidy table."""
    df = pd.DataFrame({"Cluster": cluster_labels, "Gender": gender_labels})

    counts = pd.crosstab(df["Cluster"], df["Gender"])
    percentages = pd.crosstab(df["Cluster"], df["Gender"], normalize="index") * 100

    counts.columns = [gender_mapping.get(col, str(col)) for col in counts.columns]
    percentages.columns = [gender_mapping.get(col, str(col)) for col in percentages.columns]

    combined = counts.astype(str) + " (" + percentages.round(1).astype(str) + "%)"
    combined.insert(0, "Method", method_name)

    return combined.reset_index()


def mann_whitney_between_gender_dominated_clusters(df_numeric, cluster_labels, cluster_a, cluster_b):
    """Run Mann–Whitney U between male- and female-dominated clusters for all features."""
    df = df_numeric.copy()
    df["Cluster"] = cluster_labels
    features = [col for col in df.columns if col != "Cluster"]
    results = []

    for feature in features:
        group_a = df[df["Cluster"] == cluster_a][feature]
        group_b = df[df["Cluster"] == cluster_b][feature]
        try:
            stat, p = mannwhitneyu(group_a, group_b, alternative='two-sided')
            results.append({
                "Feature": feature,
                f"Mean_Cluster_{cluster_a}": group_a.mean(),
                f"Mean_Cluster_{cluster_b}": group_b.mean(),
                "U_statistic": stat,
                "p_value": p,
                "Significant": p < 0.05
            })
        except Exception:
            continue

    df_results = pd.DataFrame(results).sort_values("p_value")
    df_results["p_value"] = df_results["p_value"].apply(lambda p: f"{p:.4f}" if p >= 0.05 else f"{p:.2e}")
    return df_results


def kruskal_test_across_clusters(df_numeric, cluster_labels):
    """Run Kruskal–Wallis across all clusters per feature."""
    df = df_numeric.copy()
    df["Cluster"] = cluster_labels
    features = [col for col in df.columns if col != "Cluster"]
    results = []

    for feature in features:
        groups = [df[df["Cluster"] == c][feature] for c in sorted(df["Cluster"].unique())]
        try:
            stat, p = kruskal(*groups)
            results.append({
                "Feature": feature,
                "Kruskal_statistic": stat,
                "p_value": p,
                "Significant": p < 0.05
            })
        except Exception:
            continue

    df_results = pd.DataFrame(results).sort_values("p_value")
    df_results["p_value"] = df_results["p_value"].apply(lambda p: f"{p:.4f}" if p >= 0.05 else f"{p:.2e}")
    return df_results


def summarize_feature_direction(df_numeric, cluster_labels, significant_features):
    """Summarize which cluster has the highest/lowest mean for each significant feature."""
    df = df_numeric.copy()
    df["Cluster"] = cluster_labels
    cluster_means = df.groupby("Cluster")[significant_features].mean()
    results = []

    for feature in significant_features:
        means = cluster_means[feature]
        top_cluster = means.idxmax()
        other_means = means.drop(index=top_cluster)
        is_high = all(means[top_cluster] > val for val in other_means)
        is_low = all(means[top_cluster] < val for val in other_means)
        direction = "Higher than others" if is_high else ("Lower than others" if is_low else "Intermediate")

        row = {
            "Feature": feature,
            "Highest in Cluster": top_cluster,
            "Direction of Effect": direction,
        }
        row.update({f"Mean_Cluster_{k}": round(v, 2) for k, v in means.items()})
        results.append(row)

    summary_df = pd.DataFrame(results)
    return summary_df


def save_direction_summary_table(df_summary, fig_name="fig5C"):
    """Save the feature-direction summary table to PDF (with some features excluded)."""
    df_summary = df_summary[~df_summary["Feature"].isin(EXCLUDE_FROM_TABLE)].copy()
    df_summary["Feature"] = df_summary["Feature"].apply(_display_name)

    max_col_len = max([len(str(x)) for x in df_summary.columns] +
                      [len(str(x)) for x in df_summary.values.flatten()]) if not df_summary.empty else 12
    fig_width = min(18, max(11, 0.22 * max_col_len))
    fig_height = 0.5 + 0.45 * max(1, len(df_summary))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    table = plt.table(
        cellText=df_summary.values,
        colLabels=df_summary.columns,
        loc='center',
        cellLoc='center',
        colLoc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.3)

    save_pdf(fig_name, dpi=300, close=True)


def run_analysis():
    """
    Main pipeline: load data, PCA + KMeans, find gender-dominated clusters,
    perform tests, summarize, and save PDF table.
    """
    from data_loader import load_and_preprocess_data

    X_scaled, gender_labels, df_numeric, gender_mapping = load_and_preprocess_data(
        "gym_members_exercise_tracking.csv", label_column="Gender"
    )

    # ✅ use KMeans-based PCA pipeline and name the method accordingly
    X_pca, pca_labels, _ = kmeans_pca_visualization(X_scaled, n_clusters=3, fig_name="fig3B")

    pca_male_cluster, pca_female_cluster = find_gender_dominated_clusters(pca_labels, gender_labels, gender_mapping)

    mann_df = mann_whitney_between_gender_dominated_clusters(
        df_numeric, pca_labels,
        cluster_a=pca_male_cluster,
        cluster_b=pca_female_cluster
    )

    kruskal_df = kruskal_test_across_clusters(df_numeric, pca_labels)

    mann_sig = mann_df[mann_df["Significant"]]["Feature"].tolist()
    kruskal_sig = kruskal_df[kruskal_df["Significant"]]["Feature"].tolist()
    combined_sig_features = sorted(set(mann_sig + kruskal_sig))

    summary_df = summarize_feature_direction(df_numeric, pca_labels, combined_sig_features)

    save_direction_summary_table(summary_df, fig_name="fig5C")
