# anomaly_detection.py
import pandas as pd
import numpy as np
from typing import Iterable, Optional, Dict, Any, Tuple

from data_loader import load_and_preprocess_data


def _iqr_bounds(series: pd.Series, k: float = 1.5) -> Tuple[float, float, float, float, float]:
    """
    Compute IQR-based lower/upper bounds for a numeric Series.
    Returns (Q1, Q3, IQR, lower_bound, upper_bound).
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return q1, q3, iqr, lower, upper


def detect_outliers_iqr(
    df_numeric: pd.DataFrame,
    label_column: str = "Gender",
    features: Optional[Iterable[str]] = None,
    group_by_label: bool = True,
    k: float = 1.5,
) -> Dict[str, Any]:
    """
    IQR-based outlier removal on a numeric dataframe produced by your loader.
    Optionally computes bounds within each label group to avoid biased trimming.

    Parameters
    ----------
    df_numeric : pd.DataFrame
        Dataframe with numeric features and an encoded label column.
    label_column : str
        Name of the label column (e.g., "Gender" encoded 0/1).
    features : iterable[str] | None
        Which columns to check. If None, uses all numeric columns except label_column.
    group_by_label : bool
        If True, compute IQR bounds within each label group separately.
    k : float
        IQR multiplier (1.5 standard; 3.0 is more conservative).

    Returns
    -------
    dict with:
        - "df_clean": filtered dataframe (same columns as input)
        - "removed_indices": Index of removed rows
        - "summary": per-feature (and per-group) summary DataFrame
        - "params": dictionary of parameters used
    """
    if label_column not in df_numeric.columns:
        raise ValueError(f"'{label_column}' not found in dataframe")

    # Determine columns to check
    if features is None:
        cols = df_numeric.select_dtypes(include=[np.number]).columns.tolist()
        if label_column in cols:
            cols.remove(label_column)
    else:
        cols = [c for c in features if c in df_numeric.columns]

    # Masks and summaries
    keep_mask = pd.Series(True, index=df_numeric.index)
    summary_rows = []

    if group_by_label:
        # Compute per-group bounds
        for g in sorted(df_numeric[label_column].unique()):
            group_idx = df_numeric[df_numeric[label_column] == g].index
            for col in cols:
                s = df_numeric.loc[group_idx, col].dropna()
                if s.empty:
                    continue
                q1, q3, iqr, lo, hi = _iqr_bounds(s, k=k)
                col_mask = df_numeric[col].between(lo, hi) | ~df_numeric.index.isin(group_idx)
                # Only rows in this group are constrained by these bounds
                keep_mask &= col_mask

                removed = int((~df_numeric.loc[group_idx, col].between(lo, hi)).sum())
                total_g = int(len(group_idx))
                summary_rows.append({
                    "Feature": col,
                    "Group": g,
                    "Q1": q1,
                    "Q3": q3,
                    "IQR": iqr,
                    "Lower_Bound": lo,
                    "Upper_Bound": hi,
                    "Removed_in_Group": removed,
                    "Group_Size": total_g,
                    "Removed_%_in_Group": (removed / total_g * 100) if total_g > 0 else 0.0,
                })
    else:
        # Global bounds (ignoring label groups)
        for col in cols:
            s = df_numeric[col].dropna()
            if s.empty:
                continue
            q1, q3, iqr, lo, hi = _iqr_bounds(s, k=k)
            col_mask = df_numeric[col].between(lo, hi)
            keep_mask &= col_mask

            removed = int((~col_mask).sum())
            total = int(len(df_numeric))
            summary_rows.append({
                "Feature": col,
                "Group": "ALL",
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "Lower_Bound": lo,
                "Upper_Bound": hi,
                "Removed_in_Group": removed,
                "Group_Size": total,
                "Removed_%_in_Group": (removed / total * 100) if total > 0 else 0.0,
            })

    removed_indices = df_numeric.index[~keep_mask]
    df_clean = df_numeric.loc[keep_mask].copy()

    # Aggregate a compact per-feature view as well
    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["Feature", "Group"]
    ).reset_index(drop=True)

    # Console summary
    print("=== Outlier Removal (IQR) ===")
    print(f"Checked columns: {cols}")
    print(f"Grouping by label: {group_by_label} (label column = '{label_column}')")
    print(f"IQR multiplier (k): {k}")
    print(f"Original dataset size: {df_numeric.shape[0]} rows")
    print(f"Cleaned dataset size:  {df_clean.shape[0]} rows")
    print(f"Number of rows removed: {df_numeric.shape[0] - df_clean.shape[0]}")

    return {
        "df_clean": df_clean,
        "removed_indices": removed_indices,
        "summary": summary_df,
        "params": {
            "label_column": label_column,
            "features": cols,
            "group_by_label": group_by_label,
            "k": k,
        },
    }


def run_anomaly_detection(
    csv_path: str = "gym_members_exercise_tracking.csv",
    label_column: str = "Gender",
    features: Optional[Iterable[str]] = None,
    group_by_label: bool = True,
    k: float = 1.5,
    save_prefix: str = "anomalies",
) -> Dict[str, Any]:
    """
    Convenience wrapper:
    - Loads data via your loader (ensures consistent encoding)
    - Runs IQR-based detection
    - Saves cleaned data & summaries to CSV
    """
    # Load using your pipeline to keep encoding consistent
    _, _, df_numeric, gender_mapping = load_and_preprocess_data(csv_path, label_column=label_column)
    res = detect_outliers_iqr(
        df_numeric=df_numeric,
        label_column=label_column,
        features=features,
        group_by_label=group_by_label,
        k=k,
    )

    # Save outputs
    res["df_clean"].to_csv(f"{save_prefix}_cleaned.csv", index=False)
    pd.Series(res["removed_indices"]).to_csv(f"{save_prefix}_removed_indices.csv", index=False, header=["index"])
    res["summary"].to_csv(f"{save_prefix}_summary.csv", index=False)

    # Optional: report class names if available
    if gender_mapping:
        print("Label mapping:", gender_mapping)

    print(f"✓ Saved: {save_prefix}_cleaned.csv, {save_prefix}_removed_indices.csv, {save_prefix}_summary.csv")
    return res


if __name__ == "__main__":
    # Example run with defaults
    run_anomaly_detection(
        csv_path="gym_members_exercise_tracking.csv",
        label_column="Gender",
        features=None,           # or pass a list of columns to check
        group_by_label=True,     # set to False for global bounds
        k=1.5,                   # use 3.0 for a more conservative trim
        save_prefix="anomalies"
    )
