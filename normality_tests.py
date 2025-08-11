# assumptions_tests.py
import numpy as np
import pandas as pd
from scipy.stats import shapiro, levene

from data_loader import load_and_preprocess_data  # uses your existing loader


def _safe_shapiro(x: pd.Series) -> float:
    """
    Run Shapiro–Wilk with guards:
    - returns np.nan if not enough data (n < 3) or too many (n > 5000)
    - returns np.nan if the data are constant or any numerical issue occurs
    """
    x = pd.Series(x).dropna().astype(float)
    n = len(x)
    if n < 3 or n > 5000:
        return np.nan
    try:
        return float(shapiro(x).pvalue)
    except Exception:
        return np.nan


def _safe_levene(x: pd.Series, y: pd.Series) -> float:
    """
    Run Levene’s test with guards:
    - returns np.nan if either group has < 2 values
    - robust center='median' (default) is used by scipy
    """
    x = pd.Series(x).dropna().astype(float)
    y = pd.Series(y).dropna().astype(float)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    try:
        return float(levene(x, y).pvalue)
    except Exception:
        return np.nan


def normality_and_variance_checks(
    csv_path: str = "gym_members_exercise_tracking.csv",
    label_column: str = "Gender",
    features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load data via your loader and run:
      - Shapiro–Wilk normality per gender (0/1) for each feature
      - Levene equality-of-variances across genders for each feature

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    label_column : str
        Name of the numeric gender column produced by the loader (0/1).
    features : list[str] | None
        Which features to test. If None, will test all numeric columns except the label.

    Returns
    -------
    pd.DataFrame
        A table with p-values for normality by gender and Levene variance equality.
    """
    # Use your loader: returns (X_scaled, gender_encoded, df_numeric, gender_mapping)
    _, gender_encoded, df_numeric, gender_mapping = load_and_preprocess_data(
        csv_path, label_column=label_column
    )

    # df_numeric contains numeric features + the encoded label column
    if features is None:
        features = [c for c in df_numeric.columns if c != label_column]

    # Ensure label is numeric 0/1
    if df_numeric[label_column].dtype.kind not in "iu":
        df_numeric[label_column] = pd.to_numeric(df_numeric[label_column], errors="coerce").astype("Int64")

    # Split by encoded gender: assume 0/1 from the loader
    male_code = 0
    female_code = 1
    # If mapping looks reversed, still fine—this will just label columns accordingly below
    male_name = gender_mapping.get(male_code, "Male")
    female_name = gender_mapping.get(female_code, "Female")

    results = {
        "Feature": [],
        f"Normality_{male_name}_p": [],
        f"Normality_{female_name}_p": [],
        "Equal_Variance_p": [],
    }

    for feat in features:
        if feat == label_column:
            continue

        male_data = df_numeric.loc[df_numeric[label_column] == male_code, feat]
        female_data = df_numeric.loc[df_numeric[label_column] == female_code, feat]

        p_male = _safe_shapiro(male_data)
        p_female = _safe_shapiro(female_data)
        p_var = _safe_levene(male_data, female_data)

        results["Feature"].append(feat)
        results[f"Normality_{male_name}_p"].append(p_male)
        results[f"Normality_{female_name}_p"].append(p_female)
        results["Equal_Variance_p"].append(p_var)

    df_out = pd.DataFrame(results)
    # Optional: pretty formatting of p-values while keeping numeric dtype available
    # df_out_formatted = df_out.copy()
    # for col in df_out.columns[1:]:
    #     df_out_formatted[col] = df_out[col].map(lambda p: f"{p:.3e}" if pd.notnull(p) else "NaN")

    return df_out


if __name__ == "__main__":
    table = normality_and_variance_checks()
    print("\n=== Normality & Equal-Variance Checks ===")
    print(table.to_string(index=False))
