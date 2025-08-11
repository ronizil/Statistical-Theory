import pandas as pd
from scipy import stats


def mann_whitney_by_gender(df: pd.DataFrame, gender_column: str = "Gender"):
    """
    Perform a Mann-Whitney U test for each numeric feature, split by gender.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing numeric features and a numeric gender column (0/1).
    gender_column : str
        Name of the gender column (default: "Gender").

    Returns
    -------
    pd.DataFrame
        Table with feature names and p-values.
    """
    results = []

    # Loop over all numeric columns except the gender column
    for col in df.select_dtypes(include=[float, int]).columns:
        if col == gender_column:
            continue

        # Split data by gender (assuming 1 = Female, 0 = Male)
        group_female = df[df[gender_column] == 1][col].dropna()
        group_male = df[df[gender_column] == 0][col].dropna()

        # Only run test if both groups have enough samples
        if len(group_female) > 10 and len(group_male) > 10:
            stat, p = stats.mannwhitneyu(group_female, group_male)
            results.append({"Feature": col, "p_value": p})

    return pd.DataFrame(results)
