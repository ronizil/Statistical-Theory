import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier

from visualizations import save_pdf


def _maybe_one_hot_workout_type(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode Workout_Type only if it is categorical/text."""
    if "Workout_Type" in X.columns:
        if X["Workout_Type"].dtype == "object" or str(X["Workout_Type"].dtype).startswith("category"):
            X = pd.get_dummies(X, columns=["Workout_Type"], drop_first=True)
    return X


def _bootstrap_roc_ci(y_true, y_score, n_boot=500, seed=42, grid_size=201):
    """
    Bootstrap 95% CI bands for ROC. Returns:
    fpr_grid, mean_tpr_on_grid, lower_band, upper_band, (auc_lo, auc_hi)
    Note: AUC shown in the legend will be the empirical test AUC (not the bootstrap mean),
    so it matches your other evaluation code.
    """
    rng = np.random.default_rng(seed)
    fpr_grid = np.linspace(0, 1, grid_size)
    tprs = np.zeros((n_boot, grid_size))
    aucs = np.zeros(n_boot)

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        fpr, tpr, _ = roc_curve(yt, ys)
        aucs[b] = roc_auc_score(yt, ys)
        tpr_interp = np.interp(fpr_grid, fpr, tpr, left=0.0, right=1.0)
        tpr_interp[0] = 0.0
        tprs[b] = tpr_interp

    mean_tpr = tprs.mean(axis=0)
    lo = np.quantile(tprs, 0.025, axis=0)
    hi = np.quantile(tprs, 0.975, axis=0)
    auc_lo = np.quantile(aucs, 0.025)
    auc_hi = np.quantile(aucs, 0.975)
    return fpr_grid, mean_tpr, lo, hi, (auc_lo, auc_hi)


def gender_prediction_model_analysis(
    df_numeric: pd.DataFrame,
    gender_column: str = "Gender",
    gender_mapping: dict | None = None,
    n_boot: int = 500,
    save_name_confusion: str = "fig6A",
    save_name_importance: str = "fig6B",
    save_name_roc: str = "fig6C",
):
    """RF: behavioral vs combined → save fig6A (confusions), fig6B (combined importances), fig6C (ROC+95% CI)."""

    # feature sets
    behavioral = ["Workout_Frequency (days/week)", "Session_Duration (hours)", "Workout_Type", "Experience_Level"]
    physiological = ["Resting_BPM", "Avg_BPM", "Max_BPM"]

    # keep only available columns
    cols_needed = [gender_column] + behavioral + physiological
    cols_present = [c for c in cols_needed if c in df_numeric.columns]
    df = df_numeric[cols_present].dropna().copy()

    if gender_column not in df.columns:
        raise ValueError(f"{gender_column} not found in dataframe")

    # labels for confusion matrices
    if gender_mapping is not None and 0 in gender_mapping and 1 in gender_mapping:
        label_names = [gender_mapping[0], gender_mapping[1]]
    else:
        label_names = ["Class 0", "Class 1"]

    y = df[gender_column].astype(int).copy()

    # build X_b (behavioral only) and X_c (combined)
    X_b = df.drop(columns=[gender_column] + [c for c in physiological if c in df.columns]).copy()
    X_c = df.drop(columns=[gender_column]).copy()

    X_b = _maybe_one_hot_workout_type(X_b)
    X_c = _maybe_one_hot_workout_type(X_c)

    # one synchronized split
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(idx, test_size=0.3, stratify=y, random_state=42)
    Xb_train, Xb_test = X_b.iloc[idx_train], X_b.iloc[idx_test]
    Xc_train, Xc_test = X_c.iloc[idx_train], X_c.iloc[idx_test]
    yb_train, yb_test = y.iloc[idx_train], y.iloc[idx_test]
    yc_train, yc_test = y.iloc[idx_train], y.iloc[idx_test]

    # train RF models
    model_b = RandomForestClassifier(random_state=42)
    model_b.fit(Xb_train, yb_train)

    model_c = RandomForestClassifier(random_state=42)
    model_c.fit(Xc_train, yc_train)

    # predictions and reports
    yb_pred = model_b.predict(Xb_test)
    yc_pred = model_c.predict(Xc_test)

    acc_b = accuracy_score(yb_test, yb_pred)
    acc_c = accuracy_score(yc_test, yc_pred)
    print(f"Accuracy (Behavioral): {acc_b:.2%}")
    print(f"Accuracy (Behavioral + Physiological): {acc_c:.2%}\n")

    print("=== Classification Report (Behavioral) ===")
    print(classification_report(yb_test, yb_pred, target_names=label_names, zero_division=0))
    print("\n=== Classification Report (Combined) ===")
    print(classification_report(yc_test, yc_pred, target_names=label_names, zero_division=0))

    # === fig6A: confusion matrices (two panels) ===
    fig_cm, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_b = confusion_matrix(yb_test, yb_pred)
    cm_c = confusion_matrix(yc_test, yc_pred)
    sns.heatmap(cm_b, annot=True, fmt='d',
                xticklabels=label_names, yticklabels=label_names,
                ax=axes[0], cmap="Blues")
    axes[0].set_title("Confusion Matrix - Behavioral")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    sns.heatmap(cm_c, annot=True, fmt='d',
                xticklabels=label_names, yticklabels=label_names,
                ax=axes[1], cmap="Greens")
    axes[1].set_title("Confusion Matrix - Combined")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
    fig_cm.tight_layout()
    save_pdf(save_name_confusion, dpi=300, close=True)

    # === fig6B: ONLY combined feature importance ===
    imp_c = model_c.feature_importances_
    order_c = np.argsort(imp_c)[::-1]
    fig_imp_c, axc = plt.subplots(figsize=(8, 5))
    sns.barplot(x=imp_c[order_c], y=np.array(X_c.columns)[order_c], ax=axc, palette="Greens")
    axc.set_title("Feature Importance - Combined")
    fig_imp_c.tight_layout()
    save_pdf(save_name_importance, dpi=300, close=True)  # saves as figures_pdf/fig6B.pdf

    # === fig6C: ROC with 95% CI (step lines) ===
    yb_proba = model_b.predict_proba(Xb_test)[:, 1]
    yc_proba = model_c.predict_proba(Xc_test)[:, 1]

    # CI bands from bootstrap
    fpr_b_grid, mean_tpr_b, lo_b, hi_b, (auc_lo_b, auc_hi_b) = _bootstrap_roc_ci(
        yb_test, yb_proba, n_boot=n_boot, seed=42
    )
    fpr_c_grid, mean_tpr_c, lo_c, hi_c, (auc_lo_c, auc_hi_c) = _bootstrap_roc_ci(
        yc_test, yc_proba, n_boot=n_boot, seed=43
    )

    # empirical ROC (step) and empirical AUC to match your other code
    fpr_b_emp, tpr_b_emp, _ = roc_curve(yb_test, yb_proba)
    fpr_c_emp, tpr_c_emp, _ = roc_curve(yc_test, yc_proba)
    auc_b_emp = roc_auc_score(yb_test, yb_proba)
    auc_c_emp = roc_auc_score(yc_test, yc_proba)

    fig_roc, axr = plt.subplots(figsize=(8, 5))
    axr.step(fpr_b_emp, tpr_b_emp, where="post",
             label=f"Behavioral (AUC={auc_b_emp:.3f} [{auc_lo_b:.3f},{auc_hi_b:.3f}])",
             color="blue")
    axr.step(fpr_c_emp, tpr_c_emp, where="post",
             label=f"Combined (AUC={auc_c_emp:.3f} [{auc_lo_c:.3f},{auc_hi_c:.3f}])",
             color="green")
    axr.fill_between(fpr_b_grid, lo_b, hi_b, color="blue", alpha=0.15, linewidth=0)
    axr.fill_between(fpr_c_grid, lo_c, hi_c, color="green", alpha=0.15, linewidth=0)
    axr.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axr.set_xlabel("False Positive Rate")
    axr.set_ylabel("True Positive Rate")
    axr.set_title("ROC Curve Comparison (95% CI)")
    axr.legend()
    axr.grid(True)
    fig_roc.tight_layout()
    save_pdf(save_name_roc, dpi=300, close=True)

    return {
        "accuracy_behavioral": acc_b,
        "accuracy_combined": acc_c,
        "model_behavioral": model_b,
        "model_combined": model_c
    }
