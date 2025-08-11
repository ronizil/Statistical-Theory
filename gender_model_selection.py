import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from visualizations import save_pdf  # Function to save PDF files into figures_pdf folder

warnings.filterwarnings("ignore")


def model_selection(
    df_numeric: pd.DataFrame,
    gender_column: str = "Gender",
    gender_mapping: dict | None = None,
    save_name: str = "fig6D",
):
    """
    Port of the original Colab snippet to PyCharm:
    - Assumes df_numeric is already loaded and encoded via load_and_preprocess_data.
    - Runs 6 models (RF, LR, SVM, KNN, NB, MLP) to predict Gender.
    - Prints classification reports + Accuracy and AUC scores.
    - Saves comparative ROC plot as a PDF file (default fig6D) using visualizations.save_pdf.

    Parameters
    ----------
    df_numeric : pd.DataFrame
        Must include the numerically encoded gender column and features.
    gender_column : str
        Name of the gender column (default "Gender" as returned by loader with 0/1 encoding).
    gender_mapping : dict | None
        Mapping {0: 'Male', 1: 'Female'} or reversed (from loader).
        If None, defaults to ["Class 0", "Class 1"].
    save_name : str
        Filename (without extension) for saving the ROC plot PDF. Default is "fig6D".
    """

    # === Define feature groups exactly as in your Colab snippet ===
    behavioral = ["Workout_Frequency (days/week)", "Session_Duration (hours)",
                  "Workout_Type", "Experience_Level"]
    physiological = ["Resting_BPM", "Avg_BPM", "Max_BPM"]

    # Keep only columns present in the dataframe
    keep_cols = [c for c in behavioral + physiological if c in df_numeric.columns]
    if gender_column not in df_numeric.columns:
        raise ValueError(f"'{gender_column}' column not found in dataframe")

    # Prepare feature matrix X and target vector y
    X = df_numeric[keep_cols].copy()
    y = df_numeric[gender_column].astype(int).copy()

    # In the Colab version, get_dummies was done only on Workout_Type.
    # Here we apply it only if that column is categorical/textual;
    # if already encoded as numeric by loader, leave it as is.
    if "Workout_Type" in X.columns:
        if X["Workout_Type"].dtype == "object" or str(X["Workout_Type"].dtype).startswith("category"):
            X = pd.get_dummies(X, columns=["Workout_Type"], drop_first=True)

    # Train/test split (same as Colab: 70/30, stratified, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    # Define models exactly as in Colab
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "MLP (Neural Net)": MLPClassifier(max_iter=1000, random_state=42),
    }

    # Class names for reports
    if gender_mapping is not None and 0 in gender_mapping and 1 in gender_mapping:
        label_names = [gender_mapping[0], gender_mapping[1]]
    else:
        # Default neutral class names if no mapping is provided
        label_names = ["Class 0", "Class 1"]

    # Run models and collect results
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Get continuous scores for ROC (probabilities or decision function)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        print(f"\n{name}:\nAccuracy = {acc:.3f}, AUC = {auc:.3f}")
        print(classification_report(y_test, y_pred, target_names=label_names))

        results.append((name, acc, auc))

    # === ROC curves comparison: save as PDF (no plt.show()) ===
    plt.figure(figsize=(10, 6))
    green_palette = sns.color_palette("Greens", n_colors=len(models))

    for (name, model), color in zip(models.items(), green_palette):
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_pdf(save_name, dpi=300, close=True)  # Saves to figures_pdf/{save_name}.pdf by default

    # Optional: return results table for further use
    return pd.DataFrame(results, columns=["Model", "Accuracy", "AUC"])
