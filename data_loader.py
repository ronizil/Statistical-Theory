import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(filepath: str, label_column: str):
    df = pd.read_csv(filepath)

    if label_column not in df.columns:
        raise ValueError(f"'{label_column}' column not found in dataframe")

    # Preserve original gender column (text)
    gender_labels_text = df[label_column].astype(str)

    # Encode gender to numeric
    gender_encoder = LabelEncoder()
    gender_encoded = gender_encoder.fit_transform(gender_labels_text)

    # Save mapping: 0 -> Male, 1 -> Female (or vice versa)
    gender_mapping = dict(zip(gender_encoded, gender_encoder.classes_))

    # Drop label column and encode the rest
    features = df.drop(columns=[label_column]).copy()
    for col in features.columns:
        if features[col].dtype == 'object' or features[col].dtype.name == 'category':
            features[col] = LabelEncoder().fit_transform(features[col].astype(str))

    # Standardize features
    X_scaled = StandardScaler().fit_transform(features)

    # Return encoded data
    df_numeric = features.copy()
    df_numeric[label_column] = gender_encoded

    return X_scaled, gender_encoded, df_numeric, gender_mapping
