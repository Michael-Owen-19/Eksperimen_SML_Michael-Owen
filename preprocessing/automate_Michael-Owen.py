import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from joblib import dump
import os

def mark_0_as_nan(X):
        X = X.copy()
        X[(X == 0) | (pd.isna(X))] = np.nan
        return X

def preprocess_data(data, target_column, save_path, file_path):
    # Copy data to avoid modifying original
    data = data.copy()

    # === Step 4 & 5: Replace missing values for ca and thalach with 0 and cast to int ===
    if 'ca' in data.columns:
        data['ca'] = data['ca'].fillna(0).astype(int)
    if 'thalach' in data.columns:
        data['thalach'] = data['thalach'].fillna(0).astype(int)

    # Save column names (excluding target) as CSV header
    feature_columns = data.columns.drop(target_column)
    pd.DataFrame(columns=feature_columns).to_csv(file_path, index=False)
    print(f"Nama kolom berhasil disimpan ke: {file_path}")

    # Identify numeric and categorical columns
    numeric_features = data[feature_columns].select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = data[feature_columns].select_dtypes(include=['object']).columns.tolist()

    # === Step 2 & 3: Custom imputers for trestbps and chol (treat both NaN and 0 as missing) ===

    special_numeric_imputer = Pipeline(steps=[
        ('zero_to_nan', FunctionTransformer(mark_0_as_nan, validate=False)),
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Separate special columns
    special_numeric_features = []
    for col in ['trestbps', 'chol']:
        if col in numeric_features:
            special_numeric_features.append(col)
            numeric_features.remove(col)

    # Normal numeric pipeline
    normal_numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Column transformer combining all
    preprocessor = ColumnTransformer(transformers=[
        ('num_normal', normal_numeric_transformer, numeric_features),
        ('num_special', special_numeric_imputer, special_numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Split features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fit-transform and transform
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Save pipeline
    dump(preprocessor, save_path)

    return X_train, X_test, y_train, y_test

target_column = 'num'
script_dir = os.path.dirname(os.path.abspath(__file__))

dataset_folder_path = os.path.join(script_dir, 'dataset_preprocessing')

if not os.path.exists(dataset_folder_path):
    os.makedirs(dataset_folder_path)

preprocess_data_path = os.path.join(script_dir, 'dataset_preprocessing', 'preprocessor_pipeline.joblib')
header_file_path = os.path.join(script_dir, 'dataset_preprocessing', 'header_data.csv')
file_path = os.path.join(script_dir, '..', 'dataset', 'heart_disease_uci_raw.csv')
df = pd.read_csv(file_path)
df = df.drop(columns=['id'])

X_train, X_test, y_train, y_test = preprocess_data(df, target_column, preprocess_data_path, header_file_path)
print(X_train[2])