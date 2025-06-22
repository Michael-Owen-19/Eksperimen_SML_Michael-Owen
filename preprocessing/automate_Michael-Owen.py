import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from joblib import dump
import os

def preprocess_data(data, target_column, save_path, file_path):
    data = data.copy()

    if 'ca' in data.columns:
        data['ca'] = data['ca'].fillna(0).astype(int)
    if 'thalach' in data.columns:
        data['thalach'] = data['thalach'].fillna(0).astype(int)

    feature_columns = data.columns.drop(target_column)
    pd.DataFrame(columns=feature_columns).to_csv(file_path, index=False)
    print(f"Nama kolom berhasil disimpan ke: {file_path}")

    int_numeric_features = data[feature_columns].select_dtypes(include=['int64']).columns.tolist()
    float_numeric_features = data[feature_columns].select_dtypes(include=['float64']).columns.tolist()
    categorical_features = data[feature_columns].select_dtypes(include=['object']).columns.tolist()

    int_numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', StandardScaler())
    ])

    float_numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num_int', int_numeric_transformer, int_numeric_features),
        ('num_float', float_numeric_transformer, float_numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Split features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

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
X_train_path = os.path.join(script_dir, 'dataset_preprocessing', 'X_train.joblib')
X_test_path = os.path.join(script_dir, 'dataset_preprocessing', 'X_test.joblib')
y_train_path = os.path.join(script_dir, 'dataset_preprocessing', 'y_train.joblib')
y_test_path = os.path.join(script_dir, 'dataset_preprocessing', 'y_test.joblib')
X_train_csv_path = os.path.join(script_dir, 'dataset_preprocessing', 'X_train.csv')
X_test_csv_path = os.path.join(script_dir, 'dataset_preprocessing', 'X_test.csv')
y_train_csv_path = os.path.join(script_dir, 'dataset_preprocessing', 'y_train.csv')
y_test_csv_path = os.path.join(script_dir, 'dataset_preprocessing', 'y_test.csv')

df = pd.read_csv(file_path)
df = df.drop(columns=['id'])

df['ca'] = df['ca'].astype('Int64')
df['thalach'] = df['thalach'].astype('Int64')

X_train, X_test, y_train, y_test = preprocess_data(df, target_column, preprocess_data_path, header_file_path)
print(X_train[2])
pd.DataFrame(X_train).to_csv(X_train_csv_path, index=False)
pd.DataFrame(X_test).to_csv(X_test_csv_path, index=False)
pd.DataFrame(y_train).to_csv(y_train_csv_path, index=False)
pd.DataFrame(y_test).to_csv(y_test_csv_path, index=False)
print("Preprocessing completed and files saved successfully.")