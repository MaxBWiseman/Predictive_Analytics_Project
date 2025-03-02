import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Replace 'Very High' with 10 in 'Stress Level' and convert columns to numeric
class DataTypeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Replace 'Very High' with 10 in 'Stress Level'
        X['Stress Level'] = X['Stress Level'].replace('Very High', 10)

        # Convert columns to numeric, handling non-numeric values
        X['Sleep Duration (hours)'] = pd.to_numeric(X['Sleep Duration (hours)'], errors='coerce')
        X['Stress Level'] = pd.to_numeric(X['Stress Level'], errors='coerce')

        return X


# Impute missing values (not for Activity Level)
from sklearn.impute import KNNImputer

class KNNImputerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.imputer = KNNImputer(n_neighbors=self.n_neighbors)

    def fit(self, X, y=None):
        # Identify numerical columns
        self.num_cols = X.select_dtypes(include=[np.number]).columns
        self.imputer.fit(X[self.num_cols])
        return self

    def transform(self, X):
        X = X.copy()
        # Impute numerical columns
        X[self.num_cols] = self.imputer.transform(X[self.num_cols])
        return X


# Handle Outliers
from feature_engine.outliers import Winsorizer

class WinsorizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.winsorizers = {}

    def fit(self, X, y=None):
        X = X.copy()
        # Define and fit Winsorizers for each variable
        self.winsorizers['Sleep Duration (hours)'] = Winsorizer(
            capping_method='iqr', tail='both', fold=1.5, variables=['Sleep Duration (hours)']
        ).fit(X)
        self.winsorizers['Heart Rate (BPM)'] = Winsorizer(
            capping_method='iqr', tail='right', fold=1.5, variables=['Heart Rate (BPM)']
        ).fit(X)
        self.winsorizers['Blood Oxygen Level (%)'] = Winsorizer(
            capping_method='iqr', tail='left', fold=1.5, variables=['Blood Oxygen Level (%)']
        ).fit(X)
        self.winsorizers['Step Count'] = Winsorizer(
            capping_method='iqr', tail='right', fold=1.5, variables=['Step Count']
        ).fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        # Apply Winsorizers
        for winsorizer in self.winsorizers.values():
            X = winsorizer.transform(X)
        return X


class FloatToInt(BaseEstimator, TransformerMixin):
    def __init__(self, columns=['Stress Level']):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Convert specified columns to integer
        for col in self.columns:
            X[col] = X[col].astype(int)
        return X

# Predict missing values in 'Activity Level'
from sklearn.ensemble import RandomForestClassifier

class CategoricalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column='Activity Level', random_state=42):
        self.target_column = target_column
        self.random_state = random_state
        self.classifier = RandomForestClassifier(random_state=self.random_state)
        self.original_categories = None

    def fit(self, X, y=None):
        X = X.copy()
        # Save original categories
        X[self.target_column] = X[self.target_column].astype('category')
        self.original_categories = X[self.target_column].cat.categories

        # Encode target column
        X[self.target_column] = X[self.target_column].cat.codes.replace(-1, np.nan)

        # Split data
        self.train_data = X.dropna(subset=[self.target_column])
        self.test_data = X[X[self.target_column].isna()]

        # Features and target
        self.X_train = self.train_data.drop(columns=[self.target_column])
        self.y_train = self.train_data[self.target_column].astype(int)

        # Fit classifier
        self.classifier.fit(self.X_train, self.y_train)
        return self

    def transform(self, X):
        X = X.copy()
        # Encode target column
        X[self.target_column] = X[self.target_column].astype('category')
        X[self.target_column] = X[self.target_column].cat.codes.replace(-1, np.nan)

        # Identify missing values
        missing_mask = X[self.target_column].isna()
        if missing_mask.any():
            X_missing = X[missing_mask]
            X_missing_features = X_missing.drop(columns=[self.target_column])
            # Predict missing values
            X.loc[missing_mask, self.target_column] = self.classifier.predict(X_missing_features)

        # Convert to int and decode categories
        X[self.target_column] = X[self.target_column].astype(int)
        X[self.target_column] = pd.Categorical.from_codes(
            X[self.target_column], categories=self.original_categories
        )
        return X


# Correct mis-spelled values in Activity Level
class CategoryCorrector(BaseEstimator, TransformerMixin):
    def __init__(self, column='Activity Level'):
        self.column = column
        self.class_mapping = {
            'Seddentary': 'Sedentary',
            'Highly_Active': 'Highly Active',
            'Actve': 'Active'
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.column] = X[self.column].replace(self.class_mapping)
        return X


# Smooth data using K-Nearest Neighbors
from sklearn.neighbors import NearestNeighbors

class DataSmoother(BaseEstimator, TransformerMixin):
    def __init__(self, k=3):
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=self.k)

    def fit(self, X, y=None):
        X = X.copy()
        self.num_cols = X.select_dtypes(include=[np.number]).columns
        self.nn.fit(X[self.num_cols])
        return self

    def transform(self, X):
        X = X.copy()
        distances, indices = self.nn.kneighbors(X[self.num_cols])

        # Smooth numerical columns
        for i, col in enumerate(self.num_cols):
            X[col] = [np.mean(X.iloc[indices[row_idx]][col]) for row_idx in range(len(X))]
        return X


# Trim outliers again after smoothing
from feature_engine.outliers import OutlierTrimmer

class OutlierTrimmerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.trimmers = {}

    def fit(self, X, y=None):
        X = X.copy()
        # Define and fit Outlier Trimmers for each variable
        self.trimmers['Heart Rate (BPM)'] = OutlierTrimmer(
            capping_method='quantiles', tail='right', fold=0.05, variables=['Heart Rate (BPM)']
        ).fit(X)
        self.trimmers['Blood Oxygen Level (%)'] = OutlierTrimmer(
            capping_method='quantiles', tail='left', fold=0.05, variables=['Blood Oxygen Level (%)']
        ).fit(X)
        self.trimmers['Sleep Duration (hours)'] = OutlierTrimmer(
            capping_method='quantiles', tail='both', fold=0.05, variables=['Sleep Duration (hours)']
        ).fit(X)
        self.trimmers['Step Count'] = OutlierTrimmer(
            capping_method='quantiles', tail='right', fold=0.05, variables=['Step Count']
        ).fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        # Apply Outlier Trimmers sequentially
        for trimmer in self.trimmers.values():
            X = trimmer.transform(X)
        return X

# This is important else even the categoric column will be minmax scaled resulting in much different components when evaluated with a PCA
from sklearn.preprocessing import MinMaxScaler , StandardScaler , RobustScaler
class DataFrameScaler(BaseEstimator, TransformerMixin):
    def __init__(self, exclude_columns=None):
        self.exclude_columns = exclude_columns
        self.scaler = RobustScaler()
    
    def fit(self, X, y=None):
        X_to_scale = X.drop(columns=self.exclude_columns)
        self.scaler.fit(X_to_scale)
        return self
    
    def transform(self, X):
        X_to_scale = X.drop(columns=self.exclude_columns)
        X_excluded = X[self.exclude_columns]
        X_scaled = pd.DataFrame(self.scaler.transform(X_to_scale), columns=X_to_scale.columns)
        X_final = pd.concat([X_scaled, X_excluded.reset_index(drop=True)], axis=1)
        return X_final


from sklearn.pipeline import Pipeline
# encoding
from feature_engine.encoding import OrdinalEncoder 

def CleanDataPipeline():
    cleaning_engineering_pipeline = Pipeline([
        ('data_type_transformer', DataTypeTransformer()),
        ('knn_imputer', KNNImputerTransformer(n_neighbors=3)),
        ('winsorizer_transformer', WinsorizerTransformer()),
        ('categorical_imputer', CategoricalImputer()),
        ('category_corrector', CategoryCorrector()),
        ('data_smoother', DataSmoother(k=3)),
        ('outlier_trimmer_transformer', OutlierTrimmerTransformer()),
        ('encoder', OrdinalEncoder(encoding_method='arbitrary', variables=["Activity Level"])),
        ('scaler', DataFrameScaler(exclude_columns=["Activity Level", "Stress Level"])),
        ('float_to_int', FloatToInt()),
        ])
    return cleaning_engineering_pipeline

import plotly.express as px

def load_data():
    df = pd.read_csv("src/unclean_smartwatch_health_data.csv")
    df = df.drop("User ID", axis=1)
    pipeline = CleanDataPipeline()
    df = pipeline.fit_transform(df)
    return df

def load_pkl_file(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)